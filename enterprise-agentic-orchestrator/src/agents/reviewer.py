"""
Reviewer Agent
===============
CrewAI-based independent reviewer agent that validates the analyst's
credit risk assessment by performing its own data retrieval, running
all stress test scenarios, and producing a Pydantic-validated ReviewReport
with agree/disagree determination, confidence scoring, and issues found.
"""

import logging
import time
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tasks.task_output import TaskOutput

from src.agents.base import AgentResponse, BaseAgent
from src.models.reports import ConfidenceLevel, ReviewReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt template (AGNT-06 rules)
# ---------------------------------------------------------------------------

REVIEWER_SYSTEM_TEMPLATE = """You are {role}.
{backstory}

Your goal: {goal}

MANDATORY RULES — violations will cause your output to be rejected:

1. INDEPENDENT RETRIEVAL: You MUST independently retrieve source data using
   your own tool calls. NEVER reuse the analyst's retrieval or trust the
   analyst's figures at face value.

2. STRESS TESTING: You MUST run ALL 5 stress test scenarios using the
   stress_tester tool. Skipping stress tests will cause rejection.

3. COMPARISON: Compare your independent findings against the analyst's
   report. Document every discrepancy you find.

4. ANOMALY DETECTION: Flag any inconsistencies, unsupported claims, or
   anomalies in the analyst's work in the issues_found list.

5. CONFIDENCE CALIBRATION: Set confidence_level to HIGH only if your
   independent analysis fully agrees with the analyst's assessment.
   Use MEDIUM if minor differences exist. Use LOW if significant
   discrepancies are found.

6. AGREEMENT DETERMINATION: If the analyst's figures do not match your
   independent tool results, you MUST set agrees_with_analyst=false
   and explain the discrepancies in your reasoning.

7. OUTPUT FORMAT: Your output MUST be valid JSON conforming to the
   ReviewReport schema with fields: report_id, application_id,
   analyst_report_id, agrees_with_analyst, confidence_level,
   quality_score, stress_test_results, issues_found, reasoning.

8. ERROR HANDLING: If a tool returns an error or no data, state
   "insufficient data" for that aspect rather than guessing.
"""


# ---------------------------------------------------------------------------
# Guardrail function
# ---------------------------------------------------------------------------


def validate_review_output(result: TaskOutput) -> tuple[bool, Any]:
    """Validate the reviewer agent's output meets governance requirements.

    Checks:
    - Pydantic model is present (output_pydantic parsed successfully)
    - stress_test_results is non-empty (reviewer must run stress tests)
    - quality_score is between 0.0 and 1.0
    - reasoning is non-empty
    - analyst_report_id is non-empty

    Returns:
        (True, result) on success, (False, error_message) on failure.
    """
    try:
        if result.pydantic is None:
            return (False, "Output could not be parsed into ReviewReport")

        report = result.pydantic

        if not report.stress_test_results:
            return (
                False,
                "stress_test_results is empty — reviewer must run all 5 stress test scenarios",
            )

        if not (0.0 <= report.quality_score <= 1.0):
            return (
                False,
                f"quality_score {report.quality_score} is outside valid range 0.0-1.0",
            )

        if not report.reasoning or not report.reasoning.strip():
            return (False, "reasoning field is empty — review must include reasoning")

        if not report.analyst_report_id or not report.analyst_report_id.strip():
            return (
                False,
                "analyst_report_id is empty — review must reference the analyst report",
            )

        return (True, result)

    except Exception as e:
        return (False, f"Validation error: {e}")


# ---------------------------------------------------------------------------
# ReviewerAgent
# ---------------------------------------------------------------------------


class ReviewerAgent(BaseAgent):
    """Independent risk reviewer agent backed by a single-agent CrewAI Crew.

    Validates the analyst's credit risk assessment by independently retrieving
    source data, running stress tests, and flagging discrepancies. Produces a
    Pydantic-validated ReviewReport with agree/disagree and confidence scoring.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(
            name="independent_reviewer",
            role="Independent Risk Reviewer",
            framework="crewai",
            config=config,
        )
        # LLM created lazily in _get_llm() so instantiation does not
        # require provider SDKs (e.g. azure-ai-inference) at import time.
        self._llm: LLM | None = None

    def _get_llm(self) -> LLM:
        """Lazily create the LLM instance on first use."""
        if self._llm is None:
            self._llm = LLM(
                model=self.config.get("model", "azure/gpt-4"),
                temperature=0.2,  # AGNT-04: slightly higher than analyst for diversity
                max_tokens=4000,
            )
        return self._llm

    async def execute(
        self, context: dict, tools: list | None = None
    ) -> AgentResponse:
        """Run the independent review crew and return an AgentResponse.

        Args:
            context: Must contain ``context["application"]`` with loan details
                     and ``context["analyst_report"]`` with the analyst's output.
            tools: Optional list of CrewAI BaseTool instances.  If *None*,
                   the default reviewer tool-set is loaded lazily.

        Returns:
            AgentResponse with the ReviewReport as output dict.
        """
        start = time.perf_counter()

        # Lazy-load default tools only when caller does not supply them
        if tools is None:
            from src.agents.tools_adapter import get_reviewer_tools

            tools = get_reviewer_tools()

        analyst_report = context.get("analyst_report", {})

        crewai_agent = Agent(
            role="Independent Risk Reviewer",
            goal=(
                "Independently validate the analyst's credit risk assessment. "
                "Retrieve your own data, run stress tests, and flag any "
                "discrepancies or unsupported claims."
            ),
            backstory=(
                "You are a senior risk reviewer at a UK financial institution. "
                "Your job is to independently verify analyst assessments — not "
                "rubber-stamp them. You have caught significant errors in past "
                "reviews. You trust only your own tool results, never the "
                "analyst's figures at face value."
            ),
            tools=tools,
            llm=self._get_llm(),
            # AGNT-04: max_iter=3 may be tight for 5+ tools — may need
            # increase if reviewer consistently hits iteration limit
            max_iter=3,
            verbose=self.config.get("verbose", False),
            allow_delegation=False,
            system_template=REVIEWER_SYSTEM_TEMPLATE,
        )

        task = Task(
            description=self._build_task_description(context, analyst_report),
            expected_output=(
                "A complete ReviewReport JSON with agrees_with_analyst, "
                "confidence_level, quality_score, stress_test_results (all 5 "
                "scenarios), issues_found, and reasoning. Base your assessment "
                "on YOUR independent tool results, not the analyst's."
            ),
            agent=crewai_agent,
            output_pydantic=ReviewReport,
            guardrail=validate_review_output,
            max_retries=2,
        )

        crew = Crew(
            agents=[crewai_agent],
            tasks=[task],
            process=Process.sequential,
            verbose=self.config.get("verbose", False),
        )

        result = await crew.akickoff(
            inputs={
                "application_id": context["application"]["application_id"],
            }
        )

        report: ReviewReport = result.pydantic

        # Defensive token extraction
        try:
            tokens = result.token_usage.total_tokens if result.token_usage else 0
        except Exception:
            tokens = 0

        latency = (time.perf_counter() - start) * 1000  # ms

        return AgentResponse(
            agent_name=self.name,
            agent_framework=self.framework,
            output=report.model_dump(mode="json"),
            reasoning_trace=report.reasoning,
            confidence=self._map_confidence_level(report.confidence_level),
            sources_used=[],
            tokens_used=tokens,
            latency_ms=latency,
        )

    def _build_task_description(
        self, context: dict, analyst_report: dict
    ) -> str:
        """Build a structured task prompt from the application and analyst report."""
        app = context["application"]
        applicant = app.get("applicant", {})

        application_id = app.get("application_id", "unknown")
        company_name = applicant.get("company_name", "unknown")
        sector = app.get("sector", "unknown")
        loan_amount = app.get("loan_amount", "unknown")
        currency = app.get("currency", "GBP")
        purpose = app.get("purpose", "unknown")
        years_trading = app.get("years_trading", "unknown")

        # Extract analyst summary for comparison
        analyst_credit_score = analyst_report.get("credit_score", "N/A")
        analyst_recommendation = analyst_report.get("recommendation", "N/A")
        analyst_pd = "N/A"
        risk_metrics = analyst_report.get("risk_metrics", {})
        if isinstance(risk_metrics, dict):
            analyst_pd = risk_metrics.get("probability_of_default", "N/A")
        analyst_report_id = analyst_report.get("report_id", "unknown")

        return f"""Independently review the analyst's credit risk assessment for the following loan application.

APPLICATION DETAILS:
- Application ID: {application_id}
- Company: {company_name}
- Sector: {sector}
- Loan Amount: {currency} {loan_amount}
- Purpose: {purpose}
- Years Trading: {years_trading}

ANALYST'S REPORT (for comparison only — do NOT trust these figures):
- Analyst Report ID: {analyst_report_id}
- Credit Score: {analyst_credit_score}
- Recommendation: {analyst_recommendation}
- Probability of Default: {analyst_pd}

REQUIRED STEPS (you must complete ALL of these):

1. INDEPENDENTLY use rag_financial_lookup to retrieve source data for
   "{company_name}" (do NOT reuse analyst data).
2. INDEPENDENTLY use rag_sector_analysis for a "{sector}" sector assessment.
3. Run stress_tester with the retrieved financial data across all 5 stress
   scenarios to assess borrower resilience.
4. Use credit_scorer independently to verify the analyst's credit score.
5. Use risk_calculator independently to verify the analyst's risk metrics.
6. Compare YOUR results against the analyst's report above.
7. Flag any discrepancies, unsupported claims, or anomalies in issues_found.

If your independent analysis differs from the analyst, set
agrees_with_analyst=false and explain all discrepancies in your reasoning.
Set analyst_report_id to "{analyst_report_id}"."""

    def _map_confidence_level(self, level: ConfidenceLevel) -> float:
        """Map ConfidenceLevel enum to a float for AgentResponse.confidence.

        HIGH=0.9, MEDIUM=0.6, LOW=0.3.
        """
        mapping = {
            ConfidenceLevel.HIGH: 0.9,
            ConfidenceLevel.MEDIUM: 0.6,
            ConfidenceLevel.LOW: 0.3,
        }
        return mapping.get(level, 0.5)
