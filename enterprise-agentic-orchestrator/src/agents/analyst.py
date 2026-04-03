"""
Analyst Agent
==============
CrewAI-based credit analyst agent that produces a Pydantic-validated
AnalysisReport from a loan application.  Uses a single-agent Crew with
system-template enforcement, output_pydantic validation, and a guardrail
function to ensure grounded, evidence-based output.
"""

import logging
import time
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tasks.task_output import TaskOutput

from src.agents.base import AgentResponse, BaseAgent
from src.models.reports import AnalysisReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt template (AGNT-03 rules)
# ---------------------------------------------------------------------------

ANALYST_SYSTEM_TEMPLATE = """You are {role}.
{backstory}

Your goal: {goal}

MANDATORY RULES — violations will cause your output to be rejected:

1. GROUNDING: Every numerical claim MUST come from a tool result. Never
   invent figures. If you do not have a tool result for a value, state
   "insufficient data" rather than guessing.

2. CITATIONS: Every claim MUST cite its source using the citation from
   tool output.  Include the tool name and key data point in each
   source_citation entry.

3. PROTECTED CHARACTERISTICS: You MUST NOT use or reference protected
   characteristics in your analysis: age, gender, ethnicity, religion,
   disability, sexual orientation, marital status, or pregnancy.

4. OUTPUT FORMAT: Your output MUST be valid JSON conforming to the
   AnalysisReport schema with fields: report_id, application_id,
   credit_score, risk_metrics, sector_outlook, recommendation,
   reasoning, source_citations.

5. TOOL USAGE: You MUST use ALL of these tools during your analysis:
   credit_scorer, risk_calculator, rag_financial_lookup,
   rag_sector_analysis, historical_comparator.  Do not skip any tool.

6. ERROR HANDLING: If a tool returns an error or no data, state
   "insufficient data" for that section rather than guessing.
"""


# ---------------------------------------------------------------------------
# Guardrail function
# ---------------------------------------------------------------------------


def validate_analysis_output(result: TaskOutput) -> tuple[bool, Any]:
    """Validate the analyst agent's output meets governance requirements.

    Checks:
    - Pydantic model is present (output_pydantic parsed successfully)
    - source_citations is non-empty (grounding requirement)
    - credit_score is within 0-100 range
    - risk_metrics has probability_of_default >= 0
    - reasoning is non-empty

    Returns:
        (True, result) on success, (False, error_message) on failure.
    """
    try:
        if result.pydantic is None:
            return (False, "Output could not be parsed into AnalysisReport")

        report = result.pydantic

        if not report.source_citations:
            return (False, "source_citations is empty — every claim must cite its source")

        if not (0 <= report.credit_score <= 100):
            return (False, f"credit_score {report.credit_score} is outside valid range 0-100")

        if report.risk_metrics.probability_of_default < 0:
            return (
                False,
                f"probability_of_default {report.risk_metrics.probability_of_default} "
                f"must be >= 0",
            )

        if not report.reasoning or not report.reasoning.strip():
            return (False, "reasoning field is empty — analysis must include reasoning")

        return (True, result)

    except Exception as e:
        return (False, f"Validation error: {e}")


# ---------------------------------------------------------------------------
# AnalystAgent
# ---------------------------------------------------------------------------


class AnalystAgent(BaseAgent):
    """Credit analyst agent backed by a single-agent CrewAI Crew.

    Produces an AnalysisReport from a loan application context by invoking
    all five required analytical tools and applying guardrail validation.
    """

    def __init__(self, config: dict) -> None:
        super().__init__(
            name="financial_analyst",
            role="Financial Analyst",
            framework="crewai",
            config=config,
        )
        self.llm = LLM(
            model=config.get("model", "azure/gpt-4"),
            temperature=0.1,
            max_tokens=4000,
        )

    async def execute(
        self, context: dict, tools: list | None = None
    ) -> AgentResponse:
        """Run the credit analysis crew and return an AgentResponse.

        Args:
            context: Must contain ``context["application"]`` with loan details.
            tools: Optional list of CrewAI BaseTool instances.  If *None*,
                   the default analyst tool-set is loaded lazily.

        Returns:
            AgentResponse with the AnalysisReport as output dict.
        """
        start = time.perf_counter()

        # Lazy-load default tools only when caller does not supply them
        if tools is None:
            from src.agents.tools_adapter import get_analyst_tools

            tools = get_analyst_tools()

        crewai_agent = Agent(
            role="Senior Financial Analyst",
            goal=(
                "Produce a thorough, evidence-based credit risk analysis for "
                "the given loan application. Use every required tool and cite "
                "all sources."
            ),
            backstory=(
                "You are an experienced credit risk analyst at a UK financial "
                "institution. You have 15 years of experience assessing SME "
                "lending applications. You never guess -- every figure comes "
                "from your analytical tools and data sources."
            ),
            tools=tools,
            llm=self.llm,
            max_iter=5,  # AGNT-01: may be tight for complex applications
            verbose=self.config.get("verbose", False),
            allow_delegation=False,
            system_template=ANALYST_SYSTEM_TEMPLATE,
        )

        task = Task(
            description=self._build_task_description(context),
            expected_output=(
                "A complete AnalysisReport JSON with credit_score, risk_metrics, "
                "sector_outlook, recommendation, reasoning, and source_citations. "
                "Every numerical value must come from a tool result."
            ),
            agent=crewai_agent,
            output_pydantic=AnalysisReport,
            guardrail=validate_analysis_output,
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

        report: AnalysisReport = result.pydantic

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
            confidence=self._derive_confidence(report),
            sources_used=report.source_citations,
            tokens_used=tokens,
            latency_ms=latency,
        )

    def _build_task_description(self, context: dict) -> str:
        """Build a structured task prompt from the loan application context."""
        app = context["application"]
        applicant = app.get("applicant", {})

        application_id = app.get("application_id", "unknown")
        company_name = applicant.get("company_name", "unknown")
        sector = app.get("sector", "unknown")
        loan_amount = app.get("loan_amount", "unknown")
        currency = app.get("currency", "GBP")
        purpose = app.get("purpose", "unknown")
        years_trading = app.get("years_trading", "unknown")
        ccj_count = app.get("ccj_count", 0)

        return f"""Analyse the following loan application and produce a complete AnalysisReport.

APPLICATION DETAILS:
- Application ID: {application_id}
- Company: {company_name}
- Sector: {sector}
- Loan Amount: {currency} {loan_amount}
- Purpose: {purpose}
- Years Trading: {years_trading}
- CCJ Count: {ccj_count}

REQUIRED STEPS (you must complete ALL of these):

1. Use rag_financial_lookup to retrieve the applicant's financial documents
   and key financial metrics.
2. Use rag_sector_analysis to assess current conditions and outlook for the
   "{sector}" sector.
3. Use credit_scorer with the retrieved financial data to calculate a credit
   score for this applicant.
4. Use risk_calculator with the credit score to compute PD, LGD, EAD, and
   expected loss risk metrics.
5. Use historical_comparator to find similar past lending decisions for
   comparison and precedent.

Produce an AnalysisReport with ALL fields populated from tool results.
Every source_citation must include the tool name and key data point.
Do NOT invent any figures — every number must come from a tool result."""

    def _derive_confidence(self, report: AnalysisReport) -> float:
        """Heuristic confidence score based on output completeness.

        Starts at 0.8, with deductions for incomplete output signals.
        """
        confidence = 0.8

        if len(report.source_citations) < 3:
            confidence -= 0.1

        if len(report.reasoning) < 100:
            confidence -= 0.1

        if not report.sector_outlook or report.sector_outlook.lower() == "unknown":
            confidence -= 0.1

        return max(0.0, min(1.0, confidence))
