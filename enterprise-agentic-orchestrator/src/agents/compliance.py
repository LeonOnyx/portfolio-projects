"""
Compliance Agent
================
AutoGen-based regulatory compliance agent that executes 5 sequential
UK regulatory checks and produces a Pydantic-validated ComplianceReport.
Wraps AutoGen 0.4's AssistantAgent as a BaseAgent subclass with lazy
imports for all AutoGen modules.
"""

import json
import logging
import time
from typing import Any

from src.agents.base import AgentResponse, BaseAgent
from src.models.reports import ComplianceReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt (AGNT-09 rules)
# ---------------------------------------------------------------------------

COMPLIANCE_SYSTEM_PROMPT = """You are a Regulatory Compliance Officer.

You verify lending decisions against UK regulations. You execute exactly 5 checks
in this order, each citing the specific regulation:

1. CONSUMER DUTY CHECK - FCA Consumer Duty (PS22/9):
   Verify the decision considers customer outcomes and fair value.

2. FAIR LENDING CHECK - Equality Act 2010 / FCA PRIN 2.1:
   Verify no protected characteristics influence the decision.

3. RISK APPETITE CHECK - Internal Risk Appetite Framework:
   Verify the exposure is within approved risk parameters.

4. CONCENTRATION CHECK - CRR Article 395 / Large Exposures:
   Use the concentration_checker tool to verify portfolio limits.

5. DOCUMENTATION CHECK - FCA SYSC 9.1 / Record Keeping:
   Verify all required documentation is present and complete.

MANDATORY RULES:
- A SINGLE check failure means overall_passed = false
- You MUST NOT assess credit quality -- only regulatory compliance
- You MUST cite the specific regulation for each check
- You MUST use rag_policy_lookup to retrieve the relevant regulation text
- You MUST use concentration_checker for the concentration check
- Write in plain English suitable for regulatory submission
"""


# ---------------------------------------------------------------------------
# ComplianceAgent
# ---------------------------------------------------------------------------


class ComplianceAgent(BaseAgent):
    """Regulatory compliance agent backed by AutoGen's AssistantAgent.

    Executes 5 sequential regulatory checks (Consumer Duty, Fair Lending,
    Risk Appetite, Concentration, Documentation) and produces a
    ComplianceReport where a single check failure means overall failure.

    A fresh AssistantAgent is created per execute() call to avoid
    state pollution across requests (research pitfall 2).
    """

    def __init__(self, config: dict) -> None:
        super().__init__(
            name="compliance_officer",
            role="Regulatory Compliance Officer",
            framework="autogen",
            config=config,
        )
        # Model client created lazily in _get_model_client() so
        # instantiation does not require autogen-ext at import time.
        self._model_client = None
        # AGNT-07: cap conversation turns to prevent runaway agent loops
        self._max_turns = config.get("max_turns", 3)

    def _get_model_client(self):
        """Lazily create the OpenAI model client on first use.

        Temperature is fixed at 0.0 per AGNT-07 for deterministic
        compliance checks.
        """
        if self._model_client is None:
            from autogen_ext.models.openai import OpenAIChatCompletionClient

            self._model_client = OpenAIChatCompletionClient(
                model=self.config.get("llm_model", "gpt-4o"),
                temperature=0.0,
            )
        return self._model_client

    async def execute(
        self, context: dict, tools: list | None = None
    ) -> AgentResponse:
        """Run the compliance checks and return an AgentResponse.

        Creates a fresh AutoGen AssistantAgent per call (stateful agent
        pattern -- research pitfall 2). Passes max_tool_iterations to
        cap the agent's tool-use rounds per AGNT-07.

        Args:
            context: Must contain application data, analyst recommendation,
                and reviewer assessment for compliance verification.
            tools: Optional list of AutoGen FunctionTool instances.  If
                *None*, the default compliance tool-set is loaded lazily.

        Returns:
            AgentResponse with the ComplianceReport as output dict.
        """
        start = time.perf_counter()

        # Lazy-load default tools only when caller does not supply them
        if tools is None:
            from src.agents.tools_adapter import get_compliance_tools_autogen

            tools = get_compliance_tools_autogen()

        try:
            # Lazy import -- no module-level autogen dependency
            from autogen_agentchat.agents import AssistantAgent
            from autogen_agentchat.messages import StructuredMessage

            # Fresh agent per call to avoid state pollution
            agent = AssistantAgent(
                name="compliance_officer",
                model_client=self._get_model_client(),
                tools=tools,
                system_message=COMPLIANCE_SYSTEM_PROMPT,
                output_content_type=ComplianceReport,
                reflect_on_tool_use=True,
                max_tool_iterations=self._max_turns,
            )

            task_str = self._build_task(context)
            result = await agent.run(task=task_str)

            # Extract ComplianceReport from result messages
            report = self._extract_report(result, StructuredMessage)

            latency = (time.perf_counter() - start) * 1000

            # Build reasoning trace from message history
            reasoning_trace = self._extract_reasoning(result)

            # Extract sources from tool call results
            sources_used = self._extract_sources(result)

            # Estimate token usage from model usage metadata
            tokens_used = self._extract_tokens(result)

            return AgentResponse(
                agent_name=self.name,
                agent_framework=self.framework,
                output=report.model_dump(mode="json"),
                reasoning_trace=reasoning_trace,
                confidence=1.0 if report.overall_passed else 0.0,
                sources_used=sources_used,
                tokens_used=tokens_used,
                latency_ms=latency,
            )

        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            logger.error(
                "[%s] Compliance check failed: %s", self.name, exc, exc_info=True
            )
            return AgentResponse(
                agent_name=self.name,
                agent_framework=self.framework,
                output={},
                reasoning_trace=f"Error: {exc}",
                confidence=0.0,
                sources_used=[],
                tokens_used=0,
                latency_ms=latency,
            )

    def _build_task(self, context: dict) -> str:
        """Build a structured task prompt from the application context."""
        app = context.get("application", {})
        applicant = app.get("applicant", {})

        application_id = app.get("application_id", "unknown")
        company_name = applicant.get("company_name", "unknown")
        sector = app.get("sector", "unknown")
        loan_amount = app.get("loan_amount", "unknown")
        currency = app.get("currency", "GBP")

        # Extract analyst and reviewer assessments if available
        analyst = context.get("analyst_report", {})
        analyst_recommendation = analyst.get("recommendation", "unknown")
        analyst_reasoning = analyst.get("reasoning", "not provided")

        reviewer = context.get("reviewer_report", {})
        reviewer_agrees = reviewer.get("agrees_with_analyst", "unknown")
        reviewer_reasoning = reviewer.get("reasoning", "not provided")

        # Portfolio context for concentration check
        portfolio = context.get("portfolio", {})
        portfolio_total = portfolio.get("total", 10000000)
        exposures_by_name = json.dumps(
            portfolio.get("exposures_by_name", {}), default=str
        )
        exposures_by_sector = json.dumps(
            portfolio.get("exposures_by_sector", {}), default=str
        )

        return f"""Perform a full regulatory compliance check on the following lending decision.

APPLICATION DETAILS:
- Application ID: {application_id}
- Company: {company_name}
- Sector: {sector}
- Loan Amount: {currency} {loan_amount}

ANALYST RECOMMENDATION: {analyst_recommendation}
Analyst reasoning: {analyst_reasoning}

REVIEWER ASSESSMENT: Agrees with analyst: {reviewer_agrees}
Reviewer reasoning: {reviewer_reasoning}

PORTFOLIO CONTEXT (for concentration check):
- Portfolio total: {currency} {portfolio_total}
- Existing exposures by borrower: {exposures_by_name}
- Existing exposures by sector: {exposures_by_sector}
- Borrower name for this application: {company_name}
- Sector for this application: {sector}
- Loan amount for this application: {loan_amount}

Execute ALL 5 regulatory checks in order:
1. CONSUMER DUTY CHECK (FCA Consumer Duty PS22/9)
2. FAIR LENDING CHECK (Equality Act 2010 / FCA PRIN 2.1)
3. RISK APPETITE CHECK (Internal Risk Appetite Framework)
4. CONCENTRATION CHECK (CRR Article 395 / Large Exposures) -- use concentration_checker tool
5. DOCUMENTATION CHECK (FCA SYSC 9.1 / Record Keeping)

Use rag_policy_lookup to retrieve relevant regulation text for each check.
Use concentration_checker for the concentration check with the portfolio data above.

Produce a ComplianceReport with application_id="{application_id}" and one ComplianceCheckResult per check."""

    def _extract_report(self, result: Any, structured_message_cls: type) -> ComplianceReport:
        """Extract the ComplianceReport from the TaskResult.

        Tries structured output first (output_content_type), then falls
        back to parsing the last assistant message as JSON.
        """
        # Try structured output (StructuredMessage with Pydantic content)
        for msg in reversed(result.messages):
            if isinstance(msg, structured_message_cls):
                content = msg.content
                if isinstance(content, ComplianceReport):
                    return content
                # If content is a dict, validate it
                if isinstance(content, dict):
                    return ComplianceReport.model_validate(content)

        # Fallback: parse last text message as JSON
        for msg in reversed(result.messages):
            if hasattr(msg, "content") and isinstance(msg.content, str):
                text = msg.content.strip()
                # Try to extract JSON from markdown code block
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                try:
                    return ComplianceReport.model_validate_json(text)
                except Exception:
                    continue

        raise ValueError("Could not extract ComplianceReport from agent output")

    def _extract_reasoning(self, result: Any) -> str:
        """Build a reasoning trace from the message history."""
        traces = []
        for msg in result.messages:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                source = getattr(msg, "source", "unknown")
                traces.append(f"[{source}] {msg.content[:500]}")
        return "\n---\n".join(traces) if traces else "No reasoning trace available"

    def _extract_sources(self, result: Any) -> list:
        """Extract source references from tool call results."""
        sources = []
        for msg in result.messages:
            msg_type = type(msg).__name__
            if msg_type == "ToolCallSummaryMessage":
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content:
                    sources.append(content[:200])
        return sources

    def _extract_tokens(self, result: Any) -> int:
        """Sum token usage from message metadata."""
        total = 0
        for msg in result.messages:
            usage = getattr(msg, "models_usage", None)
            if usage is not None:
                total += getattr(usage, "prompt_tokens", 0)
                total += getattr(usage, "completion_tokens", 0)
        return total
