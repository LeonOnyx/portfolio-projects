"""Unit tests for ComplianceAgent (AutoGen-based regulatory compliance).

Tests agent construction, execute() success/error paths, single-fail-
means-overall-fail semantics, report extraction, and error handling.

MOCKING STRATEGY: We mock at the highest feasible level -- patching
the AutoGen AssistantAgent and StructuredMessage imports inside
ComplianceAgent.execute() (lazy imports inside the try block). This
avoids brittle coupling to AutoGen internals while still exercising
the agent's output parsing logic (_extract_report, _extract_reasoning,
_extract_sources, _extract_tokens).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import AgentResponse
from src.agents.compliance import ComplianceAgent


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def compliance_config() -> dict:
    """Minimal config dict for ComplianceAgent."""
    return {
        "llm_model": "gpt-4o",
        "max_turns": 3,
        "require_grounding": False,
    }


@pytest.fixture()
def agent(compliance_config) -> ComplianceAgent:
    """ComplianceAgent instance."""
    return ComplianceAgent(compliance_config)


@pytest.fixture()
def compliance_context() -> dict:
    """Valid context dict for ComplianceAgent.execute()."""
    return {
        "application": {
            "application_id": "APP-TEST-001",
            "applicant": {"company_name": "Acme Ltd"},
            "sector": "technology",
            "loan_amount": 250000,
            "currency": "GBP",
        },
        "analyst_report": {
            "recommendation": "APPROVE",
            "reasoning": "Strong financials with good cash flow.",
        },
        "reviewer_report": {
            "agrees_with_analyst": True,
            "reasoning": "Concur with analyst assessment.",
        },
        "portfolio": {
            "total": 10000000,
            "exposures_by_name": {"Acme Ltd": 0},
            "exposures_by_sector": {"technology": 2000000},
        },
    }


def _make_compliance_checks(all_pass: bool = True) -> list[dict]:
    """Build 5 compliance check results. If all_pass=False, check 3 fails."""
    checks = [
        {
            "check_name": "Consumer Duty Check",
            "passed": True,
            "regulation_cited": "FCA Consumer Duty (PS22/9)",
            "details": "Customer outcomes considered. Fair value demonstrated.",
        },
        {
            "check_name": "Fair Lending Check",
            "passed": True,
            "regulation_cited": "Equality Act 2010 / FCA PRIN 2.1",
            "details": "No protected characteristics influence decision.",
        },
        {
            "check_name": "Risk Appetite Check",
            "passed": all_pass,  # This one can fail
            "regulation_cited": "Internal Risk Appetite Framework",
            "details": (
                "Within risk appetite." if all_pass
                else "Exposure exceeds sector limit of 25% portfolio."
            ),
        },
        {
            "check_name": "Concentration Check",
            "passed": True,
            "regulation_cited": "CRR Article 395 / Large Exposures",
            "details": "Single-name and sector within limits.",
        },
        {
            "check_name": "Documentation Check",
            "passed": True,
            "regulation_cited": "FCA SYSC 9.1 / Record Keeping",
            "details": "All required documentation present.",
        },
    ]
    return checks


def _build_compliance_report_dict(all_pass: bool = True) -> dict:
    """Build a ComplianceReport-shaped dict."""
    checks = _make_compliance_checks(all_pass=all_pass)
    return {
        "report_id": "RPT-COMPLIANCE-001",
        "application_id": "APP-TEST-001",
        "checks": checks,
        "overall_passed": all_pass,
    }


def _build_mock_task_result(report_dict: dict, structured_message_cls=None):
    """Build a mock AutoGen TaskResult with messages.

    Produces a single StructuredMessage containing a validated
    ComplianceReport instance.
    """
    from src.models.reports import ComplianceReport

    report = ComplianceReport.model_validate(report_dict)

    # Create a mock StructuredMessage
    structured_msg = MagicMock()
    structured_msg.content = report
    structured_msg.source = "compliance_officer"
    structured_msg.models_usage = None

    # Create a text message for reasoning trace
    text_msg = MagicMock()
    text_msg.content = "Performing regulatory compliance checks..."
    text_msg.source = "compliance_officer"
    text_msg.models_usage = None

    # Create a tool call summary message for source extraction
    tool_msg = MagicMock()
    type(tool_msg).__name__ = "ToolCallSummaryMessage"
    tool_msg.content = "FCA Consumer Duty PS22/9 policy text retrieved"
    tool_msg.source = "tool"
    tool_msg.models_usage = None

    # Create a message with token usage
    usage_msg = MagicMock()
    usage_msg.content = "Compliance analysis complete."
    usage_msg.source = "compliance_officer"
    usage_model = MagicMock()
    usage_model.prompt_tokens = 800
    usage_model.completion_tokens = 400
    usage_msg.models_usage = usage_model

    result = MagicMock()
    result.messages = [text_msg, tool_msg, structured_msg, usage_msg]

    return result, report


# ===========================================================================
# Construction tests
# ===========================================================================


class TestComplianceAgentInit:
    """ComplianceAgent.__init__ sets name, role, framework, config."""

    def test_name(self, agent):
        assert agent.name == "compliance_officer"

    def test_role(self, agent):
        assert agent.role == "Regulatory Compliance Officer"

    def test_framework(self, agent):
        assert agent.framework == "autogen"

    def test_config_stored(self, agent, compliance_config):
        assert agent.config is compliance_config

    def test_model_client_starts_none(self, agent):
        """_model_client is None before first _get_model_client call."""
        assert agent._model_client is None

    def test_max_turns_from_config(self, agent):
        assert agent._max_turns == 3

    def test_max_turns_default(self):
        """Default max_turns is 3 when not in config."""
        agent = ComplianceAgent({})
        assert agent._max_turns == 3


# ===========================================================================
# execute() success -- all checks pass
# ===========================================================================


class TestComplianceAgentExecuteAllPass:
    """execute() returns correct AgentResponse when all compliance checks pass."""

    async def test_returns_agent_response(self, agent, compliance_context):
        report_dict = _build_compliance_report_dict(all_pass=True)
        mock_result, _ = _build_mock_task_result(report_dict)

        # Build a mock StructuredMessage class for isinstance checks
        MockStructuredMessage = type(mock_result.messages[2])

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MockStructuredMessage),
        ):
            response = await agent.execute(compliance_context, tools=[])

        assert isinstance(response, AgentResponse)

    async def test_agent_name(self, agent, compliance_context):
        report_dict = _build_compliance_report_dict(all_pass=True)
        mock_result, _ = _build_mock_task_result(report_dict)
        MockStructuredMessage = type(mock_result.messages[2])

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MockStructuredMessage),
        ):
            response = await agent.execute(compliance_context, tools=[])

        assert response.agent_name == "compliance_officer"

    async def test_agent_framework(self, agent, compliance_context):
        report_dict = _build_compliance_report_dict(all_pass=True)
        mock_result, _ = _build_mock_task_result(report_dict)
        MockStructuredMessage = type(mock_result.messages[2])

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MockStructuredMessage),
        ):
            response = await agent.execute(compliance_context, tools=[])

        assert response.agent_framework == "autogen"

    async def test_overall_passed_true(self, agent, compliance_context):
        report_dict = _build_compliance_report_dict(all_pass=True)
        mock_result, _ = _build_mock_task_result(report_dict)
        MockStructuredMessage = type(mock_result.messages[2])

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MockStructuredMessage),
        ):
            response = await agent.execute(compliance_context, tools=[])

        assert response.output["overall_passed"] is True

    async def test_checks_count(self, agent, compliance_context):
        report_dict = _build_compliance_report_dict(all_pass=True)
        mock_result, _ = _build_mock_task_result(report_dict)
        MockStructuredMessage = type(mock_result.messages[2])

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MockStructuredMessage),
        ):
            response = await agent.execute(compliance_context, tools=[])

        assert len(response.output["checks"]) == 5

    async def test_confidence_when_passed(self, agent, compliance_context):
        """Confidence is 1.0 when all checks pass."""
        report_dict = _build_compliance_report_dict(all_pass=True)
        mock_result, _ = _build_mock_task_result(report_dict)
        MockStructuredMessage = type(mock_result.messages[2])

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MockStructuredMessage),
        ):
            response = await agent.execute(compliance_context, tools=[])

        assert response.confidence == 1.0

    async def test_sources_extracted(self, agent, compliance_context):
        """Tool call summary messages are extracted as sources."""
        report_dict = _build_compliance_report_dict(all_pass=True)
        mock_result, _ = _build_mock_task_result(report_dict)
        MockStructuredMessage = type(mock_result.messages[2])

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MockStructuredMessage),
        ):
            response = await agent.execute(compliance_context, tools=[])

        assert isinstance(response.sources_used, list)
        assert len(response.sources_used) >= 1

    async def test_tokens_extracted(self, agent, compliance_context):
        """Token usage is summed from models_usage metadata."""
        report_dict = _build_compliance_report_dict(all_pass=True)
        mock_result, _ = _build_mock_task_result(report_dict)
        MockStructuredMessage = type(mock_result.messages[2])

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MockStructuredMessage),
        ):
            response = await agent.execute(compliance_context, tools=[])

        assert response.tokens_used == 1200  # 800 prompt + 400 completion

    async def test_each_check_has_required_fields(self, agent, compliance_context):
        """Each check_result has check_name, regulation_cited, passed, details."""
        report_dict = _build_compliance_report_dict(all_pass=True)
        mock_result, _ = _build_mock_task_result(report_dict)
        MockStructuredMessage = type(mock_result.messages[2])

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MockStructuredMessage),
        ):
            response = await agent.execute(compliance_context, tools=[])

        for check in response.output["checks"]:
            assert "check_name" in check
            assert "regulation_cited" in check
            assert "passed" in check
            assert "details" in check


# ===========================================================================
# execute() -- single-fail-means-overall-fail (AGNT-09)
# ===========================================================================


class TestComplianceAgentSingleFail:
    """Single check failure means overall_passed=False per AGNT-09."""

    async def test_overall_passed_false_on_single_failure(
        self, agent, compliance_context
    ):
        report_dict = _build_compliance_report_dict(all_pass=False)
        mock_result, _ = _build_mock_task_result(report_dict)
        MockStructuredMessage = type(mock_result.messages[2])

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MockStructuredMessage),
        ):
            response = await agent.execute(compliance_context, tools=[])

        assert response.output["overall_passed"] is False

    async def test_confidence_zero_on_failure(self, agent, compliance_context):
        """Confidence is 0.0 when compliance fails."""
        report_dict = _build_compliance_report_dict(all_pass=False)
        mock_result, _ = _build_mock_task_result(report_dict)
        MockStructuredMessage = type(mock_result.messages[2])

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MockStructuredMessage),
        ):
            response = await agent.execute(compliance_context, tools=[])

        assert response.confidence == 0.0

    async def test_failed_check_identified(self, agent, compliance_context):
        """The failed check (Risk Appetite) is identifiable in output."""
        report_dict = _build_compliance_report_dict(all_pass=False)
        mock_result, _ = _build_mock_task_result(report_dict)
        MockStructuredMessage = type(mock_result.messages[2])

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MockStructuredMessage),
        ):
            response = await agent.execute(compliance_context, tools=[])

        failed_checks = [
            c for c in response.output["checks"] if not c["passed"]
        ]
        assert len(failed_checks) == 1
        assert failed_checks[0]["check_name"] == "Risk Appetite Check"


# ===========================================================================
# execute() error handling
# ===========================================================================


class TestComplianceAgentExecuteError:
    """execute() handles errors gracefully with meaningful error messages."""

    async def test_autogen_error_returns_error_response(
        self, agent, compliance_context
    ):
        """When AutoGen agent.run() raises, execute returns error AgentResponse."""
        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(
            side_effect=RuntimeError("Model endpoint unavailable")
        )

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MagicMock),
        ):
            response = await agent.execute(compliance_context, tools=[])

        assert isinstance(response, AgentResponse)
        assert response.agent_name == "compliance_officer"
        # P0-5: error path now returns a proper error-indicating dict
        assert response.output["overall_passed"] is False
        assert response.output["application_id"] == "APP-TEST-001"
        assert "Compliance check failed" in response.output["error"]
        assert "Model endpoint unavailable" in response.reasoning_trace
        assert response.confidence == 0.0

    async def test_import_error_returns_error_response(
        self, agent, compliance_context
    ):
        """When AutoGen imports fail, execute returns error AgentResponse."""
        with patch.dict(
            "sys.modules",
            {"autogen_agentchat": None, "autogen_agentchat.agents": None},
        ):
            response = await agent.execute(compliance_context, tools=[])

        assert isinstance(response, AgentResponse)
        # P0-5: error path now returns a proper error-indicating dict
        assert response.output["overall_passed"] is False
        assert response.confidence == 0.0

    async def test_report_extraction_failure_returns_error(
        self, agent, compliance_context
    ):
        """When _extract_report can't find a report, error response returned."""
        # Result with no StructuredMessage and no valid JSON in text messages
        bad_msg = MagicMock()
        bad_msg.content = "I couldn't complete the task"
        bad_msg.source = "compliance_officer"
        bad_msg.models_usage = None

        bad_result = MagicMock()
        bad_result.messages = [bad_msg]

        mock_agent_instance = MagicMock()
        mock_agent_instance.run = AsyncMock(return_value=bad_result)

        MockStructuredMessage = type("StructuredMessage", (), {})

        with (
            patch(
                "src.agents.compliance.ComplianceAgent._get_model_client",
                return_value=MagicMock(),
            ),
            patch("autogen_agentchat.agents.AssistantAgent", return_value=mock_agent_instance),
            patch("autogen_agentchat.messages.StructuredMessage", MockStructuredMessage),
        ):
            response = await agent.execute(compliance_context, tools=[])

        # P0-5: error path now returns a proper error-indicating dict
        assert response.output["overall_passed"] is False
        assert "Error" in response.reasoning_trace


# ===========================================================================
# _extract_report
# ===========================================================================


class TestExtractReport:
    """_extract_report extracts ComplianceReport from TaskResult messages."""

    def test_extract_from_structured_message(self, agent):
        """Extracts report from a StructuredMessage with Pydantic content."""
        from src.models.reports import ComplianceReport

        report_dict = _build_compliance_report_dict(all_pass=True)
        report = ComplianceReport.model_validate(report_dict)

        StructuredMessage = type("StructuredMessage", (), {})
        msg = StructuredMessage()
        msg.content = report

        result = MagicMock()
        result.messages = [msg]

        extracted = agent._extract_report(result, StructuredMessage)
        assert isinstance(extracted, ComplianceReport)
        assert extracted.overall_passed is True
        assert len(extracted.checks) == 5

    def test_extract_from_json_text(self, agent):
        """Falls back to parsing JSON from text message."""
        report_dict = _build_compliance_report_dict(all_pass=True)

        StructuredMessage = type("StructuredMessage", (), {})

        text_msg = MagicMock()
        text_msg.content = json.dumps(report_dict)
        text_msg.source = "compliance_officer"

        result = MagicMock()
        result.messages = [text_msg]

        extracted = agent._extract_report(result, StructuredMessage)
        assert extracted.overall_passed is True

    def test_extract_from_json_code_block(self, agent):
        """Extracts JSON from markdown code block."""
        report_dict = _build_compliance_report_dict(all_pass=True)

        StructuredMessage = type("StructuredMessage", (), {})

        text_msg = MagicMock()
        text_msg.content = f"```json\n{json.dumps(report_dict)}\n```"
        text_msg.source = "compliance_officer"

        result = MagicMock()
        result.messages = [text_msg]

        extracted = agent._extract_report(result, StructuredMessage)
        assert extracted.overall_passed is True

    def test_extract_fails_when_no_report(self, agent):
        """Raises ValueError when no report found."""
        StructuredMessage = type("StructuredMessage", (), {})

        text_msg = MagicMock()
        text_msg.content = "Some random text with no JSON"
        text_msg.source = "compliance_officer"

        result = MagicMock()
        result.messages = [text_msg]

        with pytest.raises(ValueError, match="Could not extract ComplianceReport"):
            agent._extract_report(result, StructuredMessage)

    def test_extract_from_dict_content(self, agent):
        """Extracts report from StructuredMessage with dict content."""
        from src.models.reports import ComplianceReport

        report_dict = _build_compliance_report_dict(all_pass=True)

        StructuredMessage = type("StructuredMessage", (), {})
        msg = StructuredMessage()
        msg.content = report_dict

        result = MagicMock()
        result.messages = [msg]

        extracted = agent._extract_report(result, StructuredMessage)
        assert isinstance(extracted, ComplianceReport)


# ===========================================================================
# _extract_reasoning, _extract_sources, _extract_tokens
# ===========================================================================


class TestExtractHelpers:
    """Tests for reasoning, source, and token extraction methods."""

    def test_extract_reasoning(self, agent):
        msg1 = MagicMock()
        msg1.content = "Starting compliance analysis"
        msg1.source = "compliance_officer"

        msg2 = MagicMock()
        msg2.content = "All checks completed successfully"
        msg2.source = "compliance_officer"

        result = MagicMock()
        result.messages = [msg1, msg2]

        reasoning = agent._extract_reasoning(result)
        assert "Starting compliance" in reasoning
        assert "All checks completed" in reasoning

    def test_extract_reasoning_empty(self, agent):
        result = MagicMock()
        result.messages = []

        reasoning = agent._extract_reasoning(result)
        assert "No reasoning trace" in reasoning

    def test_extract_sources_from_tool_call_summary(self, agent):
        tool_msg = MagicMock()
        type(tool_msg).__name__ = "ToolCallSummaryMessage"
        tool_msg.content = "Policy retrieved: FCA Consumer Duty PS22/9"

        result = MagicMock()
        result.messages = [tool_msg]

        sources = agent._extract_sources(result)
        assert len(sources) == 1
        assert "FCA Consumer Duty" in sources[0]

    def test_extract_sources_ignores_non_tool_messages(self, agent):
        text_msg = MagicMock()
        type(text_msg).__name__ = "TextMessage"
        text_msg.content = "Some analysis text"

        result = MagicMock()
        result.messages = [text_msg]

        sources = agent._extract_sources(result)
        assert len(sources) == 0

    def test_extract_tokens_sums_usage(self, agent):
        msg1 = MagicMock()
        usage1 = MagicMock()
        usage1.prompt_tokens = 500
        usage1.completion_tokens = 200
        msg1.models_usage = usage1

        msg2 = MagicMock()
        usage2 = MagicMock()
        usage2.prompt_tokens = 300
        usage2.completion_tokens = 100
        msg2.models_usage = usage2

        result = MagicMock()
        result.messages = [msg1, msg2]

        tokens = agent._extract_tokens(result)
        assert tokens == 1100

    def test_extract_tokens_handles_none_usage(self, agent):
        msg = MagicMock()
        msg.models_usage = None

        result = MagicMock()
        result.messages = [msg]

        tokens = agent._extract_tokens(result)
        assert tokens == 0


# ===========================================================================
# validate_output (BaseAgent)
# ===========================================================================


class TestComplianceValidateOutput:
    """BaseAgent.validate_output for compliance responses."""

    def test_valid_response(self, agent):
        response = AgentResponse(
            agent_name="compliance_officer",
            agent_framework="autogen",
            output={"overall_passed": True, "checks": []},
            reasoning_trace="All checks passed",
            confidence=1.0,
            sources_used=[],
            tokens_used=1200,
            latency_ms=2500.0,
        )
        # require_grounding is False for compliance config
        assert agent.validate_output(response) is True

    def test_empty_output_fails(self, agent):
        response = AgentResponse(
            agent_name="compliance_officer",
            agent_framework="autogen",
            output={},
            reasoning_trace="",
            confidence=0.0,
            sources_used=[],
            tokens_used=0,
            latency_ms=0.0,
        )
        assert agent.validate_output(response) is False


# ===========================================================================
# to_audit_entry
# ===========================================================================


class TestComplianceToAuditEntry:
    """to_audit_entry() returns dict with expected keys."""

    def test_audit_entry_keys(self, agent):
        response = AgentResponse(
            agent_name="compliance_officer",
            agent_framework="autogen",
            output={"overall_passed": True},
            reasoning_trace="All checks passed",
            confidence=1.0,
            sources_used=["FCA policy doc"],
            tokens_used=1200,
            latency_ms=2500.0,
        )
        entry = agent.to_audit_entry(response)

        assert entry["agent"] == "compliance_officer"
        assert entry["framework"] == "autogen"
        assert entry["role"] == "Regulatory Compliance Officer"
        assert entry["confidence"] == 1.0
        assert entry["sources_count"] == 1
        assert entry["tokens_used"] == 1200
        assert "timestamp" in entry
        assert "output_valid" in entry


# ===========================================================================
# _build_task
# ===========================================================================


class TestBuildTask:
    """_build_task produces a structured prompt from context."""

    def test_task_includes_application_id(self, agent, compliance_context):
        task = agent._build_task(compliance_context)
        assert "APP-TEST-001" in task

    def test_task_includes_company_name(self, agent, compliance_context):
        task = agent._build_task(compliance_context)
        assert "Acme Ltd" in task

    def test_task_includes_all_five_checks(self, agent, compliance_context):
        task = agent._build_task(compliance_context)
        assert "CONSUMER DUTY CHECK" in task
        assert "FAIR LENDING CHECK" in task
        assert "RISK APPETITE CHECK" in task
        assert "CONCENTRATION CHECK" in task
        assert "DOCUMENTATION CHECK" in task

    def test_task_includes_analyst_recommendation(self, agent, compliance_context):
        task = agent._build_task(compliance_context)
        assert "APPROVE" in task

    def test_task_handles_missing_context(self, agent):
        """Missing context keys default to safe values."""
        task = agent._build_task({})
        assert "unknown" in task


# ===========================================================================
# ComplianceReport model validator (belt-and-braces)
# ===========================================================================


class TestComplianceReportValidator:
    """ComplianceReport.overall_passed_requires_all_checks model validator."""

    def test_all_pass_with_overall_true_valid(self):
        from src.models.reports import ComplianceReport

        report = ComplianceReport.model_validate(
            _build_compliance_report_dict(all_pass=True)
        )
        assert report.overall_passed is True

    def test_one_fail_with_overall_false_valid(self):
        from src.models.reports import ComplianceReport

        report = ComplianceReport.model_validate(
            _build_compliance_report_dict(all_pass=False)
        )
        assert report.overall_passed is False

    def test_one_fail_with_overall_true_invalid(self):
        """overall_passed=True with a failed check raises ValidationError."""
        from pydantic import ValidationError

        from src.models.reports import ComplianceReport

        data = _build_compliance_report_dict(all_pass=False)
        data["overall_passed"] = True  # Force invalid state

        with pytest.raises(ValidationError, match="overall_passed"):
            ComplianceReport.model_validate(data)
