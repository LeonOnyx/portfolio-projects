"""Unit tests for AnalystAgent (CrewAI-based credit analyst).

Tests agent construction, execute() success/error paths, output
validation (BaseAgent.validate_output), audit entry generation, and
the guardrail function.

MOCKING STRATEGY: We mock at the Crew level (src.agents.analyst.Crew)
to prevent any real LLM/CrewAI SDK instantiation. We also mock
_get_llm on the AnalystAgent instance to prevent Azure OpenAI SDK
from loading.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.analyst import AnalystAgent, validate_analysis_output
from src.agents.base import AgentResponse


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def analyst_config() -> dict:
    """Minimal config dict for AnalystAgent."""
    return {
        "model": "azure/gpt-4",
        "verbose": False,
        "require_grounding": True,
    }


@pytest.fixture()
def analyst(analyst_config) -> AnalystAgent:
    """AnalystAgent instance with _get_llm patched to prevent SDK init."""
    with patch.object(AnalystAgent, "_get_llm", return_value=MagicMock()):
        agent = AnalystAgent(analyst_config)
    return agent


@pytest.fixture()
def analysis_context() -> dict:
    """Minimal valid context dict for AnalystAgent.execute()."""
    return {
        "application": {
            "application_id": "APP-TEST-001",
            "applicant": {"company_name": "Acme Ltd"},
            "sector": "technology",
            "loan_amount": 250000,
            "currency": "GBP",
            "purpose": "Working capital",
            "years_trading": 10,
            "ccj_count": 0,
        }
    }


@pytest.fixture()
def mock_analysis_report_dict() -> dict:
    """Dict matching AnalysisReport schema for mocked crew output."""
    return {
        "report_id": "RPT-ANALYST-001",
        "application_id": "APP-TEST-001",
        "credit_score": 72,
        "risk_metrics": {
            "probability_of_default": 0.08,
            "loss_given_default": 0.45,
            "exposure_at_default": "250000.00",
            "expected_loss": "9000.00",
        },
        "sector_outlook": "stable",
        "recommendation": "APPROVE",
        "reasoning": (
            "Strong financial position with consistent revenue growth. "
            "Debt-to-asset ratio is healthy at 0.40. Cash balance provides "
            "adequate liquidity buffer."
        ),
        "source_citations": [
            {"source": "credit_scorer", "reference": "Score: 72/100"},
            {"source": "risk_calculator", "reference": "PD: 0.08"},
            {"source": "rag_financial_lookup", "reference": "Revenue: 1M"},
        ],
    }


# ===========================================================================
# Construction tests
# ===========================================================================


class TestAnalystAgentInit:
    """AnalystAgent.__init__ sets name, role, framework, config."""

    def test_name(self, analyst):
        assert analyst.name == "financial_analyst"

    def test_role(self, analyst):
        assert analyst.role == "Financial Analyst"

    def test_framework(self, analyst):
        assert analyst.framework == "crewai"

    def test_config_stored(self, analyst, analyst_config):
        assert analyst.config is analyst_config

    def test_llm_starts_none(self, analyst_config):
        """_llm is None before first _get_llm call."""
        # Must avoid the fixture that patches _get_llm
        agent = AnalystAgent(analyst_config)
        assert agent._llm is None


# ===========================================================================
# execute() success
# ===========================================================================


class TestAnalystAgentExecuteSuccess:
    """execute() returns an AgentResponse with correct structure when Crew is mocked."""

    @pytest.fixture()
    def mock_crew_result(self, mock_analysis_report_dict):
        """Build a mock CrewOutput with .pydantic and .token_usage."""
        from src.models.reports import AnalysisReport

        report = AnalysisReport.model_validate(mock_analysis_report_dict)

        result = MagicMock()
        result.pydantic = report
        token_usage = MagicMock()
        token_usage.total_tokens = 1500
        result.token_usage = token_usage
        return result

    async def test_returns_agent_response(
        self, analyst, analysis_context, mock_crew_result
    ):
        """execute() returns an AgentResponse dataclass."""
        with (
            patch("src.agents.analyst.Crew") as MockCrew,
            patch("src.agents.analyst.Agent"),
            patch("src.agents.analyst.Task"),
            patch.object(analyst, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_crew_result)
            MockCrew.return_value = mock_crew_instance

            response = await analyst.execute(analysis_context, tools=[])

        assert isinstance(response, AgentResponse)

    async def test_agent_name(self, analyst, analysis_context, mock_crew_result):
        with (
            patch("src.agents.analyst.Crew") as MockCrew,
            patch("src.agents.analyst.Agent"),
            patch("src.agents.analyst.Task"),
            patch.object(analyst, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_crew_result)
            MockCrew.return_value = mock_crew_instance

            response = await analyst.execute(analysis_context, tools=[])

        assert response.agent_name == "financial_analyst"

    async def test_agent_framework(self, analyst, analysis_context, mock_crew_result):
        with (
            patch("src.agents.analyst.Crew") as MockCrew,
            patch("src.agents.analyst.Agent"),
            patch("src.agents.analyst.Task"),
            patch.object(analyst, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_crew_result)
            MockCrew.return_value = mock_crew_instance

            response = await analyst.execute(analysis_context, tools=[])

        assert response.agent_framework == "crewai"

    async def test_output_has_expected_keys(
        self, analyst, analysis_context, mock_crew_result
    ):
        with (
            patch("src.agents.analyst.Crew") as MockCrew,
            patch("src.agents.analyst.Agent"),
            patch("src.agents.analyst.Task"),
            patch.object(analyst, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_crew_result)
            MockCrew.return_value = mock_crew_instance

            response = await analyst.execute(analysis_context, tools=[])

        assert "credit_score" in response.output
        assert "recommendation" in response.output
        assert "risk_metrics" in response.output
        assert "source_citations" in response.output

    async def test_confidence_range(
        self, analyst, analysis_context, mock_crew_result
    ):
        with (
            patch("src.agents.analyst.Crew") as MockCrew,
            patch("src.agents.analyst.Agent"),
            patch("src.agents.analyst.Task"),
            patch.object(analyst, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_crew_result)
            MockCrew.return_value = mock_crew_instance

            response = await analyst.execute(analysis_context, tools=[])

        assert 0.0 <= response.confidence <= 1.0

    async def test_sources_used_list(
        self, analyst, analysis_context, mock_crew_result
    ):
        with (
            patch("src.agents.analyst.Crew") as MockCrew,
            patch("src.agents.analyst.Agent"),
            patch("src.agents.analyst.Task"),
            patch.object(analyst, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_crew_result)
            MockCrew.return_value = mock_crew_instance

            response = await analyst.execute(analysis_context, tools=[])

        assert isinstance(response.sources_used, list)
        assert len(response.sources_used) == 3

    async def test_tokens_used(
        self, analyst, analysis_context, mock_crew_result
    ):
        with (
            patch("src.agents.analyst.Crew") as MockCrew,
            patch("src.agents.analyst.Agent"),
            patch("src.agents.analyst.Task"),
            patch.object(analyst, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_crew_result)
            MockCrew.return_value = mock_crew_instance

            response = await analyst.execute(analysis_context, tools=[])

        assert response.tokens_used == 1500

    async def test_latency_positive(
        self, analyst, analysis_context, mock_crew_result
    ):
        with (
            patch("src.agents.analyst.Crew") as MockCrew,
            patch("src.agents.analyst.Agent"),
            patch("src.agents.analyst.Task"),
            patch.object(analyst, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_crew_result)
            MockCrew.return_value = mock_crew_instance

            response = await analyst.execute(analysis_context, tools=[])

        assert response.latency_ms >= 0.0


# ===========================================================================
# execute() error handling
# ===========================================================================


class TestAnalystAgentExecuteError:
    """execute() propagates exceptions when Crew/LLM fails."""

    async def test_crew_raises_propagates(self, analyst, analysis_context):
        """When Crew.akickoff raises, exception propagates to caller."""
        with (
            patch("src.agents.analyst.Crew") as MockCrew,
            patch("src.agents.analyst.Agent"),
            patch("src.agents.analyst.Task"),
            patch.object(analyst, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(
                side_effect=RuntimeError("LLM timeout: model not responding")
            )
            MockCrew.return_value = mock_crew_instance

            with pytest.raises(RuntimeError, match="LLM timeout"):
                await analyst.execute(analysis_context, tools=[])

    async def test_token_usage_none_defaults_zero(
        self, analyst, analysis_context
    ):
        """When token_usage is None, tokens_used defaults to 0."""
        from src.models.reports import AnalysisReport

        report = AnalysisReport(
            application_id="APP-TEST-001",
            credit_score=72,
            risk_metrics={
                "probability_of_default": 0.08,
                "loss_given_default": 0.45,
                "exposure_at_default": "250000.00",
                "expected_loss": "9000.00",
            },
            sector_outlook="stable",
            recommendation="APPROVE",
            reasoning="Adequate financials observed across all tool results.",
            source_citations=[{"source": "credit_scorer", "reference": "72"}],
        )

        result = MagicMock()
        result.pydantic = report
        result.token_usage = None

        with (
            patch("src.agents.analyst.Crew") as MockCrew,
            patch("src.agents.analyst.Agent"),
            patch("src.agents.analyst.Task"),
            patch.object(analyst, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=result)
            MockCrew.return_value = mock_crew_instance

            response = await analyst.execute(analysis_context, tools=[])

        assert response.tokens_used == 0


# ===========================================================================
# validate_output (BaseAgent)
# ===========================================================================


class TestAnalystValidateOutput:
    """BaseAgent.validate_output checks output, confidence, sources."""

    def test_valid_response(self, analyst):
        response = AgentResponse(
            agent_name="financial_analyst",
            agent_framework="crewai",
            output={"credit_score": 72, "recommendation": "APPROVE"},
            reasoning_trace="Good analysis",
            confidence=0.8,
            sources_used=[{"tool": "credit_scorer"}],
            tokens_used=1500,
            latency_ms=3200.0,
        )
        assert analyst.validate_output(response) is True

    def test_empty_output_fails(self, analyst):
        response = AgentResponse(
            agent_name="financial_analyst",
            agent_framework="crewai",
            output={},
            reasoning_trace="",
            confidence=0.8,
            sources_used=[{"tool": "credit_scorer"}],
            tokens_used=0,
            latency_ms=0.0,
        )
        assert analyst.validate_output(response) is False

    def test_confidence_above_one_fails(self, analyst):
        response = AgentResponse(
            agent_name="financial_analyst",
            agent_framework="crewai",
            output={"credit_score": 72},
            reasoning_trace="Analysis",
            confidence=1.5,
            sources_used=[{"tool": "credit_scorer"}],
            tokens_used=1500,
            latency_ms=3200.0,
        )
        assert analyst.validate_output(response) is False

    def test_confidence_negative_fails(self, analyst):
        response = AgentResponse(
            agent_name="financial_analyst",
            agent_framework="crewai",
            output={"credit_score": 72},
            reasoning_trace="Analysis",
            confidence=-0.1,
            sources_used=[{"tool": "credit_scorer"}],
            tokens_used=1500,
            latency_ms=3200.0,
        )
        assert analyst.validate_output(response) is False

    def test_no_sources_with_grounding_required_fails(self, analyst):
        """When require_grounding=True (default), empty sources fails."""
        response = AgentResponse(
            agent_name="financial_analyst",
            agent_framework="crewai",
            output={"credit_score": 72},
            reasoning_trace="Analysis",
            confidence=0.8,
            sources_used=[],
            tokens_used=1500,
            latency_ms=3200.0,
        )
        assert analyst.validate_output(response) is False

    def test_no_sources_without_grounding_required_passes(self):
        """When require_grounding=False, empty sources passes."""
        agent = AnalystAgent({"require_grounding": False})
        response = AgentResponse(
            agent_name="financial_analyst",
            agent_framework="crewai",
            output={"credit_score": 72},
            reasoning_trace="Analysis",
            confidence=0.8,
            sources_used=[],
            tokens_used=1500,
            latency_ms=3200.0,
        )
        assert agent.validate_output(response) is True


# ===========================================================================
# to_audit_entry
# ===========================================================================


class TestAnalystToAuditEntry:
    """to_audit_entry() returns dict with expected keys."""

    def test_audit_entry_keys(self, analyst):
        response = AgentResponse(
            agent_name="financial_analyst",
            agent_framework="crewai",
            output={"credit_score": 72},
            reasoning_trace="Analysis",
            confidence=0.8,
            sources_used=[{"tool": "credit_scorer"}, {"tool": "risk_calculator"}],
            tokens_used=1500,
            latency_ms=3200.0,
        )
        entry = analyst.to_audit_entry(response)

        assert entry["agent"] == "financial_analyst"
        assert entry["framework"] == "crewai"
        assert entry["role"] == "Financial Analyst"
        assert entry["confidence"] == 0.8
        assert entry["sources_count"] == 2
        assert entry["tokens_used"] == 1500
        assert entry["latency_ms"] == 3200.0
        assert "timestamp" in entry
        assert "output_valid" in entry

    def test_audit_entry_output_valid_flag(self, analyst):
        """output_valid reflects validate_output result."""
        valid_response = AgentResponse(
            agent_name="financial_analyst",
            agent_framework="crewai",
            output={"credit_score": 72},
            reasoning_trace="Analysis",
            confidence=0.8,
            sources_used=[{"tool": "credit_scorer"}],
            tokens_used=1500,
            latency_ms=3200.0,
        )
        entry = analyst.to_audit_entry(valid_response)
        assert entry["output_valid"] is True


# ===========================================================================
# Guardrail function
# ===========================================================================


class TestValidateAnalysisOutput:
    """validate_analysis_output guardrail checks output quality."""

    def test_valid_output_passes(self, mock_analysis_report_dict):
        from src.models.reports import AnalysisReport

        report = AnalysisReport.model_validate(mock_analysis_report_dict)
        result = MagicMock()
        result.pydantic = report

        passed, out = validate_analysis_output(result)
        assert passed is True
        assert out is result

    def test_none_pydantic_fails(self):
        result = MagicMock()
        result.pydantic = None

        passed, msg = validate_analysis_output(result)
        assert passed is False
        assert "could not be parsed" in msg

    def test_empty_citations_fails(self, mock_analysis_report_dict):
        from src.models.reports import AnalysisReport

        data = {**mock_analysis_report_dict, "source_citations": []}
        report = AnalysisReport.model_validate(data)
        result = MagicMock()
        result.pydantic = report

        passed, msg = validate_analysis_output(result)
        assert passed is False
        assert "source_citations" in msg

    def test_empty_reasoning_fails(self, mock_analysis_report_dict):
        from src.models.reports import AnalysisReport

        data = {**mock_analysis_report_dict, "reasoning": "   "}
        report = AnalysisReport.model_validate(data)
        result = MagicMock()
        result.pydantic = report

        passed, msg = validate_analysis_output(result)
        assert passed is False
        assert "reasoning" in msg


# ===========================================================================
# _derive_confidence
# ===========================================================================


class TestDeriveConfidence:
    """_derive_confidence returns correct values based on report completeness."""

    def test_full_report_high_confidence(self, analyst, mock_analysis_report_dict):
        from src.models.reports import AnalysisReport

        report = AnalysisReport.model_validate(mock_analysis_report_dict)
        confidence = analyst._derive_confidence(report)
        assert confidence == 0.8

    def test_few_citations_reduces_confidence(self, analyst, mock_analysis_report_dict):
        from src.models.reports import AnalysisReport

        data = {
            **mock_analysis_report_dict,
            "source_citations": [{"source": "one", "reference": "r1"}],
        }
        report = AnalysisReport.model_validate(data)
        confidence = analyst._derive_confidence(report)
        assert confidence == pytest.approx(0.7, abs=0.01)

    def test_short_reasoning_reduces_confidence(self, analyst, mock_analysis_report_dict):
        from src.models.reports import AnalysisReport

        data = {**mock_analysis_report_dict, "reasoning": "Short."}
        report = AnalysisReport.model_validate(data)
        confidence = analyst._derive_confidence(report)
        assert confidence == pytest.approx(0.7, abs=0.01)

    def test_unknown_sector_reduces_confidence(self, analyst, mock_analysis_report_dict):
        from src.models.reports import AnalysisReport

        data = {**mock_analysis_report_dict, "sector_outlook": "unknown"}
        report = AnalysisReport.model_validate(data)
        confidence = analyst._derive_confidence(report)
        assert confidence == pytest.approx(0.7, abs=0.01)

    def test_confidence_never_below_zero(self, analyst, mock_analysis_report_dict):
        from src.models.reports import AnalysisReport

        data = {
            **mock_analysis_report_dict,
            "source_citations": [{"source": "one", "reference": "r1"}],
            "reasoning": "Short.",
            "sector_outlook": "unknown",
        }
        report = AnalysisReport.model_validate(data)
        confidence = analyst._derive_confidence(report)
        assert confidence >= 0.0
