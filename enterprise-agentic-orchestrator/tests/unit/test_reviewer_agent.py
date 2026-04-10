"""Unit tests for ReviewerAgent (CrewAI-based independent reviewer).

Tests agent construction, execute() success/error paths, reviewer
disagreement scenarios, output validation, and audit entry generation.

MOCKING STRATEGY: We mock at the Crew level (src.agents.reviewer.Crew)
to prevent any real LLM/CrewAI SDK instantiation. We also mock
_get_llm on the ReviewerAgent instance to prevent Azure OpenAI SDK
from loading.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import AgentResponse
from src.agents.reviewer import ReviewerAgent, validate_review_output
from src.models.reports import ConfidenceLevel


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture()
def reviewer_config() -> dict:
    """Minimal config dict for ReviewerAgent."""
    return {
        "model": "azure/gpt-4",
        "verbose": False,
        "require_grounding": False,  # Reviewer returns empty sources_used
    }


@pytest.fixture()
def reviewer(reviewer_config) -> ReviewerAgent:
    """ReviewerAgent instance with _get_llm patched."""
    with patch.object(ReviewerAgent, "_get_llm", return_value=MagicMock()):
        agent = ReviewerAgent(reviewer_config)
    return agent


@pytest.fixture()
def review_context() -> dict:
    """Valid context dict with application and analyst_report."""
    return {
        "application": {
            "application_id": "APP-TEST-001",
            "applicant": {"company_name": "Acme Ltd"},
            "sector": "technology",
            "loan_amount": 250000,
            "currency": "GBP",
            "purpose": "Working capital",
            "years_trading": 10,
        },
        "analyst_report": {
            "report_id": "RPT-ANALYST-001",
            "credit_score": 72,
            "recommendation": "APPROVE",
            "risk_metrics": {"probability_of_default": 0.08},
            "reasoning": "Strong financials.",
        },
    }


@pytest.fixture()
def mock_review_report_dict() -> dict:
    """Dict matching ReviewReport schema -- reviewer AGREES."""
    return {
        "report_id": "RPT-REVIEWER-001",
        "application_id": "APP-TEST-001",
        "analyst_report_id": "RPT-ANALYST-001",
        "agrees_with_analyst": True,
        "confidence_level": "HIGH",
        "quality_score": 0.85,
        "stress_test_results": [
            {"scenario": "base_case", "survives": True},
            {"scenario": "revenue_decline_20", "survives": True},
            {"scenario": "revenue_decline_40", "survives": True},
            {"scenario": "cost_increase_30", "survives": True},
            {"scenario": "combined_stress", "survives": False},
        ],
        "issues_found": [],
        "reasoning": (
            "Independent analysis confirms the analyst's assessment. "
            "Credit score within 2 points of independently calculated value. "
            "Stress tests show resilience in 4 of 5 scenarios."
        ),
    }


@pytest.fixture()
def mock_disagree_report_dict() -> dict:
    """Dict matching ReviewReport schema -- reviewer DISAGREES."""
    return {
        "report_id": "RPT-REVIEWER-002",
        "application_id": "APP-TEST-001",
        "analyst_report_id": "RPT-ANALYST-001",
        "agrees_with_analyst": False,
        "confidence_level": "LOW",
        "quality_score": 0.45,
        "stress_test_results": [
            {"scenario": "base_case", "survives": True},
            {"scenario": "revenue_decline_20", "survives": False},
            {"scenario": "revenue_decline_40", "survives": False},
            {"scenario": "cost_increase_30", "survives": False},
            {"scenario": "combined_stress", "survives": False},
        ],
        "issues_found": [
            "Analyst credit score 72 diverges from independent calculation of 58",
            "PD estimate of 0.08 too optimistic given stress test failures",
        ],
        "reasoning": (
            "Significant discrepancies found. Independent credit score of 58 "
            "is 14 points below the analyst's 72. Stress tests show the "
            "borrower fails 4 of 5 scenarios."
        ),
    }


def _build_mock_crew_result(report_dict: dict):
    """Build a mock CrewOutput from a report dict."""
    from src.models.reports import ReviewReport

    report = ReviewReport.model_validate(report_dict)
    result = MagicMock()
    result.pydantic = report
    token_usage = MagicMock()
    token_usage.total_tokens = 1200
    result.token_usage = token_usage
    return result


# ===========================================================================
# Construction tests
# ===========================================================================


class TestReviewerAgentInit:
    """ReviewerAgent.__init__ sets name, role, framework, config."""

    def test_name(self, reviewer):
        assert reviewer.name == "independent_reviewer"

    def test_role(self, reviewer):
        assert reviewer.role == "Independent Risk Reviewer"

    def test_framework(self, reviewer):
        assert reviewer.framework == "crewai"

    def test_config_stored(self, reviewer, reviewer_config):
        assert reviewer.config is reviewer_config

    def test_llm_starts_none(self, reviewer_config):
        """_llm is None before first _get_llm call."""
        agent = ReviewerAgent(reviewer_config)
        assert agent._llm is None


# ===========================================================================
# execute() success -- reviewer agrees
# ===========================================================================


class TestReviewerAgentExecuteAgree:
    """execute() returns correctly structured AgentResponse when reviewer agrees."""

    async def test_returns_agent_response(
        self, reviewer, review_context, mock_review_report_dict
    ):
        mock_result = _build_mock_crew_result(mock_review_report_dict)

        with (
            patch("src.agents.reviewer.Crew") as MockCrew,
            patch("src.agents.reviewer.Agent"),
            patch("src.agents.reviewer.Task"),
            patch.object(reviewer, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_result)
            MockCrew.return_value = mock_crew_instance

            response = await reviewer.execute(review_context, tools=[])

        assert isinstance(response, AgentResponse)

    async def test_agent_name(
        self, reviewer, review_context, mock_review_report_dict
    ):
        mock_result = _build_mock_crew_result(mock_review_report_dict)

        with (
            patch("src.agents.reviewer.Crew") as MockCrew,
            patch("src.agents.reviewer.Agent"),
            patch("src.agents.reviewer.Task"),
            patch.object(reviewer, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_result)
            MockCrew.return_value = mock_crew_instance

            response = await reviewer.execute(review_context, tools=[])

        assert response.agent_name == "independent_reviewer"

    async def test_agent_framework(
        self, reviewer, review_context, mock_review_report_dict
    ):
        mock_result = _build_mock_crew_result(mock_review_report_dict)

        with (
            patch("src.agents.reviewer.Crew") as MockCrew,
            patch("src.agents.reviewer.Agent"),
            patch("src.agents.reviewer.Task"),
            patch.object(reviewer, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_result)
            MockCrew.return_value = mock_crew_instance

            response = await reviewer.execute(review_context, tools=[])

        assert response.agent_framework == "crewai"

    async def test_output_has_review_keys(
        self, reviewer, review_context, mock_review_report_dict
    ):
        mock_result = _build_mock_crew_result(mock_review_report_dict)

        with (
            patch("src.agents.reviewer.Crew") as MockCrew,
            patch("src.agents.reviewer.Agent"),
            patch("src.agents.reviewer.Task"),
            patch.object(reviewer, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_result)
            MockCrew.return_value = mock_crew_instance

            response = await reviewer.execute(review_context, tools=[])

        assert "quality_score" in response.output
        assert "confidence_level" in response.output
        assert "agrees_with_analyst" in response.output
        assert "stress_test_results" in response.output

    async def test_confidence_for_high(
        self, reviewer, review_context, mock_review_report_dict
    ):
        """HIGH confidence maps to 0.9."""
        mock_result = _build_mock_crew_result(mock_review_report_dict)

        with (
            patch("src.agents.reviewer.Crew") as MockCrew,
            patch("src.agents.reviewer.Agent"),
            patch("src.agents.reviewer.Task"),
            patch.object(reviewer, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_result)
            MockCrew.return_value = mock_crew_instance

            response = await reviewer.execute(review_context, tools=[])

        assert response.confidence == 0.9

    async def test_tokens_used(
        self, reviewer, review_context, mock_review_report_dict
    ):
        mock_result = _build_mock_crew_result(mock_review_report_dict)

        with (
            patch("src.agents.reviewer.Crew") as MockCrew,
            patch("src.agents.reviewer.Agent"),
            patch("src.agents.reviewer.Task"),
            patch.object(reviewer, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_result)
            MockCrew.return_value = mock_crew_instance

            response = await reviewer.execute(review_context, tools=[])

        assert response.tokens_used == 1200


# ===========================================================================
# execute() -- reviewer disagrees
# ===========================================================================


class TestReviewerAgentExecuteDisagree:
    """execute() correctly reflects disagreement and low confidence."""

    async def test_disagree_output(
        self, reviewer, review_context, mock_disagree_report_dict
    ):
        mock_result = _build_mock_crew_result(mock_disagree_report_dict)

        with (
            patch("src.agents.reviewer.Crew") as MockCrew,
            patch("src.agents.reviewer.Agent"),
            patch("src.agents.reviewer.Task"),
            patch.object(reviewer, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_result)
            MockCrew.return_value = mock_crew_instance

            response = await reviewer.execute(review_context, tools=[])

        assert response.output["agrees_with_analyst"] is False

    async def test_low_confidence_mapping(
        self, reviewer, review_context, mock_disagree_report_dict
    ):
        """LOW confidence maps to 0.3."""
        mock_result = _build_mock_crew_result(mock_disagree_report_dict)

        with (
            patch("src.agents.reviewer.Crew") as MockCrew,
            patch("src.agents.reviewer.Agent"),
            patch("src.agents.reviewer.Task"),
            patch.object(reviewer, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_result)
            MockCrew.return_value = mock_crew_instance

            response = await reviewer.execute(review_context, tools=[])

        assert response.confidence == 0.3

    async def test_disagree_issues_populated(
        self, reviewer, review_context, mock_disagree_report_dict
    ):
        mock_result = _build_mock_crew_result(mock_disagree_report_dict)

        with (
            patch("src.agents.reviewer.Crew") as MockCrew,
            patch("src.agents.reviewer.Agent"),
            patch("src.agents.reviewer.Task"),
            patch.object(reviewer, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_result)
            MockCrew.return_value = mock_crew_instance

            response = await reviewer.execute(review_context, tools=[])

        assert len(response.output["issues_found"]) == 2


# ===========================================================================
# execute() -- analyst_report in context
# ===========================================================================


class TestReviewerContextUsage:
    """Reviewer uses analyst_report from context."""

    async def test_analyst_report_accessed(
        self, reviewer, review_context, mock_review_report_dict
    ):
        """Crew is called with application_id from context."""
        mock_result = _build_mock_crew_result(mock_review_report_dict)

        with (
            patch("src.agents.reviewer.Crew") as MockCrew,
            patch("src.agents.reviewer.Agent"),
            patch("src.agents.reviewer.Task") as MockTask,
            patch.object(reviewer, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_result)
            MockCrew.return_value = mock_crew_instance

            await reviewer.execute(review_context, tools=[])

        # Verify Task was called with a description containing analyst data
        task_call_kwargs = MockTask.call_args
        description = task_call_kwargs.kwargs.get(
            "description", task_call_kwargs.args[0] if task_call_kwargs.args else ""
        )
        # The description should reference the analyst report data
        assert "RPT-ANALYST-001" in description or "72" in description

    async def test_missing_analyst_report_handled(
        self, reviewer, mock_review_report_dict
    ):
        """Missing analyst_report defaults to empty dict, no crash."""
        context_no_analyst = {
            "application": {
                "application_id": "APP-TEST-001",
                "applicant": {"company_name": "Acme Ltd"},
                "sector": "technology",
                "loan_amount": 250000,
            }
        }
        mock_result = _build_mock_crew_result(mock_review_report_dict)

        with (
            patch("src.agents.reviewer.Crew") as MockCrew,
            patch("src.agents.reviewer.Agent"),
            patch("src.agents.reviewer.Task"),
            patch.object(reviewer, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(return_value=mock_result)
            MockCrew.return_value = mock_crew_instance

            response = await reviewer.execute(context_no_analyst, tools=[])

        assert isinstance(response, AgentResponse)


# ===========================================================================
# execute() error handling
# ===========================================================================


class TestReviewerAgentExecuteError:
    """execute() propagates exceptions when Crew/LLM fails."""

    async def test_crew_raises_propagates(self, reviewer, review_context):
        """When Crew.akickoff raises, exception propagates."""
        with (
            patch("src.agents.reviewer.Crew") as MockCrew,
            patch("src.agents.reviewer.Agent"),
            patch("src.agents.reviewer.Task"),
            patch.object(reviewer, "_get_llm", return_value=MagicMock()),
        ):
            mock_crew_instance = MagicMock()
            mock_crew_instance.akickoff = AsyncMock(
                side_effect=RuntimeError("LLM connection refused")
            )
            MockCrew.return_value = mock_crew_instance

            with pytest.raises(RuntimeError, match="LLM connection refused"):
                await reviewer.execute(review_context, tools=[])


# ===========================================================================
# validate_output (BaseAgent)
# ===========================================================================


class TestReviewerValidateOutput:
    """BaseAgent.validate_output for reviewer responses."""

    def test_valid_response(self, reviewer):
        response = AgentResponse(
            agent_name="independent_reviewer",
            agent_framework="crewai",
            output={"quality_score": 0.85, "agrees_with_analyst": True},
            reasoning_trace="Agreed with analyst",
            confidence=0.9,
            sources_used=[],  # Reviewer returns empty sources
            tokens_used=1200,
            latency_ms=2800.0,
        )
        # require_grounding is False for reviewer config
        assert reviewer.validate_output(response) is True

    def test_empty_output_fails(self, reviewer):
        response = AgentResponse(
            agent_name="independent_reviewer",
            agent_framework="crewai",
            output={},
            reasoning_trace="",
            confidence=0.9,
            sources_used=[],
            tokens_used=0,
            latency_ms=0.0,
        )
        assert reviewer.validate_output(response) is False


# ===========================================================================
# to_audit_entry
# ===========================================================================


class TestReviewerToAuditEntry:
    """to_audit_entry() returns dict with expected keys."""

    def test_audit_entry_keys(self, reviewer):
        response = AgentResponse(
            agent_name="independent_reviewer",
            agent_framework="crewai",
            output={"quality_score": 0.85},
            reasoning_trace="Review complete",
            confidence=0.9,
            sources_used=[],
            tokens_used=1200,
            latency_ms=2800.0,
        )
        entry = reviewer.to_audit_entry(response)

        assert entry["agent"] == "independent_reviewer"
        assert entry["framework"] == "crewai"
        assert entry["role"] == "Independent Risk Reviewer"
        assert entry["confidence"] == 0.9
        assert entry["sources_count"] == 0
        assert entry["tokens_used"] == 1200
        assert "timestamp" in entry
        assert "output_valid" in entry


# ===========================================================================
# Guardrail function
# ===========================================================================


class TestValidateReviewOutput:
    """validate_review_output guardrail checks output quality."""

    def test_valid_output_passes(self, mock_review_report_dict):
        from src.models.reports import ReviewReport

        report = ReviewReport.model_validate(mock_review_report_dict)
        result = MagicMock()
        result.pydantic = report

        passed, out = validate_review_output(result)
        assert passed is True
        assert out is result

    def test_none_pydantic_fails(self):
        result = MagicMock()
        result.pydantic = None

        passed, msg = validate_review_output(result)
        assert passed is False
        assert "could not be parsed" in msg

    def test_empty_stress_tests_fails(self, mock_review_report_dict):
        from src.models.reports import ReviewReport

        data = {**mock_review_report_dict, "stress_test_results": []}
        report = ReviewReport.model_validate(data)
        result = MagicMock()
        result.pydantic = report

        passed, msg = validate_review_output(result)
        assert passed is False
        assert "stress_test_results" in msg

    def test_empty_reasoning_fails(self, mock_review_report_dict):
        from src.models.reports import ReviewReport

        data = {**mock_review_report_dict, "reasoning": "   "}
        report = ReviewReport.model_validate(data)
        result = MagicMock()
        result.pydantic = report

        passed, msg = validate_review_output(result)
        assert passed is False
        assert "reasoning" in msg

    def test_empty_analyst_report_id_fails(self, mock_review_report_dict):
        from src.models.reports import ReviewReport

        data = {**mock_review_report_dict, "analyst_report_id": ""}
        report = ReviewReport.model_validate(data)
        result = MagicMock()
        result.pydantic = report

        passed, msg = validate_review_output(result)
        assert passed is False
        assert "analyst_report_id" in msg


# ===========================================================================
# _map_confidence_level
# ===========================================================================


class TestMapConfidenceLevel:
    """_map_confidence_level maps enum to float correctly."""

    def test_high(self, reviewer):
        assert reviewer._map_confidence_level(ConfidenceLevel.HIGH) == 0.9

    def test_medium(self, reviewer):
        assert reviewer._map_confidence_level(ConfidenceLevel.MEDIUM) == 0.6

    def test_low(self, reviewer):
        assert reviewer._map_confidence_level(ConfidenceLevel.LOW) == 0.3
