"""Unit tests for routing functions and escalation triggers.

Routing functions live in src.orchestrator.
Escalation trigger evaluation lives in src.orchestrator_decision.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Routing functions -- from src.orchestrator (NOT src.orchestrator_decision)
from src.orchestrator import (
    route_after_decision,
    route_after_grounding_analysis,
    route_after_grounding_compliance,
    route_after_grounding_review,
    route_after_intake,
)

# Escalation triggers -- from src.orchestrator_decision
from src.orchestrator_decision import check_escalation_triggers


# ---------------------------------------------------------------------------
# Helper: build a mock ConfigLoader whose guardrails() returns only the
# given trigger names and a max_retries value.
# ---------------------------------------------------------------------------


_DEFAULT_THRESHOLDS: dict[str, float | None] = {
    "high_value_loan": 500_000,
    "low_reviewer_confidence": 0.5,
    "low_average_grounding": 0.75,
}


def _mock_config_with_triggers(trigger_names: list[str], max_retries: int = 2):
    """Return a patch context for ConfigLoader that yields guardrails
    with the specified escalation triggers and grounding.max_retries."""

    triggers = []
    for name in trigger_names:
        t = MagicMock()
        t.name = name
        t.threshold = _DEFAULT_THRESHOLDS.get(name)
        triggers.append(t)

    guardrails = MagicMock()
    guardrails.escalation.triggers = triggers
    guardrails.grounding.max_retries = max_retries

    mock_loader_cls = MagicMock()
    mock_loader_cls.return_value.guardrails.return_value = guardrails

    return patch("src.config.settings.ConfigLoader", mock_loader_cls)


# ===========================================================================
# Escalation trigger tests
# ===========================================================================


class TestHighValueLoanTrigger:
    """high_value_loan trigger fires when loan amount > 500000."""

    def test_fires_on_high_value(self):
        state = {
            "application": {"loan": {"amount_requested": 600000}},
        }
        with _mock_config_with_triggers(["high_value_loan"]):
            reasons = check_escalation_triggers(state)
        assert len(reasons) == 1
        assert "high_value_loan" in reasons[0]

    def test_does_not_fire_on_low_value(self):
        state = {
            "application": {"loan": {"amount_requested": 400000}},
        }
        with _mock_config_with_triggers(["high_value_loan"]):
            reasons = check_escalation_triggers(state)
        assert len(reasons) == 0


class TestDeterioratingSectorTrigger:
    """deteriorating_sector trigger fires on 'deteriorating' outlook."""

    def test_fires_on_deteriorating(self):
        state = {
            "analysis_result": {"sector_outlook": "deteriorating"},
        }
        with _mock_config_with_triggers(["deteriorating_sector"]):
            reasons = check_escalation_triggers(state)
        assert len(reasons) == 1
        assert "deteriorating_sector" in reasons[0]

    def test_does_not_fire_on_stable(self):
        state = {
            "analysis_result": {"sector_outlook": "stable"},
        }
        with _mock_config_with_triggers(["deteriorating_sector"]):
            reasons = check_escalation_triggers(state)
        assert len(reasons) == 0


class TestComplianceFailureTrigger:
    """compliance_failure trigger fires when overall_passed is False."""

    def test_fires_on_failure(self):
        state = {
            "compliance_result": {"overall_passed": False},
        }
        with _mock_config_with_triggers(["compliance_failure"]):
            reasons = check_escalation_triggers(state)
        assert len(reasons) == 1
        assert "compliance_failure" in reasons[0]

    def test_does_not_fire_on_pass(self):
        state = {
            "compliance_result": {"overall_passed": True},
        }
        with _mock_config_with_triggers(["compliance_failure"]):
            reasons = check_escalation_triggers(state)
        assert len(reasons) == 0


class TestLowReviewerConfidenceTrigger:
    """low_reviewer_confidence trigger fires when confidence_level maps < 0.5."""

    def test_fires_on_low(self):
        state = {
            "review_result": {"confidence_level": "LOW"},
        }
        with _mock_config_with_triggers(["low_reviewer_confidence"]):
            reasons = check_escalation_triggers(state)
        assert len(reasons) == 1
        assert "low_reviewer_confidence" in reasons[0]

    def test_does_not_fire_on_high(self):
        state = {
            "review_result": {"confidence_level": "HIGH"},
        }
        with _mock_config_with_triggers(["low_reviewer_confidence"]):
            reasons = check_escalation_triggers(state)
        assert len(reasons) == 0


class TestGroundingFailureTrigger:
    """grounding_failure trigger fires when retries exhausted for a checkpoint."""

    def test_fires_on_exhausted_retries(self):
        state = {
            "grounding_scores": [
                {"checkpoint": "post_analyst", "is_grounded": False, "score": 0.4},
                {"checkpoint": "post_analyst", "is_grounded": False, "score": 0.3},
            ],
        }
        with _mock_config_with_triggers(["grounding_failure"], max_retries=2):
            reasons = check_escalation_triggers(state)
        assert len(reasons) == 1
        assert "grounding_failure" in reasons[0]

    def test_does_not_fire_with_single_attempt(self):
        state = {
            "grounding_scores": [
                {"checkpoint": "post_analyst", "is_grounded": False, "score": 0.4},
            ],
        }
        with _mock_config_with_triggers(["grounding_failure"], max_retries=2):
            reasons = check_escalation_triggers(state)
        assert len(reasons) == 0


class TestLowAverageGroundingTrigger:
    """low_average_grounding trigger fires when average score < 0.75."""

    def test_fires_on_low_average(self):
        state = {
            "grounding_scores": [
                {"score": 0.5},
                {"score": 0.7},
            ],
        }
        with _mock_config_with_triggers(["low_average_grounding"]):
            reasons = check_escalation_triggers(state)
        assert len(reasons) == 1
        assert "low_average_grounding" in reasons[0]

    def test_does_not_fire_on_high_average(self):
        state = {
            "grounding_scores": [
                {"score": 0.80},
                {"score": 0.85},
            ],
        }
        with _mock_config_with_triggers(["low_average_grounding"]):
            reasons = check_escalation_triggers(state)
        assert len(reasons) == 0


class TestEscalationGracefulDegradation:
    """Escalation with empty/missing state fields returns empty list."""

    def test_empty_state_returns_no_triggers(self):
        with _mock_config_with_triggers([
            "high_value_loan",
            "deteriorating_sector",
            "compliance_failure",
            "low_reviewer_confidence",
            "grounding_failure",
            "low_average_grounding",
        ]):
            reasons = check_escalation_triggers({})
        assert reasons == []

    def test_none_state_fields_return_no_triggers(self):
        state = {
            "application": None,
            "analysis_result": None,
            "compliance_result": None,
            "review_result": None,
            "grounding_scores": None,
        }
        with _mock_config_with_triggers([
            "high_value_loan",
            "deteriorating_sector",
            "compliance_failure",
            "low_reviewer_confidence",
            "grounding_failure",
            "low_average_grounding",
        ]):
            reasons = check_escalation_triggers(state)
        assert reasons == []


# ===========================================================================
# Routing function tests
# ===========================================================================


class TestRouteAfterIntake:
    """route_after_intake: errors with INTAKE stage -> escalate."""

    def test_intake_error_routes_to_escalate(self):
        state = {
            "errors": [{"stage": "intake", "error": "Validation failed"}],
        }
        assert route_after_intake(state) == "escalate"

    def test_no_errors_routes_to_analysis(self):
        state = {"errors": []}
        assert route_after_intake(state) == "analysis"

    def test_missing_errors_routes_to_analysis(self):
        state = {}
        assert route_after_intake(state) == "analysis"


class TestRouteAfterGroundingAnalysis:
    """route_after_grounding_analysis: is_grounded, retries, circuit breaker."""

    def test_grounded_routes_to_review(self):
        state = {
            "grounding_scores": [
                {"checkpoint": "post_analyst", "is_grounded": True, "score": 0.9},
            ],
        }
        assert route_after_grounding_analysis(state) == "review"

    def test_not_grounded_single_attempt_retries(self):
        state = {
            "grounding_scores": [
                {"checkpoint": "post_analyst", "is_grounded": False, "score": 0.4},
            ],
        }
        assert route_after_grounding_analysis(state) == "analysis"

    def test_not_grounded_retries_exhausted_escalates(self):
        state = {
            "grounding_scores": [
                {"checkpoint": "post_analyst", "is_grounded": False, "score": 0.3},
                {"checkpoint": "post_analyst", "is_grounded": False, "score": 0.4},
            ],
        }
        assert route_after_grounding_analysis(state) == "escalate"

    def test_circuit_broken_escalates(self):
        state = {
            "grounding_scores": [
                {"checkpoint": "post_analyst", "is_grounded": False, "circuit_broken": True},
            ],
        }
        assert route_after_grounding_analysis(state) == "escalate"

    def test_no_matching_entries_escalates(self):
        state = {
            "grounding_scores": [
                {"checkpoint": "post_reviewer", "is_grounded": True},
            ],
        }
        assert route_after_grounding_analysis(state) == "escalate"


class TestRouteAfterGroundingReview:
    """route_after_grounding_review: grounded=compliance, exhausted=escalate."""

    def test_grounded_routes_to_compliance(self):
        state = {
            "grounding_scores": [
                {"checkpoint": "post_reviewer", "is_grounded": True, "score": 0.85},
            ],
        }
        assert route_after_grounding_review(state) == "compliance"

    def test_retries_exhausted_escalates(self):
        state = {
            "grounding_scores": [
                {"checkpoint": "post_reviewer", "is_grounded": False, "score": 0.3},
                {"checkpoint": "post_reviewer", "is_grounded": False, "score": 0.4},
            ],
        }
        assert route_after_grounding_review(state) == "escalate"


class TestRouteAfterGroundingCompliance:
    """route_after_grounding_compliance: grounded=decision, exhausted=escalate."""

    def test_grounded_routes_to_decision(self):
        state = {
            "grounding_scores": [
                {"checkpoint": "post_compliance", "is_grounded": True, "score": 0.9},
            ],
        }
        assert route_after_grounding_compliance(state) == "decision"

    def test_retries_exhausted_escalates(self):
        state = {
            "grounding_scores": [
                {"checkpoint": "post_compliance", "is_grounded": False, "score": 0.3},
                {"checkpoint": "post_compliance", "is_grounded": False, "score": 0.4},
            ],
        }
        assert route_after_grounding_compliance(state) == "escalate"


class TestRouteAfterDecision:
    """route_after_decision: requires_escalation -> escalate; otherwise end."""

    def test_escalation_required_routes_to_escalate(self):
        state = {"requires_escalation": True}
        assert route_after_decision(state) == "escalate"

    def test_no_escalation_routes_to_end(self):
        state = {"requires_escalation": False}
        assert route_after_decision(state) == "__end__"

    def test_missing_flag_routes_to_end(self):
        state = {}
        assert route_after_decision(state) == "__end__"
