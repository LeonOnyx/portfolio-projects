"""Unit tests for orchestrator node functions.

Tests intake_node, analysis_node, review_node, compliance_node,
decision_node, and escalation_node with mocked agent dependencies.

MOCKING NOTE: orchestrator_nodes.py uses lazy imports inside function
bodies. We patch the SOURCE modules so the lazy import picks up mocks.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.orchestrator_decision import decision_node, escalation_node
from src.orchestrator_nodes import (
    analysis_node,
    compliance_node,
    intake_node,
    review_node,
)


# ===========================================================================
# intake_node
# ===========================================================================


class TestIntakeNode:
    """intake_node validates the application, scans PII, builds audit trail."""

    async def test_intake_success(self, valid_application):
        """Successful intake returns current_stage=INTAKE and audit_trail."""
        # Patch LoanApplication to accept any dict
        mock_app = MagicMock()
        mock_app.application_id = "APP-TEST-001"

        mock_pii_result = MagicMock()
        mock_pii_result.pii_found = False
        mock_pii_result.pii_types_detected = []

        mock_detector = MagicMock()
        mock_detector.scan.return_value = mock_pii_result

        mock_trail = MagicMock()
        mock_entry = MagicMock()
        mock_entry.model_dump.return_value = {
            "stage": "intake",
            "action": "input_received",
        }
        mock_trail.entries = [mock_entry, mock_entry]

        with (
            patch("src.models.loan.LoanApplication", return_value=mock_app),
            patch("src.guardrails.pii.PIIDetector", return_value=mock_detector),
            patch("src.governance.audit.AuditTrail", return_value=mock_trail),
            patch("src.governance.audit.compute_content_hash", return_value="abc123"),
        ):
            result = await intake_node({"application": valid_application})

        assert result["current_stage"] == "intake"
        assert result["pii_detected"] is False
        assert isinstance(result["audit_trail"], list)
        assert len(result["audit_trail"]) == 2

    async def test_intake_validation_failure(self):
        """Validation error returns errors key."""
        from pydantic import ValidationError

        mock_exc = ValidationError.from_exception_data(
            title="LoanApplication",
            line_errors=[
                {
                    "type": "missing",
                    "loc": ("applicant",),
                    "msg": "Field required",
                    "input": {},
                }
            ],
        )

        with patch(
            "src.models.loan.LoanApplication", side_effect=mock_exc,
        ):
            result = await intake_node({"application": {}})

        assert "errors" in result
        assert result["current_stage"] == "intake"


# ===========================================================================
# analysis_node
# ===========================================================================


class TestAnalysisNode:
    """analysis_node wraps AnalystAgent with graceful degradation."""

    async def test_analysis_success(self, mock_analyst_response):
        """Successful analysis returns analysis_result with credit_score."""
        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(return_value=mock_analyst_response)
        mock_agent.to_audit_entry.return_value = {
            "agent": "financial_analyst",
            "confidence": 0.8,
        }

        mock_agent_cls = MagicMock(return_value=mock_agent)
        mock_tools = [MagicMock()]

        mock_config = MagicMock()
        mock_config.return_value.agents.return_value.analyst.model_dump.return_value = {
            "role": "analyst",
        }

        with (
            patch("src.agents.analyst.AnalystAgent", mock_agent_cls),
            patch("src.agents.tools_adapter.get_analyst_tools", return_value=mock_tools),
            patch("src.config.settings.ConfigLoader", mock_config),
        ):
            result = await analysis_node({
                "application": {"application_id": "APP-TEST-001"},
                "retrieved_documents": [],
            })

        assert result["current_stage"] == "analysis"
        assert result["analysis_result"]["credit_score"] == 72
        assert isinstance(result["audit_trail"], list)

    async def test_analysis_failure(self):
        """Agent failure returns errors key without raising."""
        mock_config = MagicMock()
        mock_config.return_value.agents.return_value.analyst.model_dump.return_value = {}

        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        mock_agent_cls = MagicMock(return_value=mock_agent)

        with (
            patch("src.agents.analyst.AnalystAgent", mock_agent_cls),
            patch("src.agents.tools_adapter.get_analyst_tools", return_value=[]),
            patch("src.config.settings.ConfigLoader", mock_config),
        ):
            result = await analysis_node({"application": {}})

        assert "errors" in result
        assert result["current_stage"] == "analysis"


# ===========================================================================
# review_node
# ===========================================================================


class TestReviewNode:
    """review_node wraps ReviewerAgent with graceful degradation."""

    async def test_review_success(self, mock_reviewer_response):
        """Successful review returns review_result."""
        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(return_value=mock_reviewer_response)
        mock_agent.to_audit_entry.return_value = {
            "agent": "independent_reviewer",
            "confidence": 0.85,
        }

        mock_agent_cls = MagicMock(return_value=mock_agent)

        mock_config = MagicMock()
        mock_config.return_value.agents.return_value.reviewer.model_dump.return_value = {
            "role": "reviewer",
        }

        with (
            patch("src.agents.reviewer.ReviewerAgent", mock_agent_cls),
            patch("src.agents.tools_adapter.get_reviewer_tools", return_value=[]),
            patch("src.config.settings.ConfigLoader", mock_config),
        ):
            result = await review_node({
                "application": {},
                "analysis_result": {"recommendation": "APPROVE"},
                "retrieved_documents": [],
            })

        assert result["current_stage"] == "review"
        assert result["review_result"]["quality_score"] == 0.85
        assert result["review_result"]["agrees_with_analyst"] is True


# ===========================================================================
# compliance_node
# ===========================================================================


class TestComplianceNode:
    """compliance_node wraps ComplianceAgent with graceful degradation."""

    async def test_compliance_success(self, mock_compliance_response):
        """Successful compliance returns compliance_result."""
        mock_agent = MagicMock()
        mock_agent.execute = AsyncMock(return_value=mock_compliance_response)
        mock_agent.to_audit_entry.return_value = {
            "agent": "compliance_officer",
            "confidence": 0.9,
        }

        mock_agent_cls = MagicMock(return_value=mock_agent)

        mock_config = MagicMock()
        mock_config.return_value.agents.return_value.compliance.model_dump.return_value = {
            "role": "compliance",
        }

        with (
            patch("src.agents.compliance.ComplianceAgent", mock_agent_cls),
            patch(
                "src.agents.tools_adapter.get_compliance_tools_autogen",
                return_value=[],
            ),
            patch("src.config.settings.ConfigLoader", mock_config),
        ):
            result = await compliance_node({
                "application": {},
                "analysis_result": {},
                "review_result": {},
                "retrieved_documents": [],
            })

        assert result["current_stage"] == "compliance"
        assert result["compliance_result"]["overall_passed"] is True


# ===========================================================================
# decision_node
# ===========================================================================


class TestDecisionNode:
    """decision_node applies matrix + escalation triggers."""

    async def test_decision_approved(self):
        """All-approve scenario -> APPROVED, no escalation."""
        # Build a state with approve+agree+pass + no triggers
        state = {
            "analysis_result": {
                "recommendation": "APPROVE",
                "credit_score": 72,
                "sector_outlook": "stable",
            },
            "review_result": {
                "agrees_with_analyst": True,
                "quality_score": 0.85,
                "confidence_level": "HIGH",
            },
            "compliance_result": {
                "overall_passed": True,
            },
            "grounding_scores": [
                {"score": 0.85},
            ],
            "application": {
                "application_id": "APP-TEST-001",
                "loan": {"amount_requested": 250000},
            },
        }

        # Mock ConfigLoader for check_escalation_triggers -- empty triggers
        mock_guardrails = MagicMock()
        mock_guardrails.escalation.triggers = []
        mock_guardrails.grounding.max_retries = 2
        mock_config = MagicMock()
        mock_config.return_value.guardrails.return_value = mock_guardrails

        with patch("src.config.settings.ConfigLoader", mock_config):
            result = await decision_node(state)

        assert result["final_decision"] == "APPROVED"
        assert result["requires_escalation"] is False
        assert result["confidence_score"] > 0
        assert isinstance(result["audit_trail"], list)

    async def test_decision_with_escalation(self):
        """High-value loan triggers escalation -> ESCALATED."""
        state = {
            "analysis_result": {
                "recommendation": "APPROVE",
                "credit_score": 72,
            },
            "review_result": {
                "agrees_with_analyst": True,
                "quality_score": 0.85,
                "confidence_level": "HIGH",
            },
            "compliance_result": {
                "overall_passed": True,
            },
            "grounding_scores": [],
            "application": {
                "application_id": "APP-TEST-001",
                "loan": {"amount_requested": 600000},
            },
        }

        # Mock ConfigLoader with high_value_loan trigger
        trigger = MagicMock()
        trigger.name = "high_value_loan"
        mock_guardrails = MagicMock()
        mock_guardrails.escalation.triggers = [trigger]
        mock_guardrails.grounding.max_retries = 2
        mock_config = MagicMock()
        mock_config.return_value.guardrails.return_value = mock_guardrails

        with patch("src.config.settings.ConfigLoader", mock_config):
            result = await decision_node(state)

        assert result["final_decision"] == "ESCALATED"
        assert result["requires_escalation"] is True


# ===========================================================================
# escalation_node
# ===========================================================================


class TestEscalationNode:
    """escalation_node routes to human review and records reasons."""

    async def test_escalation_with_reasons(self):
        """Escalation with upstream flag and errors."""
        state = {
            "requires_escalation": True,
            "errors": ["analysis_node: RuntimeError: LLM timeout"],
            "application": {"application_id": "APP-TEST-001"},
            "final_decision": "ERROR",
        }

        # Mock ConfigLoader for the belt-and-braces re-run
        mock_guardrails = MagicMock()
        mock_guardrails.escalation.triggers = []
        mock_guardrails.grounding.max_retries = 2
        mock_config = MagicMock()
        mock_config.return_value.guardrails.return_value = mock_guardrails

        with patch("src.config.settings.ConfigLoader", mock_config):
            result = await escalation_node(state)

        assert result["final_decision"] == "ESCALATED"
        assert result["requires_escalation"] is True
        assert result["current_stage"] == "escalate"
        assert isinstance(result["audit_trail"], list)
        assert len(result["audit_trail"]) == 1
        assert "human_review_required" in result["audit_trail"][0]["action"]
