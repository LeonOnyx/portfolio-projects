"""End-to-end integration tests for the CreditRiskOrchestrator pipeline.

Four scenarios exercise the FULL LangGraph state machine with real routing,
decision matrix, and escalation triggers. Only agent execute() calls and
grounding/embedding externals are mocked.

Scenarios
---------
1. Clean approval   -- APPROVE + agree + compliance pass = APPROVED
2. Rejection        -- REJECT + agree                    = REJECTED
3. Referral         -- APPROVE + disagree + pass         = REFERRED_TO_UNDERWRITER
4. Compliance override -- compliance fails               = ESCALATED
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import AgentResponse

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_APPLICATION = {
    "application_id": "E2E-TEST-001",
    "applicant": {
        "company_name": "Integration Test Corp",
        "company_number": "99887766",
        "sector": "technology",
        "years_trading": 8,
        "employee_count": 40,
        "contact_name": "Jane Doe",
        "contact_role": "CFO",
    },
    "loan": {
        "amount_requested": float(Decimal("250000.00")),
        "term_months": 36,
        "purpose": "Working capital expansion",
        "security_type": "unsecured",
        "currency": "GBP",
    },
    "financials": [
        {
            "year": 2024,
            "revenue": float(Decimal("1200000.00")),
            "gross_profit": float(Decimal("480000.00")),
            "net_profit": float(Decimal("240000.00")),
            "total_assets": float(Decimal("2000000.00")),
            "total_liabilities": float(Decimal("800000.00")),
            "cash_balance": float(Decimal("200000.00")),
        },
    ],
    "credit_score": 72,
    "ccj_count": 0,
}


def _make_analyst_response(
    recommendation: str = "APPROVE",
    credit_score: int = 75,
    sector_outlook: str = "stable",
) -> AgentResponse:
    return AgentResponse(
        agent_name="financial_analyst",
        agent_framework="crewai",
        output={
            "report_id": "RPT-ANALYST-E2E",
            "application_id": "E2E-TEST-001",
            "credit_score": credit_score,
            "risk_metrics": {
                "probability_of_default": 0.06,
                "loss_given_default": 0.40,
                "exposure_at_default": 250000.00,
                "expected_loss": 6000.00,
            },
            "sector_outlook": sector_outlook,
            "recommendation": recommendation,
            "reasoning": "E2E test analyst response.",
            "source_citations": [],
        },
        reasoning_trace="E2E analyst trace.",
        confidence=0.82,
        sources_used=[{"tool": "credit_scorer"}],
        tokens_used=1200,
        latency_ms=2000.0,
    )


def _make_reviewer_response(
    agrees: bool = True,
    quality_score: float = 0.85,
    confidence_level: str = "HIGH",
) -> AgentResponse:
    return AgentResponse(
        agent_name="independent_reviewer",
        agent_framework="crewai",
        output={
            "quality_score": quality_score,
            "confidence_level": confidence_level,
            "agrees_with_analyst": agrees,
            "stress_test_results": {},
            "findings": [],
        },
        reasoning_trace="E2E reviewer trace.",
        confidence=0.85,
        sources_used=[{"tool": "stress_tester"}],
        tokens_used=1000,
        latency_ms=1800.0,
    )


def _make_compliance_response(overall_passed: bool = True) -> AgentResponse:
    return AgentResponse(
        agent_name="compliance_officer",
        agent_framework="autogen",
        output={
            "overall_passed": overall_passed,
            "check_results": [
                {"check": "aml", "passed": True},
                {"check": "kyc", "passed": True},
                {"check": "sanctions", "passed": True},
                {"check": "pep", "passed": True},
                {"check": "credit_limit", "passed": overall_passed},
            ],
            "plain_language_summary": (
                "All checks passed." if overall_passed else "Credit limit check failed."
            ),
        },
        reasoning_trace="E2E compliance trace.",
        confidence=0.90,
        sources_used=[{"tool": "regulation_lookup"}],
        tokens_used=900,
        latency_ms=2200.0,
    )


def _mock_grounding_node(checkpoint_name: str):
    """Return an async callable that mimics a grounding checkpoint node."""

    async def _grounding(state: dict) -> dict:
        return {
            "grounding_results": {
                checkpoint_name: {
                    "grounding_score": 0.95,
                    "is_grounded": True,
                }
            }
        }

    return _grounding


# ---------------------------------------------------------------------------
# Mock ConfigLoader
# ---------------------------------------------------------------------------


def _build_mock_config_loader():
    """Build a ConfigLoader mock that returns enough config for the full pipeline."""
    mock_cls = MagicMock()
    instance = mock_cls.return_value

    # -- Agent configs (analyst, reviewer, compliance) -----
    agent_config_dict = {
        "name": "test_agent",
        "role": "test",
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_iter": 5,
        "require_grounding": False,
    }
    analyst_cfg = MagicMock()
    analyst_cfg.model_dump.return_value = agent_config_dict
    reviewer_cfg = MagicMock()
    reviewer_cfg.model_dump.return_value = agent_config_dict
    compliance_cfg = MagicMock()
    compliance_cfg.model_dump.return_value = agent_config_dict

    agents_config = MagicMock()
    agents_config.analyst = analyst_cfg
    agents_config.reviewer = reviewer_cfg
    agents_config.compliance = compliance_cfg
    instance.agents.return_value = agents_config

    # -- Guardrails config (escalation triggers) -----------
    guardrails = MagicMock()

    # Build trigger mocks matching Pydantic model interface
    trigger_names = [
        "high_value_loan",
        "deteriorating_sector",
        "compliance_failure",
        "low_reviewer_confidence",
        "grounding_failure",
        "low_average_grounding",
    ]
    _thresholds = {
        "high_value_loan": 500_000,
        "low_reviewer_confidence": 0.5,
        "low_average_grounding": 0.75,
    }
    triggers = []
    for name in trigger_names:
        t = MagicMock()
        t.name = name
        t.threshold = _thresholds.get(name)
        triggers.append(t)

    guardrails.escalation.triggers = triggers
    guardrails.grounding.max_retries = 2

    # PII patterns config
    guardrails.pii = MagicMock()
    guardrails.pii.patterns = []

    instance.guardrails.return_value = guardrails

    # -- App config (for retry policy) ---------------------
    processing = MagicMock()
    processing.llm_retry_attempts = 3
    processing.llm_retry_backoff_seconds = 1.0
    app_config = MagicMock()
    app_config.processing = processing
    instance.app.return_value = app_config

    return mock_cls


# ---------------------------------------------------------------------------
# Shared orchestrator factory
# ---------------------------------------------------------------------------


def _build_patched_orchestrator(
    analyst_response: AgentResponse,
    reviewer_response: AgentResponse,
    compliance_response: AgentResponse,
):
    """Construct a CreditRiskOrchestrator with agents and externals mocked.

    Returns (orchestrator, mock_create_agent_span, mock_create_grounding_span).
    """
    mock_config = _build_mock_config_loader()

    # Mock agent classes -- their execute() returns canned responses
    mock_analyst_cls = MagicMock()
    mock_analyst_instance = MagicMock()
    mock_analyst_instance.name = "financial_analyst"
    mock_analyst_instance.framework = "crewai"
    mock_analyst_instance.role = "analyst"
    mock_analyst_instance.execute = AsyncMock(return_value=analyst_response)
    mock_analyst_instance.to_audit_entry.return_value = {
        "agent": "financial_analyst",
        "confidence": analyst_response.confidence,
        "sources_count": len(analyst_response.sources_used),
        "tokens_used": analyst_response.tokens_used,
        "latency_ms": analyst_response.latency_ms,
    }
    mock_analyst_cls.return_value = mock_analyst_instance

    mock_reviewer_cls = MagicMock()
    mock_reviewer_instance = MagicMock()
    mock_reviewer_instance.name = "independent_reviewer"
    mock_reviewer_instance.framework = "crewai"
    mock_reviewer_instance.role = "reviewer"
    mock_reviewer_instance.execute = AsyncMock(return_value=reviewer_response)
    mock_reviewer_instance.to_audit_entry.return_value = {
        "agent": "independent_reviewer",
        "confidence": reviewer_response.confidence,
        "sources_count": len(reviewer_response.sources_used),
        "tokens_used": reviewer_response.tokens_used,
        "latency_ms": reviewer_response.latency_ms,
    }
    mock_reviewer_cls.return_value = mock_reviewer_instance

    mock_compliance_cls = MagicMock()
    mock_compliance_instance = MagicMock()
    mock_compliance_instance.name = "compliance_officer"
    mock_compliance_instance.framework = "autogen"
    mock_compliance_instance.role = "compliance"
    mock_compliance_instance.execute = AsyncMock(return_value=compliance_response)
    mock_compliance_instance.to_audit_entry.return_value = {
        "agent": "compliance_officer",
        "confidence": compliance_response.confidence,
        "sources_count": len(compliance_response.sources_used),
        "tokens_used": compliance_response.tokens_used,
        "latency_ms": compliance_response.latency_ms,
    }
    mock_compliance_cls.return_value = mock_compliance_instance

    # Mock PII detector
    mock_pii_cls = MagicMock()
    mock_pii_instance = MagicMock()
    mock_pii_scan = MagicMock()
    mock_pii_scan.pii_found = False
    mock_pii_scan.pii_types_detected = []
    mock_pii_instance.scan.return_value = mock_pii_scan
    mock_pii_cls.return_value = mock_pii_instance

    # Mock grounding nodes
    mock_grounding_dict = {
        "post_analyst": _mock_grounding_node("post_analyst"),
        "post_reviewer": _mock_grounding_node("post_reviewer"),
        "post_compliance": _mock_grounding_node("post_compliance"),
    }

    # Tracing mocks
    mock_create_agent_span = MagicMock(return_value=MagicMock())
    mock_end_agent_span = MagicMock()
    mock_create_grounding_span = MagicMock(return_value=MagicMock())
    mock_end_grounding_span = MagicMock()

    patches = {
        "src.agents.analyst.AnalystAgent": mock_analyst_cls,
        "src.agents.reviewer.ReviewerAgent": mock_reviewer_cls,
        "src.agents.compliance.ComplianceAgent": mock_compliance_cls,
        "src.agents.tools_adapter.get_analyst_tools": MagicMock(return_value=[]),
        "src.agents.tools_adapter.get_reviewer_tools": MagicMock(return_value=[]),
        "src.agents.tools_adapter.get_compliance_tools_autogen": MagicMock(
            return_value=[]
        ),
        "src.config.settings.ConfigLoader": mock_config,
        "src.orchestrator_nodes._get_grounding_nodes": MagicMock(
            return_value=mock_grounding_dict
        ),
        "src.guardrails.pii.PIIDetector": mock_pii_cls,
        "src.observability.tracing.create_agent_span": mock_create_agent_span,
        "src.observability.tracing.end_agent_span": mock_end_agent_span,
        "src.observability.tracing.create_grounding_span": mock_create_grounding_span,
        "src.observability.tracing.end_grounding_span": mock_end_grounding_span,
    }

    return patches, mock_create_agent_span, mock_create_grounding_span


# ===========================================================================
# E2E scenarios
# ===========================================================================


@pytest.mark.asyncio
async def test_e2e_clean_approval():
    """APPROVE + agree + compliance pass = APPROVED, no escalation."""
    analyst_resp = _make_analyst_response("APPROVE", credit_score=75, sector_outlook="stable")
    reviewer_resp = _make_reviewer_response(agrees=True, quality_score=0.85)
    compliance_resp = _make_compliance_response(overall_passed=True)

    patches, mock_agent_span, mock_grounding_span = _build_patched_orchestrator(
        analyst_resp, reviewer_resp, compliance_resp
    )

    import src.orchestrator_nodes as _onodes

    original_grounding_nodes = _onodes._grounding_nodes
    _onodes._grounding_nodes = None

    try:
        with (
            patch("src.agents.analyst.AnalystAgent", patches["src.agents.analyst.AnalystAgent"]),
            patch("src.agents.reviewer.ReviewerAgent", patches["src.agents.reviewer.ReviewerAgent"]),
            patch("src.agents.compliance.ComplianceAgent", patches["src.agents.compliance.ComplianceAgent"]),
            patch("src.agents.tools_adapter.get_analyst_tools", patches["src.agents.tools_adapter.get_analyst_tools"]),
            patch("src.agents.tools_adapter.get_reviewer_tools", patches["src.agents.tools_adapter.get_reviewer_tools"]),
            patch("src.agents.tools_adapter.get_compliance_tools_autogen", patches["src.agents.tools_adapter.get_compliance_tools_autogen"]),
            patch("src.config.settings.ConfigLoader", patches["src.config.settings.ConfigLoader"]),
            patch("src.orchestrator_nodes._get_grounding_nodes", patches["src.orchestrator_nodes._get_grounding_nodes"]),
            patch("src.guardrails.pii.PIIDetector", patches["src.guardrails.pii.PIIDetector"]),
            patch("src.observability.tracing.create_agent_span", patches["src.observability.tracing.create_agent_span"]),
            patch("src.observability.tracing.end_agent_span", patches["src.observability.tracing.end_agent_span"]),
            patch("src.observability.tracing.create_grounding_span", patches["src.observability.tracing.create_grounding_span"]),
            patch("src.observability.tracing.end_grounding_span", patches["src.observability.tracing.end_grounding_span"]),
        ):
            from src.orchestrator import CreditRiskOrchestrator

            orch = CreditRiskOrchestrator()
            result = await orch.run(_APPLICATION, request_id="E2E-APPROVE-001")

        # Decision
        assert result["final_decision"] == "APPROVED"
        assert result["requires_escalation"] is False
        assert result["confidence_score"] > 0

        # Audit trail
        assert isinstance(result["audit_trail"], list)
        assert len(result["audit_trail"]) > 0
        stages_in_trail = {
            e.get("stage") for e in result["audit_trail"] if isinstance(e, dict)
        }
        assert len(stages_in_trail) >= 2, f"Expected multi-stage audit, got {stages_in_trail}"

        # Handoff: analysis_result flows to review_node (credit_score preserved)
        assert "analysis_result" in result
        assert result["analysis_result"]["credit_score"] == 75

        # Handoff: review_result produced by reviewer (quality_score preserved)
        assert "review_result" in result
        assert result["review_result"]["quality_score"] == 0.85

        # Tracing: grounding spans created for each checkpoint
        assert mock_grounding_span.call_count >= 3, (
            f"Expected >= 3 grounding span calls, got {mock_grounding_span.call_count}"
        )

    finally:
        _onodes._grounding_nodes = original_grounding_nodes


@pytest.mark.asyncio
async def test_e2e_rejection():
    """REJECT + agree = REJECTED."""
    analyst_resp = _make_analyst_response("REJECT", credit_score=25, sector_outlook="negative")
    reviewer_resp = _make_reviewer_response(agrees=True, quality_score=0.90)
    compliance_resp = _make_compliance_response(overall_passed=True)

    patches, mock_agent_span, mock_grounding_span = _build_patched_orchestrator(
        analyst_resp, reviewer_resp, compliance_resp
    )

    import src.orchestrator_nodes as _onodes

    original_grounding_nodes = _onodes._grounding_nodes
    _onodes._grounding_nodes = None

    try:
        with (
            patch("src.agents.analyst.AnalystAgent", patches["src.agents.analyst.AnalystAgent"]),
            patch("src.agents.reviewer.ReviewerAgent", patches["src.agents.reviewer.ReviewerAgent"]),
            patch("src.agents.compliance.ComplianceAgent", patches["src.agents.compliance.ComplianceAgent"]),
            patch("src.agents.tools_adapter.get_analyst_tools", patches["src.agents.tools_adapter.get_analyst_tools"]),
            patch("src.agents.tools_adapter.get_reviewer_tools", patches["src.agents.tools_adapter.get_reviewer_tools"]),
            patch("src.agents.tools_adapter.get_compliance_tools_autogen", patches["src.agents.tools_adapter.get_compliance_tools_autogen"]),
            patch("src.config.settings.ConfigLoader", patches["src.config.settings.ConfigLoader"]),
            patch("src.orchestrator_nodes._get_grounding_nodes", patches["src.orchestrator_nodes._get_grounding_nodes"]),
            patch("src.guardrails.pii.PIIDetector", patches["src.guardrails.pii.PIIDetector"]),
            patch("src.observability.tracing.create_agent_span", patches["src.observability.tracing.create_agent_span"]),
            patch("src.observability.tracing.end_agent_span", patches["src.observability.tracing.end_agent_span"]),
            patch("src.observability.tracing.create_grounding_span", patches["src.observability.tracing.create_grounding_span"]),
            patch("src.observability.tracing.end_grounding_span", patches["src.observability.tracing.end_grounding_span"]),
        ):
            from src.orchestrator import CreditRiskOrchestrator

            orch = CreditRiskOrchestrator()
            result = await orch.run(_APPLICATION, request_id="E2E-REJECT-001")

        # Decision
        assert result["final_decision"] == "REJECTED"
        assert result["requires_escalation"] is False

        # Audit trail
        assert isinstance(result["audit_trail"], list)
        assert len(result["audit_trail"]) > 0

        # Handoff: analysis_result.recommendation persists
        assert result["analysis_result"]["recommendation"] == "REJECT"

        # Handoff: reviewer agrees with rejection
        assert result["review_result"]["agrees_with_analyst"] is True

    finally:
        _onodes._grounding_nodes = original_grounding_nodes


@pytest.mark.asyncio
async def test_e2e_referral_reviewer_disagrees():
    """APPROVE + disagree + compliance pass = REFERRED_TO_UNDERWRITER."""
    analyst_resp = _make_analyst_response("APPROVE", credit_score=60)
    reviewer_resp = _make_reviewer_response(
        agrees=False, quality_score=0.50, confidence_level="MEDIUM"
    )
    compliance_resp = _make_compliance_response(overall_passed=True)

    patches, mock_agent_span, mock_grounding_span = _build_patched_orchestrator(
        analyst_resp, reviewer_resp, compliance_resp
    )

    import src.orchestrator_nodes as _onodes

    original_grounding_nodes = _onodes._grounding_nodes
    _onodes._grounding_nodes = None

    try:
        with (
            patch("src.agents.analyst.AnalystAgent", patches["src.agents.analyst.AnalystAgent"]),
            patch("src.agents.reviewer.ReviewerAgent", patches["src.agents.reviewer.ReviewerAgent"]),
            patch("src.agents.compliance.ComplianceAgent", patches["src.agents.compliance.ComplianceAgent"]),
            patch("src.agents.tools_adapter.get_analyst_tools", patches["src.agents.tools_adapter.get_analyst_tools"]),
            patch("src.agents.tools_adapter.get_reviewer_tools", patches["src.agents.tools_adapter.get_reviewer_tools"]),
            patch("src.agents.tools_adapter.get_compliance_tools_autogen", patches["src.agents.tools_adapter.get_compliance_tools_autogen"]),
            patch("src.config.settings.ConfigLoader", patches["src.config.settings.ConfigLoader"]),
            patch("src.orchestrator_nodes._get_grounding_nodes", patches["src.orchestrator_nodes._get_grounding_nodes"]),
            patch("src.guardrails.pii.PIIDetector", patches["src.guardrails.pii.PIIDetector"]),
            patch("src.observability.tracing.create_agent_span", patches["src.observability.tracing.create_agent_span"]),
            patch("src.observability.tracing.end_agent_span", patches["src.observability.tracing.end_agent_span"]),
            patch("src.observability.tracing.create_grounding_span", patches["src.observability.tracing.create_grounding_span"]),
            patch("src.observability.tracing.end_grounding_span", patches["src.observability.tracing.end_grounding_span"]),
        ):
            from src.orchestrator import CreditRiskOrchestrator

            orch = CreditRiskOrchestrator()
            result = await orch.run(_APPLICATION, request_id="E2E-REFER-001")

        # Decision
        assert result["final_decision"] == "REFERRED_TO_UNDERWRITER"

        # Audit trail
        assert isinstance(result["audit_trail"], list)
        assert len(result["audit_trail"]) > 0

        # Handoff: reviewer disagreement preserved through pipeline
        assert result["review_result"]["agrees_with_analyst"] is False

    finally:
        _onodes._grounding_nodes = original_grounding_nodes


@pytest.mark.asyncio
async def test_e2e_compliance_override_escalation():
    """Compliance fails -> compliance_failure trigger fires -> ESCALATED."""
    analyst_resp = _make_analyst_response("APPROVE", credit_score=72)
    reviewer_resp = _make_reviewer_response(agrees=True, quality_score=0.85)
    compliance_resp = _make_compliance_response(overall_passed=False)

    patches, mock_agent_span, mock_grounding_span = _build_patched_orchestrator(
        analyst_resp, reviewer_resp, compliance_resp
    )

    import src.orchestrator_nodes as _onodes

    original_grounding_nodes = _onodes._grounding_nodes
    _onodes._grounding_nodes = None

    try:
        with (
            patch("src.agents.analyst.AnalystAgent", patches["src.agents.analyst.AnalystAgent"]),
            patch("src.agents.reviewer.ReviewerAgent", patches["src.agents.reviewer.ReviewerAgent"]),
            patch("src.agents.compliance.ComplianceAgent", patches["src.agents.compliance.ComplianceAgent"]),
            patch("src.agents.tools_adapter.get_analyst_tools", patches["src.agents.tools_adapter.get_analyst_tools"]),
            patch("src.agents.tools_adapter.get_reviewer_tools", patches["src.agents.tools_adapter.get_reviewer_tools"]),
            patch("src.agents.tools_adapter.get_compliance_tools_autogen", patches["src.agents.tools_adapter.get_compliance_tools_autogen"]),
            patch("src.config.settings.ConfigLoader", patches["src.config.settings.ConfigLoader"]),
            patch("src.orchestrator_nodes._get_grounding_nodes", patches["src.orchestrator_nodes._get_grounding_nodes"]),
            patch("src.guardrails.pii.PIIDetector", patches["src.guardrails.pii.PIIDetector"]),
            patch("src.observability.tracing.create_agent_span", patches["src.observability.tracing.create_agent_span"]),
            patch("src.observability.tracing.end_agent_span", patches["src.observability.tracing.end_agent_span"]),
            patch("src.observability.tracing.create_grounding_span", patches["src.observability.tracing.create_grounding_span"]),
            patch("src.observability.tracing.end_grounding_span", patches["src.observability.tracing.end_grounding_span"]),
        ):
            from src.orchestrator import CreditRiskOrchestrator

            orch = CreditRiskOrchestrator()
            result = await orch.run(_APPLICATION, request_id="E2E-ESCALATE-001")

        # Decision: compliance failure triggers escalation
        assert result["final_decision"] == "ESCALATED"
        assert result["requires_escalation"] is True

        # Audit trail
        assert isinstance(result["audit_trail"], list)
        assert len(result["audit_trail"]) > 0

        # Handoff: compliance_result.overall_passed=False persists
        assert result["compliance_result"]["overall_passed"] is False

        # Tracing: grounding spans created for all stages
        assert mock_grounding_span.call_count >= 3, (
            f"Expected >= 3 grounding span calls, got {mock_grounding_span.call_count}"
        )

    finally:
        _onodes._grounding_nodes = original_grounding_nodes
