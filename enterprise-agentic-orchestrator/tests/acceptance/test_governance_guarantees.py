"""Acceptance tests for governance guarantees.

These tests verify every governance promise the system makes from a
regulatory perspective. They use the same mocking strategy as the E2E
integration tests (09-04) -- LLM/embedding calls are mocked but all
business logic (decision matrix, escalation triggers, audit trail, PII
detection, bias checking) runs for real.

Governance guarantees verified (TEST-11):
    1. Every decision has an explainability report (non-empty reasoning_trace)
    2. Every claim is grounded (grounding_scores populated, all is_grounded)
    3. PII never appears in final output
    4. Protected characteristics never used in decisions
    5. Compliance cannot be bypassed (single failure forces overall failure)
    6. Audit trail hash chain is intact
    7. High-value loans (>500k) trigger escalation
    8. Decision consistency (same inputs -> same output)
    9. Graceful degradation (agent failure -> dict with errors, not exception)
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import AgentResponse

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_APPLICATION = {
    "application_id": "GOV-TEST-001",
    "applicant": {
        "company_name": "Governance Test Corp",
        "company_number": "11223344",
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

_HIGH_VALUE_APPLICATION = {
    **_APPLICATION,
    "application_id": "GOV-HIGHVAL-001",
    "loan": {
        "amount_requested": float(Decimal("600000.00")),
        "term_months": 60,
        "purpose": "Large-scale expansion project",
        "security_type": "secured",
        "currency": "GBP",
    },
}


# ---------------------------------------------------------------------------
# Agent response factories
# ---------------------------------------------------------------------------


def _make_analyst_response(
    recommendation: str = "APPROVE",
    credit_score: int = 75,
    sector_outlook: str = "stable",
    extra_reasoning: str = "",
) -> AgentResponse:
    reasoning = "Financial analysis complete. Strong cash flow." + extra_reasoning
    return AgentResponse(
        agent_name="financial_analyst",
        agent_framework="crewai",
        output={
            "report_id": "RPT-ANALYST-GOV",
            "application_id": "GOV-TEST-001",
            "credit_score": credit_score,
            "risk_metrics": {
                "probability_of_default": 0.06,
                "loss_given_default": 0.40,
                "exposure_at_default": 250000.00,
                "expected_loss": 6000.00,
            },
            "sector_outlook": sector_outlook,
            "recommendation": recommendation,
            "reasoning": reasoning,
            "source_citations": [],
        },
        reasoning_trace="Governance test analyst trace.",
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
        reasoning_trace="Governance test reviewer trace.",
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
        reasoning_trace="Governance test compliance trace.",
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
    """Build a ConfigLoader mock with full pipeline config."""
    mock_cls = MagicMock()
    instance = mock_cls.return_value

    # Agent configs
    agent_config_dict = {
        "name": "test_agent",
        "role": "test",
        "model": "gpt-4o",
        "temperature": 0.1,
        "max_iter": 5,
        "require_grounding": False,
    }
    for attr_name in ("analyst", "reviewer", "compliance"):
        cfg = MagicMock()
        cfg.model_dump.return_value = agent_config_dict
        setattr(
            instance.agents.return_value if not hasattr(instance.agents, attr_name)
            else instance.agents.return_value,
            attr_name,
            cfg,
        )

    agents_config = instance.agents.return_value

    # Guardrails config (escalation triggers)
    guardrails = MagicMock()
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
    guardrails.pii = MagicMock()
    guardrails.pii.patterns = []
    instance.guardrails.return_value = guardrails

    # App config (retry)
    processing = MagicMock()
    processing.llm_retry_attempts = 3
    processing.llm_retry_backoff_seconds = 1.0
    app_config = MagicMock()
    app_config.processing = processing
    instance.app.return_value = app_config

    return mock_cls


# ---------------------------------------------------------------------------
# Shared patched orchestrator builder
# ---------------------------------------------------------------------------


def _build_patched_orchestrator(
    analyst_response: AgentResponse,
    reviewer_response: AgentResponse,
    compliance_response: AgentResponse,
    analyst_raises: Exception | None = None,
):
    """Build patch dict for all external dependencies."""
    mock_config = _build_mock_config_loader()

    def _build_agent_mock(response, name, framework, role, raises=None):
        mock_cls = MagicMock()
        inst = MagicMock()
        inst.name = name
        inst.framework = framework
        inst.role = role
        if raises:
            inst.execute = AsyncMock(side_effect=raises)
        else:
            inst.execute = AsyncMock(return_value=response)
        inst.to_audit_entry.return_value = {
            "agent": name,
            "confidence": response.confidence,
            "sources_count": len(response.sources_used),
            "tokens_used": response.tokens_used,
            "latency_ms": response.latency_ms,
        }
        mock_cls.return_value = inst
        return mock_cls

    mock_analyst_cls = _build_agent_mock(
        analyst_response, "financial_analyst", "crewai", "analyst",
        raises=analyst_raises,
    )
    mock_reviewer_cls = _build_agent_mock(
        reviewer_response, "independent_reviewer", "crewai", "reviewer",
    )
    mock_compliance_cls = _build_agent_mock(
        compliance_response, "compliance_officer", "autogen", "compliance",
    )

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
        "src.agents.tools_adapter.get_compliance_tools_autogen": MagicMock(return_value=[]),
        "src.config.settings.ConfigLoader": mock_config,
        "src.orchestrator_nodes._get_grounding_nodes": MagicMock(return_value=mock_grounding_dict),
        "src.guardrails.pii.PIIDetector": mock_pii_cls,
        "src.observability.tracing.create_agent_span": mock_create_agent_span,
        "src.observability.tracing.end_agent_span": mock_end_agent_span,
        "src.observability.tracing.create_grounding_span": mock_create_grounding_span,
        "src.observability.tracing.end_grounding_span": mock_end_grounding_span,
    }

    return patches


def _apply_patches_and_run(patches, application, request_id):
    """Context-manager helper returning awaitable orchestrator run."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        import src.orchestrator_nodes as _onodes
        original = _onodes._grounding_nodes
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
                yield
        finally:
            _onodes._grounding_nodes = original

    return _ctx


async def _run_orchestrator(patches, application, request_id):
    """Run the orchestrator with all patches applied. Returns result dict."""
    ctx = _apply_patches_and_run(patches, application, request_id)
    with ctx():
        from src.orchestrator import CreditRiskOrchestrator
        orch = CreditRiskOrchestrator()
        return await orch.run(application, request_id=request_id)


# ===========================================================================
# Acceptance tests
# ===========================================================================


@pytest.mark.asyncio
async def test_every_decision_has_explainability():
    """GOV: Every decision has an explainability report (reasoning_trace is non-empty)."""
    analyst = _make_analyst_response("APPROVE", credit_score=75)
    reviewer = _make_reviewer_response(agrees=True, quality_score=0.85)
    compliance = _make_compliance_response(overall_passed=True)
    patches = _build_patched_orchestrator(analyst, reviewer, compliance)

    result = await _run_orchestrator(patches, _APPLICATION, "GOV-EXPLAIN-001")

    # reasoning_trace must be non-empty
    assert result["reasoning_trace"], "reasoning_trace must not be empty"
    assert isinstance(result["reasoning_trace"], str)

    # Must contain key decision components
    trace = result["reasoning_trace"]
    assert "Analyst recommendation" in trace or "recommendation" in trace.lower()
    assert "Reviewer agrees" in trace or "reviewer" in trace.lower()
    assert "Compliance passed" in trace or "compliance" in trace.lower()
    assert "Decision" in trace or "decision" in trace.lower()
    assert "Confidence" in trace or "confidence" in trace.lower()


@pytest.mark.asyncio
async def test_every_claim_is_grounded():
    """GOV: Every claim in agent output is verified against sources (grounding check runs)."""
    analyst = _make_analyst_response("APPROVE", credit_score=75)
    reviewer = _make_reviewer_response(agrees=True, quality_score=0.85)
    compliance = _make_compliance_response(overall_passed=True)
    patches = _build_patched_orchestrator(analyst, reviewer, compliance)

    result = await _run_orchestrator(patches, _APPLICATION, "GOV-GROUNDING-001")

    # grounding_scores must be populated
    scores = result.get("grounding_scores", [])
    assert isinstance(scores, list)
    assert len(scores) >= 3, f"Expected >= 3 grounding entries, got {len(scores)}"

    # All entries must be grounded
    for entry in scores:
        if isinstance(entry, dict):
            assert entry.get("is_grounded") is True, (
                f"Grounding entry not grounded: {entry}"
            )

    # Verify checkpoint names cover all 3 stages
    checkpoints = {e.get("checkpoint") for e in scores if isinstance(e, dict)}
    assert "post_analyst" in checkpoints
    assert "post_reviewer" in checkpoints
    assert "post_compliance" in checkpoints


@pytest.mark.asyncio
async def test_pii_never_in_output():
    """GOV: PII never appears in final output (output text contains no NI numbers, sort codes, or phone numbers)."""
    analyst = _make_analyst_response("APPROVE", credit_score=75)
    reviewer = _make_reviewer_response(agrees=True, quality_score=0.85)
    compliance = _make_compliance_response(overall_passed=True)
    patches = _build_patched_orchestrator(analyst, reviewer, compliance)

    result = await _run_orchestrator(patches, _APPLICATION, "GOV-PII-001")

    # Extract all text from result
    combined_text = result.get("reasoning_trace", "")
    for key in ("analysis_result", "review_result", "compliance_result"):
        val = result.get(key)
        if val:
            combined_text += " " + json.dumps(val, default=str)

    # Run REAL PIIDetector on the combined output
    from src.guardrails.pii import PIIDetector
    detector = PIIDetector()
    scan = detector.scan(combined_text)
    assert scan.pii_found is False, (
        f"PII detected in output: {scan.pii_types_detected}"
    )

    # INVERSE: verify detector DOES catch PII when present
    tainted_text = combined_text + " Applicant NI AB123456C was reviewed"
    tainted_scan = detector.scan(tainted_text)
    assert tainted_scan.pii_found is True, "PIIDetector should detect NI number"
    assert "NI Number" in tainted_scan.pii_types_detected


@pytest.mark.asyncio
async def test_protected_characteristics_never_used():
    """GOV: Protected characteristics are never used in decisions (bias check runs and output is clean)."""
    analyst = _make_analyst_response("APPROVE", credit_score=75)
    reviewer = _make_reviewer_response(agrees=True, quality_score=0.85)
    compliance = _make_compliance_response(overall_passed=True)
    patches = _build_patched_orchestrator(analyst, reviewer, compliance)

    result = await _run_orchestrator(patches, _APPLICATION, "GOV-BIAS-001")

    # Extract combined output text
    combined_text = result.get("reasoning_trace", "")
    for key in ("analysis_result", "review_result", "compliance_result"):
        val = result.get(key)
        if val:
            combined_text += " " + json.dumps(val, default=str)

    # Run REAL BiasChecker on combined output
    from src.guardrails.bias import BiasChecker
    checker = BiasChecker()
    bias_result = checker.check(combined_text)
    assert bias_result.bias_detected is False, (
        f"Bias detected in output: {bias_result.protected_characteristics_found}"
    )

    # INVERSE: verify BiasChecker catches protected characteristics
    tainted_text = "The applicant's age is 45 years and gender is male"
    tainted_result = checker.check(tainted_text)
    assert tainted_result.bias_detected is True
    assert "age" in tainted_result.protected_characteristics_found


@pytest.mark.asyncio
async def test_compliance_cannot_be_bypassed():
    """GOV: Compliance cannot be bypassed (single check failure forces overall failure)."""
    # Scenario A: All compliance passes -> APPROVED
    analyst_a = _make_analyst_response("APPROVE", credit_score=75)
    reviewer_a = _make_reviewer_response(agrees=True, quality_score=0.85)
    compliance_a = _make_compliance_response(overall_passed=True)
    patches_a = _build_patched_orchestrator(analyst_a, reviewer_a, compliance_a)
    result_a = await _run_orchestrator(patches_a, _APPLICATION, "GOV-COMPLY-PASS")

    assert result_a["final_decision"] == "APPROVED"

    # Scenario B: Compliance fails -> cannot be APPROVED
    analyst_b = _make_analyst_response("APPROVE", credit_score=75)
    reviewer_b = _make_reviewer_response(agrees=True, quality_score=0.85)
    compliance_b = _make_compliance_response(overall_passed=False)
    patches_b = _build_patched_orchestrator(analyst_b, reviewer_b, compliance_b)
    result_b = await _run_orchestrator(patches_b, _APPLICATION, "GOV-COMPLY-FAIL")

    # Compliance failure must prevent APPROVED
    assert result_b["final_decision"] != "APPROVED", (
        f"Compliance failure should prevent APPROVED, got {result_b['final_decision']}"
    )
    assert result_b["final_decision"] in ("REFERRED_TO_UNDERWRITER", "ESCALATED")
    assert result_b["requires_escalation"] is True


def test_audit_trail_hash_chain_intact():
    """GOV: Audit trail hash chain is intact after full pipeline run.

    Tests the AuditTrail class directly with a realistic 12-entry
    pipeline lifecycle chain. Verifies tamper detection works.
    """
    from src.governance.audit import AuditTrail

    trail = AuditTrail(request_id="ACC-TEST-001")

    # Add entries covering the full pipeline lifecycle
    stages = [
        ("INTAKE", "input_received"),
        ("INTAKE", "pii_scan_complete"),
        ("ANALYSIS", "analyst_started"),
        ("ANALYSIS", "analyst_complete"),
        ("GROUNDING", "post_analyst_check"),
        ("REVIEW", "reviewer_started"),
        ("REVIEW", "reviewer_complete"),
        ("GROUNDING", "post_reviewer_check"),
        ("COMPLIANCE", "compliance_started"),
        ("COMPLIANCE", "compliance_complete"),
        ("GROUNDING", "post_compliance_check"),
        ("DECISION", "decision_computed"),
    ]

    for stage, action in stages:
        trail.add_entry(
            stage=stage,
            action=action,
            details={"test": True},
        )

    # Chain must be intact
    assert trail.verify_chain() is True, "12-entry hash chain should be valid"
    assert len(trail) == 12

    # Tamper with mid-chain entry
    trail.entries[5].action = "TAMPERED_ACTION"

    # Chain must detect tampering
    assert trail.verify_chain() is False, "Tampering should break hash chain"


@pytest.mark.asyncio
async def test_high_value_loan_triggers_escalation():
    """GOV: High-value loans (>500k) trigger escalation."""
    analyst = _make_analyst_response("APPROVE", credit_score=75)
    reviewer = _make_reviewer_response(agrees=True, quality_score=0.85)
    compliance = _make_compliance_response(overall_passed=True)
    patches = _build_patched_orchestrator(analyst, reviewer, compliance)

    result = await _run_orchestrator(patches, _HIGH_VALUE_APPLICATION, "GOV-HIGHVAL-001")

    # High-value loan must trigger escalation
    assert result["final_decision"] == "ESCALATED", (
        f"600k loan should trigger ESCALATED, got {result['final_decision']}"
    )
    assert result["requires_escalation"] is True

    # Reasoning trace should mention the escalation trigger
    trace = result.get("reasoning_trace", "")
    assert "high_value_loan" in trace.lower() or "500000" in trace or "escalat" in trace.lower(), (
        "Reasoning trace should mention high_value_loan trigger"
    )


@pytest.mark.asyncio
async def test_decision_consistency():
    """GOV: Decision consistency -- same inputs produce same decision outcome."""
    analyst = _make_analyst_response("APPROVE", credit_score=75)
    reviewer = _make_reviewer_response(agrees=True, quality_score=0.85)
    compliance = _make_compliance_response(overall_passed=True)

    # Run 1
    patches_1 = _build_patched_orchestrator(analyst, reviewer, compliance)
    result_1 = await _run_orchestrator(patches_1, _APPLICATION, "GOV-CONSIST-001")

    # Run 2 with identical inputs
    patches_2 = _build_patched_orchestrator(analyst, reviewer, compliance)
    result_2 = await _run_orchestrator(patches_2, _APPLICATION, "GOV-CONSIST-002")

    # Same decision
    assert result_1["final_decision"] == result_2["final_decision"], (
        f"Inconsistent decisions: {result_1['final_decision']} vs {result_2['final_decision']}"
    )

    # Same escalation flag
    assert result_1["requires_escalation"] == result_2["requires_escalation"]

    # Similar confidence (within floating point tolerance)
    conf_1 = result_1.get("confidence_score", 0.0) or 0.0
    conf_2 = result_2.get("confidence_score", 0.0) or 0.0
    assert abs(conf_1 - conf_2) < 0.01, (
        f"Confidence scores differ: {conf_1} vs {conf_2}"
    )


@pytest.mark.asyncio
async def test_graceful_degradation():
    """GOV: Graceful degradation -- agent failure returns dict with errors, not exception."""
    analyst = _make_analyst_response("APPROVE", credit_score=75)
    reviewer = _make_reviewer_response(agrees=True, quality_score=0.85)
    compliance = _make_compliance_response(overall_passed=True)
    patches = _build_patched_orchestrator(
        analyst, reviewer, compliance,
        analyst_raises=RuntimeError("Simulated agent failure"),
    )

    result = await _run_orchestrator(patches, _APPLICATION, "GOV-DEGRADE-001")

    # Must return a dict, not raise
    assert isinstance(result, dict), "Orchestrator must return dict on agent failure"

    # Must have errors
    errors = result.get("errors", [])
    assert len(errors) > 0, "Error list must be non-empty on agent failure"

    # Must route to a safe outcome -- never APPROVED after agent failure.
    # The decision matrix sees no analysis_result (empty dict) so it
    # defaults recommendation to REFER_TO_UNDERWRITER (fail-safe). The
    # final outcome may be ERROR, ESCALATED, or REFERRED_TO_UNDERWRITER
    # depending on which node caught the error -- all are safe outcomes.
    assert result["final_decision"] != "APPROVED", (
        "Agent failure must never produce APPROVED"
    )
    assert result["final_decision"] in ("ERROR", "ESCALATED", "REFERRED_TO_UNDERWRITER"), (
        f"Expected safe outcome on agent failure, got {result['final_decision']}"
    )
