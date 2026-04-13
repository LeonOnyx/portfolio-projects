"""Tests for Phase 11 core logic hardening fixes (P0-3, P0-6, P0-7, P1-15, P1-21, P1-22).

Covers:
- P0-3: Audit trail hash chain includes details field
- P0-6: Sector lookup handles empty results gracefully
- P0-7: Escalation thresholds loaded from guardrails.yaml config
- P1-15: ComplianceReport rejects empty checks list
- P1-21: RAG tool alpha default aligned at 0.7
- P1-22: Credit scorer rejects empty financials
"""

import inspect

import pytest


# -----------------------------------------------------------------------
# P0-3: Audit trail hash chain must cover details field
# -----------------------------------------------------------------------


def test_audit_hash_chain_covers_details():
    """P0-3: Tampering with details must break hash chain verification."""
    from src.governance.audit import AuditTrail

    trail = AuditTrail("test-123")
    trail.add_entry(stage="ANALYSIS", action="test", details={"score": 85})
    trail.add_entry(stage="REVIEW", action="test", details={"agreed": True})
    assert trail.verify_chain() is True

    # Tamper with details
    trail.entries[0].details["score"] = 999
    assert trail.verify_chain() is False, "verify_chain must detect details tampering"


def test_audit_hash_chain_untampered():
    """P0-3: Untampered chain must still verify correctly."""
    from src.governance.audit import AuditTrail

    trail = AuditTrail("test-456")
    for i in range(5):
        trail.add_entry(
            stage=f"STAGE_{i}", action=f"action_{i}", details={"i": i}
        )
    assert trail.verify_chain() is True


# -----------------------------------------------------------------------
# P1-15: ComplianceReport must reject empty checks list
# -----------------------------------------------------------------------


def test_compliance_report_rejects_empty_checks():
    """P1-15: ComplianceReport with empty checks must raise ValueError."""
    from src.models.reports import ComplianceReport

    with pytest.raises(ValueError, match="at least one"):
        ComplianceReport(
            application_id="test", checks=[], overall_passed=True
        )


def test_compliance_report_rejects_empty_checks_false():
    """P1-15: Even with overall_passed=False, empty checks must be rejected."""
    from src.models.reports import ComplianceReport

    with pytest.raises(ValueError, match="at least one"):
        ComplianceReport(
            application_id="test", checks=[], overall_passed=False
        )


# -----------------------------------------------------------------------
# P1-22: Credit scorer must reject empty financials
# -----------------------------------------------------------------------


def test_credit_scorer_rejects_empty_financials():
    """P1-22: calculate_credit_score must raise ValueError on empty financials."""
    from src.tools.credit_scorer import calculate_credit_score

    with pytest.raises(ValueError, match="at least one year"):
        calculate_credit_score(
            financials=[], years_trading=5, sector_outlook="stable"
        )


# -----------------------------------------------------------------------
# P0-6: Sector lookup must handle empty results gracefully
# -----------------------------------------------------------------------


def test_sector_lookup_handles_empty_results():
    """P0-6: Sector lookup must not crash on empty results."""
    from unittest.mock import patch

    from src.tools.sector_lookup import lookup_sector

    mock_result = {
        "query": "test",
        "source_collection": "SectorAnalysis",
        "result_count": 0,
        "results": [],
    }
    with patch(
        "src.tools.sector_lookup.rag_sector_analysis",
        return_value=mock_result,
    ):
        result = lookup_sector("Technology")
        assert result.outlook == "unknown"
        assert result.error is not None


def test_sector_lookup_handles_nonempty_count_empty_list():
    """P0-6: Sector lookup handles result_count > 0 but empty results list."""
    from unittest.mock import patch

    from src.tools.sector_lookup import lookup_sector

    mock_result = {
        "query": "test",
        "source_collection": "SectorAnalysis",
        "result_count": 3,
        "results": [],
    }
    with patch(
        "src.tools.sector_lookup.rag_sector_analysis",
        return_value=mock_result,
    ):
        result = lookup_sector("Construction")
        assert result.outlook == "unknown"
        assert result.source_count == 0


# -----------------------------------------------------------------------
# P0-7: Escalation thresholds from guardrails.yaml config
# -----------------------------------------------------------------------


def test_escalation_thresholds_from_config():
    """P0-7: Escalation thresholds must be loaded from guardrails.yaml."""
    from src.config.settings import ConfigLoader

    triggers = ConfigLoader().guardrails().escalation.triggers
    threshold_map = {
        t.name: t.threshold for t in triggers if t.threshold is not None
    }
    assert "high_value_loan" in threshold_map, (
        "high_value_loan threshold missing from config"
    )
    assert "low_reviewer_confidence" in threshold_map
    assert "low_average_grounding" in threshold_map
    assert threshold_map["high_value_loan"] == 500000
    assert threshold_map["low_reviewer_confidence"] == 0.5
    assert threshold_map["low_average_grounding"] == 0.75


# -----------------------------------------------------------------------
# P1-21: RAG tool alpha default must be 0.7
# -----------------------------------------------------------------------


def test_rag_tools_alpha_default_is_0_7():
    """P1-21: RAG tool alpha default must be 0.7, matching registry docs."""
    from src.tools.rag_tools import rag_financial_lookup

    sig = inspect.signature(rag_financial_lookup)
    assert sig.parameters["alpha"].default == 0.7
