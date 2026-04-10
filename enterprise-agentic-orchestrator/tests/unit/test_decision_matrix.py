"""Unit tests for the deterministic decision matrix (ORCH-04).

Covers all 9 rows of the matrix, edge cases (unknown recommendation,
case-insensitive input), confidence computation, and the _clamp_unit helper.
"""

from __future__ import annotations

import pytest

from src.orchestrator_decision import (
    _clamp_unit,
    _compute_confidence,
    apply_decision_matrix,
)


# ---------------------------------------------------------------------------
# Decision matrix: all 9 rows
# ---------------------------------------------------------------------------


class TestDecisionMatrix:
    """Verify every row of the 9-row deterministic decision matrix."""

    def test_approve_agree_pass(self):
        result = apply_decision_matrix("APPROVE", True, True)
        assert result == "APPROVED"

    def test_approve_agree_fail(self):
        result = apply_decision_matrix("APPROVE", True, False)
        assert result == "REFERRED_TO_UNDERWRITER"

    def test_approve_disagree_pass(self):
        result = apply_decision_matrix("APPROVE", False, True)
        assert result == "REFERRED_TO_UNDERWRITER"

    def test_approve_disagree_fail(self):
        result = apply_decision_matrix("APPROVE", False, False)
        assert result == "REJECTED"

    def test_reject_agree_pass(self):
        result = apply_decision_matrix("REJECT", True, True)
        assert result == "REJECTED"

    def test_reject_agree_fail(self):
        result = apply_decision_matrix("REJECT", True, False)
        assert result == "REJECTED"

    def test_reject_disagree_pass(self):
        result = apply_decision_matrix("REJECT", False, True)
        assert result == "REFERRED_TO_UNDERWRITER"

    def test_reject_disagree_fail(self):
        result = apply_decision_matrix("REJECT", False, False)
        assert result == "REJECTED"

    def test_refer_agree_pass(self):
        result = apply_decision_matrix("REFER_TO_UNDERWRITER", True, True)
        assert result == "REFERRED_TO_UNDERWRITER"

    def test_refer_disagree_fail(self):
        result = apply_decision_matrix("REFER_TO_UNDERWRITER", False, False)
        assert result == "REFERRED_TO_UNDERWRITER"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestDecisionMatrixEdgeCases:
    """Edge cases: unknown recommendations, case insensitivity."""

    def test_unknown_recommendation_defaults_to_referred(self):
        """Unknown analyst recommendation fails closed to human review."""
        result = apply_decision_matrix("BANANA", True, True)
        assert result == "REFERRED_TO_UNDERWRITER"

    def test_empty_recommendation_defaults_to_referred(self):
        result = apply_decision_matrix("", True, True)
        assert result == "REFERRED_TO_UNDERWRITER"

    def test_case_insensitive_approve(self):
        """Lowercase 'approve' should work just like 'APPROVE'."""
        result = apply_decision_matrix("approve", True, True)
        assert result == "APPROVED"

    def test_case_insensitive_reject(self):
        result = apply_decision_matrix("reject", True, True)
        assert result == "REJECTED"

    def test_case_insensitive_refer(self):
        result = apply_decision_matrix("refer_to_underwriter", False, False)
        assert result == "REFERRED_TO_UNDERWRITER"

    def test_whitespace_stripped(self):
        result = apply_decision_matrix("  APPROVE  ", True, True)
        assert result == "APPROVED"


# ---------------------------------------------------------------------------
# _compute_confidence
# ---------------------------------------------------------------------------


class TestComputeConfidence:
    """Confidence score computation from four signals."""

    def test_known_inputs_produce_expected_average(self):
        """credit_score=72 -> 0.72, quality_score=0.85, passed=True -> 1.0,
        average grounding=0.80. Expected average = 0.8425."""
        result = _compute_confidence(
            analysis_result={"credit_score": 72},
            review_result={"quality_score": 0.85},
            compliance_result={"overall_passed": True},
            grounding_scores=[
                {"score": 0.75},
                {"score": 0.85},
            ],
        )
        # average of [0.72, 0.85, 1.0, 0.80] = 0.8425
        assert abs(result - 0.8425) < 0.001

    def test_missing_values_fall_back_to_neutral(self):
        """Empty dicts -> each signal falls back to its default.

        credit_score missing -> default 50 -> 0.5
        quality_score missing -> default 0.5
        overall_passed missing -> default False -> 0.0
        grounding_scores empty -> 0.5

        average = (0.5 + 0.5 + 0.0 + 0.5) / 4 = 0.375
        """
        result = _compute_confidence(
            analysis_result={},
            review_result={},
            compliance_result={},
            grounding_scores=[],
        )
        assert abs(result - 0.375) < 0.001

    def test_all_perfect_signals(self):
        """All perfect inputs -> confidence = 1.0."""
        result = _compute_confidence(
            analysis_result={"credit_score": 100},
            review_result={"quality_score": 1.0},
            compliance_result={"overall_passed": True},
            grounding_scores=[{"score": 1.0}],
        )
        assert abs(result - 1.0) < 0.001

    def test_all_worst_signals(self):
        """All worst inputs -> confidence near 0."""
        result = _compute_confidence(
            analysis_result={"credit_score": 0},
            review_result={"quality_score": 0.0},
            compliance_result={"overall_passed": False},
            grounding_scores=[{"score": 0.0}],
        )
        assert abs(result - 0.0) < 0.001


# ---------------------------------------------------------------------------
# _clamp_unit
# ---------------------------------------------------------------------------


class TestClampUnit:
    """_clamp_unit clamps values to [0.0, 1.0]."""

    def test_below_zero(self):
        assert _clamp_unit(-0.5) == 0.0

    def test_above_one(self):
        assert _clamp_unit(1.5) == 1.0

    def test_within_range(self):
        assert _clamp_unit(0.5) == 0.5

    def test_exactly_zero(self):
        assert _clamp_unit(0.0) == 0.0

    def test_exactly_one(self):
        assert _clamp_unit(1.0) == 1.0
