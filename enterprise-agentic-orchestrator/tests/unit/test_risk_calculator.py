"""Unit tests for risk calculator (TEST-09).

Tests _derive_risk_rating and calculate_risk_metrics with good/poor
credit scores, secured loans, boundary values, and return type validation.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.tools.risk_calculator import (
    RiskCalculationResult,
    _derive_risk_rating,
    calculate_risk_metrics,
)


# ---------------------------------------------------------------------------
# Risk rating derivation
# ---------------------------------------------------------------------------

class TestDeriveRiskRating:
    """Tests for _derive_risk_rating() threshold logic."""

    def test_low_risk(self):
        # pd < 0.05
        assert _derive_risk_rating(0.03) == "LOW"

    def test_medium_risk(self):
        # 0.05 <= pd < 0.15
        assert _derive_risk_rating(0.10) == "MEDIUM"

    def test_high_risk(self):
        # 0.15 <= pd < 0.30
        assert _derive_risk_rating(0.25) == "HIGH"

    def test_very_high_risk(self):
        # pd >= 0.30
        assert _derive_risk_rating(0.40) == "VERY_HIGH"

    def test_boundary_low_medium(self):
        # pd exactly 0.05 -> not < 0.05, so MEDIUM
        assert _derive_risk_rating(0.05) == "MEDIUM"

    def test_boundary_medium_high(self):
        assert _derive_risk_rating(0.15) == "HIGH"

    def test_boundary_high_very_high(self):
        assert _derive_risk_rating(0.30) == "VERY_HIGH"


# ---------------------------------------------------------------------------
# Risk metrics calculation
# ---------------------------------------------------------------------------

class TestCalculateRiskMetrics:
    """Tests for calculate_risk_metrics()."""

    def test_good_credit_score(self):
        """Score 85 -> PD = 0.50 - 0.85*0.49 = 0.0835 -> MEDIUM."""
        result = calculate_risk_metrics(
            credit_score=85,
            loan_amount=Decimal("250000"),
        )

        assert result.probability_of_default == pytest.approx(0.0835, abs=0.001)
        assert result.risk_rating == "MEDIUM"
        assert result.loss_given_default == 0.60  # unsecured
        assert result.exposure_at_default == Decimal("250000")
        # EL = PD * LGD * EAD
        expected_el = Decimal(str(result.probability_of_default)) * Decimal("0.60") * Decimal("250000")
        assert abs(result.expected_loss - expected_el.quantize(Decimal("0.01"))) < Decimal("1")

    def test_poor_credit_score(self):
        """Score 20 -> PD = 0.50 - 0.20*0.49 = 0.402 -> VERY_HIGH."""
        result = calculate_risk_metrics(
            credit_score=20,
            loan_amount=Decimal("250000"),
        )

        assert result.probability_of_default == pytest.approx(0.402, abs=0.001)
        assert result.risk_rating == "VERY_HIGH"

    def test_secured_loan_reduces_lgd(self):
        """Security value > 0 should reduce LGD below 0.60."""
        result = calculate_risk_metrics(
            credit_score=70,
            loan_amount=Decimal("250000"),
            security_value=Decimal("200000"),
            security_type="property",
        )

        assert result.loss_given_default < 0.60

    def test_return_type(self):
        result = calculate_risk_metrics(
            credit_score=75,
            loan_amount=Decimal("100000"),
        )

        assert isinstance(result, RiskCalculationResult)
        assert isinstance(result.probability_of_default, float)
        assert isinstance(result.loss_given_default, float)
        assert isinstance(result.exposure_at_default, Decimal)
        assert isinstance(result.expected_loss, Decimal)
        assert isinstance(result.risk_rating, str)

    def test_boundary_score_100(self):
        """Score 100 -> PD = 0.50 - 1.0*0.49 = 0.01 (minimum)."""
        result = calculate_risk_metrics(
            credit_score=100,
            loan_amount=Decimal("100000"),
        )

        assert result.probability_of_default == pytest.approx(0.01, abs=0.001)
        assert result.risk_rating == "LOW"

    def test_boundary_score_0(self):
        """Score 0 -> PD = 0.50 - 0*0.49 = 0.50 (maximum)."""
        result = calculate_risk_metrics(
            credit_score=0,
            loan_amount=Decimal("100000"),
        )

        assert result.probability_of_default == pytest.approx(0.50, abs=0.001)
        assert result.risk_rating == "VERY_HIGH"
