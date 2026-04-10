"""Unit tests for credit scoring engine (TEST-08).

Tests all 8 normalisation helpers directly (pure functions, no mocking)
and the integrated calculate_credit_score with strong/weak applicants,
CCJ penalty, and return type validation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.tools.credit_scorer import (
    CreditScoreResult,
    _normalise_cash_coverage,
    _normalise_ccj_history,
    _normalise_debt_ratio,
    _normalise_profit_margin,
    _normalise_revenue_trend,
    _normalise_sector_outlook,
    _normalise_security_coverage,
    _normalise_years_trading,
    calculate_credit_score,
)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

class TestNormaliseProfitMargin:
    def test_at_cap(self):
        assert _normalise_profit_margin(0.30) == 1.0

    def test_above_cap(self):
        assert _normalise_profit_margin(0.50) == 1.0

    def test_zero(self):
        assert _normalise_profit_margin(0.0) == 0.0

    def test_midpoint(self):
        assert _normalise_profit_margin(0.15) == pytest.approx(0.5)


class TestNormaliseDebtRatio:
    def test_no_debt(self):
        assert _normalise_debt_ratio(0.0) == 1.0

    def test_full_debt(self):
        assert _normalise_debt_ratio(1.0) == 0.0

    def test_half_debt(self):
        assert _normalise_debt_ratio(0.5) == 0.5


class TestNormaliseRevenueTrend:
    def test_single_year(self):
        assert _normalise_revenue_trend([{"year": 2024, "revenue": 100}]) == 0.5

    def test_50pct_growth(self):
        fins = [
            {"year": 2023, "revenue": 100},
            {"year": 2024, "revenue": 150},
        ]
        assert _normalise_revenue_trend(fins) == 1.0

    def test_50pct_decline(self):
        fins = [
            {"year": 2023, "revenue": 100},
            {"year": 2024, "revenue": 50},
        ]
        assert _normalise_revenue_trend(fins) == 0.0

    def test_flat(self):
        fins = [
            {"year": 2023, "revenue": 100},
            {"year": 2024, "revenue": 100},
        ]
        # 0% growth -> (0 + 0.5)/1.0 = 0.5
        assert _normalise_revenue_trend(fins) == 0.5


class TestNormaliseCashCoverage:
    def test_good_coverage(self):
        assert _normalise_cash_coverage(150000, 800000) == pytest.approx(0.1875)

    def test_full_coverage(self):
        assert _normalise_cash_coverage(800000, 800000) == 1.0

    def test_over_coverage_capped(self):
        assert _normalise_cash_coverage(1600000, 800000) == 1.0

    def test_zero_liabilities(self):
        assert _normalise_cash_coverage(100000, 0) == 1.0


class TestNormaliseYearsTrading:
    def test_ten_years(self):
        assert _normalise_years_trading(10) == 0.5

    def test_twenty_years_cap(self):
        assert _normalise_years_trading(20) == 1.0

    def test_zero_years(self):
        assert _normalise_years_trading(0) == 0.0

    def test_above_cap(self):
        assert _normalise_years_trading(30) == 1.0


class TestNormaliseSectorOutlook:
    def test_positive(self):
        assert _normalise_sector_outlook("positive") == 0.9

    def test_stable(self):
        assert _normalise_sector_outlook("stable") == 0.6

    def test_cautious(self):
        assert _normalise_sector_outlook("cautious") == 0.3

    def test_negative(self):
        assert _normalise_sector_outlook("negative") == 0.1

    def test_unknown(self):
        assert _normalise_sector_outlook("unknown") == 0.5


class TestNormaliseCCJHistory:
    def test_zero_ccjs(self):
        assert _normalise_ccj_history(0) == 1.0

    def test_five_ccjs(self):
        # 5 * 0.2 = 1.0 deduction -> clamped to 0.0
        assert _normalise_ccj_history(5) == 0.0

    def test_six_ccjs_clamped(self):
        assert _normalise_ccj_history(6) == 0.0

    def test_two_ccjs(self):
        assert _normalise_ccj_history(2) == pytest.approx(0.6)


class TestNormaliseSecurityCoverage:
    def test_full_coverage(self):
        # 250k/250k = 1.0, capped at 1.5, mapped: 1.0/1.5
        assert _normalise_security_coverage(250000, 250000) == pytest.approx(1.0 / 1.5)

    def test_no_security(self):
        assert _normalise_security_coverage(0, 250000) == 0.0

    def test_over_coverage_capped(self):
        # 500k/250k = 2.0, capped to 1.5, mapped: 1.5/1.5 = 1.0
        assert _normalise_security_coverage(500000, 250000) == 1.0

    def test_zero_loan_amount(self):
        assert _normalise_security_coverage(100000, 0) == 0.0


# ---------------------------------------------------------------------------
# Integrated credit scoring
# ---------------------------------------------------------------------------

def _mock_scoring_config():
    """Build a mock ConfigLoader returning standard scoring config."""
    weights = MagicMock()
    weights.profit_margin = 0.20
    weights.debt_to_asset_ratio = 0.15
    weights.revenue_trend = 0.15
    weights.cash_coverage = 0.10
    weights.years_trading = 0.10
    weights.sector_outlook = 0.10
    weights.ccj_history = 0.10
    weights.security_coverage = 0.10

    credit_scoring = MagicMock()
    credit_scoring.weights = weights
    credit_scoring.ccj_penalty_per_count = 5

    scoring = MagicMock()
    scoring.credit_scoring = credit_scoring

    config = MagicMock()
    config.scoring.return_value = scoring
    return config


class TestCalculateCreditScore:
    """Tests for the integrated calculate_credit_score function."""

    @patch("src.tools.credit_scorer.ConfigLoader")
    def test_strong_applicant(self, mock_cls):
        mock_cls.return_value = _mock_scoring_config()

        financials = [
            {"year": 2023, "revenue": 800000, "profit_margin": 0.20,
             "debt_to_asset_ratio": 0.30, "cash_balance": 150000,
             "total_liabilities": 500000},
            {"year": 2024, "revenue": 1000000, "profit_margin": 0.25,
             "debt_to_asset_ratio": 0.25, "cash_balance": 200000,
             "total_liabilities": 500000},
        ]

        result = calculate_credit_score(
            financials=financials,
            years_trading=15,
            sector_outlook="positive",
            ccj_count=0,
            security_value=250000,
            loan_amount=250000,
        )

        assert isinstance(result, CreditScoreResult)
        assert result.score > 70
        assert 0 <= result.score <= 100
        assert len(result.factor_breakdown) == 8
        assert result.ccj_penalty == 0

    @patch("src.tools.credit_scorer.ConfigLoader")
    def test_weak_applicant(self, mock_cls):
        mock_cls.return_value = _mock_scoring_config()

        financials = [
            {"year": 2023, "revenue": 500000, "profit_margin": 0.10,
             "debt_to_asset_ratio": 0.70, "cash_balance": 20000,
             "total_liabilities": 600000},
            {"year": 2024, "revenue": 400000, "profit_margin": 0.05,
             "debt_to_asset_ratio": 0.80, "cash_balance": 10000,
             "total_liabilities": 700000},
        ]

        result = calculate_credit_score(
            financials=financials,
            years_trading=2,
            sector_outlook="negative",
            ccj_count=3,
        )

        assert result.score < 40
        assert result.ccj_penalty == 15

    @patch("src.tools.credit_scorer.ConfigLoader")
    def test_ccj_penalty_reduces_score(self, mock_cls):
        mock_cls.return_value = _mock_scoring_config()

        financials = [
            {"year": 2024, "revenue": 1000000, "profit_margin": 0.25,
             "debt_to_asset_ratio": 0.25, "cash_balance": 200000,
             "total_liabilities": 500000},
        ]

        # Same applicant, ccj_count=0
        result_clean = calculate_credit_score(
            financials=financials,
            years_trading=15,
            sector_outlook="positive",
            ccj_count=0,
        )

        # Same applicant, ccj_count=3
        result_ccj = calculate_credit_score(
            financials=financials,
            years_trading=15,
            sector_outlook="positive",
            ccj_count=3,
        )

        # Score drops by at least 15 (3*5 CCJ penalty) plus the ccj_history
        # normalisation factor change (1.0 -> 0.4 at 0.10 weight = 6 extra points)
        assert result_clean.score > result_ccj.score
        assert result_ccj.ccj_penalty == 15
        # The direct penalty is 15 but factor contribution adds more
        assert result_clean.score - result_ccj.score >= 15

    @patch("src.tools.credit_scorer.ConfigLoader")
    def test_extreme_ccj_clamps_to_zero(self, mock_cls):
        mock_cls.return_value = _mock_scoring_config()

        financials = [
            {"year": 2024, "revenue": 1000000, "profit_margin": 0.25,
             "debt_to_asset_ratio": 0.25, "cash_balance": 200000,
             "total_liabilities": 500000},
        ]

        result = calculate_credit_score(
            financials=financials,
            years_trading=15,
            sector_outlook="positive",
            ccj_count=20,  # 20 * 5 = 100 penalty
        )

        assert result.score == 0
        assert result.ccj_penalty == 100

    @patch("src.tools.credit_scorer.ConfigLoader")
    def test_return_type_fields(self, mock_cls):
        mock_cls.return_value = _mock_scoring_config()

        financials = [
            {"year": 2024, "revenue": 1000000, "profit_margin": 0.20,
             "debt_to_asset_ratio": 0.40, "cash_balance": 150000,
             "total_liabilities": 800000},
        ]

        result = calculate_credit_score(
            financials=financials,
            years_trading=10,
            sector_outlook="stable",
        )

        assert isinstance(result.score, int)
        assert isinstance(result.factor_breakdown, dict)
        assert isinstance(result.ccj_penalty, int)
        assert isinstance(result.raw_score, float)
        assert 0 <= result.score <= 100
