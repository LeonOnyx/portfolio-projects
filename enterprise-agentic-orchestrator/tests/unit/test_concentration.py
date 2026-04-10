"""Unit tests for portfolio concentration risk checker.

Tests check_concentration with within-limit, single-name breach,
sector breach, and both-limits-respected scenarios.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.tools.concentration_checker import ConcentrationResult, check_concentration


# ---------------------------------------------------------------------------
# Mock config helper
# ---------------------------------------------------------------------------

def _mock_concentration_config():
    """Build a mock ConfigLoader with standard concentration limits."""
    concentration = MagicMock()
    concentration.single_name_limit = 0.05
    concentration.sector_limit = 0.25

    scoring = MagicMock()
    scoring.concentration = concentration

    config = MagicMock()
    config.scoring.return_value = scoring
    return config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCheckConcentration:
    """Tests for check_concentration()."""

    @patch("src.tools.concentration_checker.ConfigLoader")
    def test_within_limits(self, mock_cls):
        """Small loan on large portfolio stays within both limits."""
        mock_cls.return_value = _mock_concentration_config()

        result = check_concentration(
            loan_amount=Decimal("100000"),
            borrower_name="Acme Ltd",
            sector="technology",
            portfolio_total=Decimal("5000000"),
            existing_exposures_by_name={},
            existing_exposures_by_sector={},
        )

        assert isinstance(result, ConcentrationResult)
        # 100k / (5M + 100k) = ~1.96% < 5%
        assert result.single_name_breach is False
        assert result.sector_breach is False
        assert result.overall_pass is True

    @patch("src.tools.concentration_checker.ConfigLoader")
    def test_single_name_breach(self, mock_cls):
        """Existing + new exposure exceeds single-name limit."""
        mock_cls.return_value = _mock_concentration_config()

        result = check_concentration(
            loan_amount=Decimal("500000"),
            borrower_name="Big Corp",
            sector="technology",
            portfolio_total=Decimal("2000000"),
            existing_exposures_by_name={"Big Corp": Decimal("200000")},
            existing_exposures_by_sector={},
        )

        # Total name: 200k + 500k = 700k; portfolio: 2M + 500k = 2.5M
        # 700k / 2.5M = 28% >> 5%
        assert result.single_name_breach is True
        assert result.overall_pass is False
        assert result.single_name_exposure_pct > 5.0

    @patch("src.tools.concentration_checker.ConfigLoader")
    def test_sector_breach(self, mock_cls):
        """Sector exposure exceeds sector limit."""
        mock_cls.return_value = _mock_concentration_config()

        result = check_concentration(
            loan_amount=Decimal("300000"),
            borrower_name="New Co",
            sector="retail",
            portfolio_total=Decimal("1000000"),
            existing_exposures_by_name={},
            existing_exposures_by_sector={"retail": Decimal("200000")},
        )

        # Sector total: 200k + 300k = 500k; portfolio: 1M + 300k = 1.3M
        # 500k / 1.3M = ~38.5% > 25%
        assert result.sector_breach is True
        assert result.overall_pass is False
        assert result.sector_exposure_pct > 25.0

    @patch("src.tools.concentration_checker.ConfigLoader")
    def test_both_limits_pass(self, mock_cls):
        """Both single-name and sector within limits."""
        mock_cls.return_value = _mock_concentration_config()

        result = check_concentration(
            loan_amount=Decimal("50000"),
            borrower_name="Small Ltd",
            sector="services",
            portfolio_total=Decimal("10000000"),
            existing_exposures_by_name={},
            existing_exposures_by_sector={"services": Decimal("1000000")},
        )

        # Name: 50k / 10.05M = ~0.5% < 5%
        # Sector: 1.05M / 10.05M = ~10.4% < 25%
        assert result.single_name_breach is False
        assert result.sector_breach is False
        assert result.overall_pass is True

    @patch("src.tools.concentration_checker.ConfigLoader")
    def test_overall_pass_requires_both(self, mock_cls):
        """overall_pass is False if either limit is breached."""
        mock_cls.return_value = _mock_concentration_config()

        # Just breach single-name but not sector
        result = check_concentration(
            loan_amount=Decimal("500000"),
            borrower_name="Big Corp",
            sector="diversified",
            portfolio_total=Decimal("2000000"),
            existing_exposures_by_name={"Big Corp": Decimal("200000")},
            existing_exposures_by_sector={},
        )

        assert result.single_name_breach is True
        assert result.sector_breach is False
        assert result.overall_pass is False  # single-name breach sufficient
