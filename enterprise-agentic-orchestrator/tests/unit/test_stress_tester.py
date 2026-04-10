"""Unit tests for stress testing tool.

Tests run_stress_tests with healthy and distressed financials,
shock formula application, and worst_case_profit computation.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.tools.stress_tester import StressTestResult, run_stress_tests


# ---------------------------------------------------------------------------
# Mock config helper
# ---------------------------------------------------------------------------

def _mock_stress_config():
    """Build a mock ConfigLoader with 5 stress scenarios from scoring.yaml."""
    scenarios = []
    scenario_data = [
        ("mild_recession", "GDP decline 1-2%", -0.10, 0.05, 1.5),
        ("severe_recession", "GDP decline 4-5%", -0.25, 0.15, 3.0),
        ("sector_shock", "Sector downturn", -0.35, 0.10, 2.5),
        ("rate_shock", "Rate increase 300bps", -0.05, 0.25, 1.8),
        ("combined_stress", "Recession + sector + rate", -0.40, 0.30, 4.0),
    ]

    for name, desc, rev_shock, cost_inc, default_mult in scenario_data:
        s = MagicMock()
        s.name = name
        s.description = desc
        s.revenue_shock = rev_shock
        s.cost_increase = cost_inc
        s.default_rate_multiplier = default_mult
        scenarios.append(s)

    stress_test = MagicMock()
    stress_test.scenarios = scenarios

    scoring = MagicMock()
    scoring.stress_test = stress_test

    config = MagicMock()
    config.scoring.return_value = scoring
    return config


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunStressTests:
    """Tests for run_stress_tests()."""

    @patch("src.tools.stress_tester.ConfigLoader")
    def test_healthy_financials(self, mock_cls):
        mock_cls.return_value = _mock_stress_config()

        # Very profitable company (50% margin) survives most stress scenarios
        result = run_stress_tests(
            revenue=Decimal("1000000"),
            total_costs=Decimal("500000"),
            net_profit=Decimal("500000"),
            credit_score=85,
        )

        assert isinstance(result, StressTestResult)
        assert len(result.scenarios) == 5
        # Highly profitable company should survive most scenarios
        assert result.scenarios_survived >= 3
        assert result.overall_resilient is True

    @patch("src.tools.stress_tester.ConfigLoader")
    def test_distressed_financials(self, mock_cls):
        mock_cls.return_value = _mock_stress_config()

        # Near break-even: any shock pushes into loss
        result = run_stress_tests(
            revenue=Decimal("1000000"),
            total_costs=Decimal("990000"),
            net_profit=Decimal("10000"),
            credit_score=30,
        )

        assert isinstance(result, StressTestResult)
        # Most scenarios should cause loss
        assert result.scenarios_survived < 3
        assert result.overall_resilient is False

    @patch("src.tools.stress_tester.ConfigLoader")
    def test_shock_formula(self, mock_cls):
        mock_cls.return_value = _mock_stress_config()

        result = run_stress_tests(
            revenue=Decimal("1000000"),
            total_costs=Decimal("800000"),
            net_profit=Decimal("200000"),
            credit_score=75,
        )

        # Verify mild_recession: revenue * (1 + (-0.10)) = 1M * 0.90 = 900k
        mild = result.scenarios[0]
        assert mild.scenario == "mild_recession"
        assert mild.stressed_revenue == Decimal("900000.00")
        # costs * (1 + 0.05) = 800k * 1.05 = 840k
        assert mild.stressed_costs == Decimal("840000.00")
        assert mild.stressed_profit == Decimal("60000.00")
        assert mild.survives is True

    @patch("src.tools.stress_tester.ConfigLoader")
    def test_worst_case_profit(self, mock_cls):
        mock_cls.return_value = _mock_stress_config()

        result = run_stress_tests(
            revenue=Decimal("1000000"),
            total_costs=Decimal("800000"),
            net_profit=Decimal("200000"),
            credit_score=75,
        )

        # worst_case_profit should be min across all scenarios
        scenario_profits = [s.stressed_profit for s in result.scenarios]
        assert result.worst_case_profit == min(scenario_profits)

    @patch("src.tools.stress_tester.ConfigLoader")
    def test_five_scenarios_present(self, mock_cls):
        mock_cls.return_value = _mock_stress_config()

        result = run_stress_tests(
            revenue=Decimal("1000000"),
            total_costs=Decimal("800000"),
            net_profit=Decimal("200000"),
            credit_score=75,
        )

        scenario_names = [s.scenario for s in result.scenarios]
        assert "mild_recession" in scenario_names
        assert "severe_recession" in scenario_names
        assert "sector_shock" in scenario_names
        assert "rate_shock" in scenario_names
        assert "combined_stress" in scenario_names

    @patch("src.tools.stress_tester.ConfigLoader")
    def test_stressed_pd_capped(self, mock_cls):
        """Stressed PD should never exceed 1.0."""
        mock_cls.return_value = _mock_stress_config()

        result = run_stress_tests(
            revenue=Decimal("1000000"),
            total_costs=Decimal("800000"),
            net_profit=Decimal("200000"),
            credit_score=10,  # base PD ~0.451, combined_stress mult=4.0
        )

        for scenario in result.scenarios:
            assert scenario.stressed_pd <= 1.0
