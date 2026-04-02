"""Stress testing tool for evaluating financial resilience under adverse scenarios.

Runs configurable stress scenarios from scoring.yaml against a borrower's
financial position, producing per-scenario impact assessments and an overall
resilience determination.
"""

from __future__ import annotations

import logging
from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, Field

from src.config.settings import ConfigLoader

logger = logging.getLogger(__name__)

# Quantize template for 2 decimal places
TWO_PLACES = Decimal("0.01")


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ScenarioResult(BaseModel):
    """Result of a single stress scenario."""

    scenario: str = Field(description="Scenario name from config")
    description: str = Field(description="Scenario description from config")
    stressed_revenue: Decimal = Field(
        max_digits=14, decimal_places=2, description="Revenue under stress"
    )
    stressed_costs: Decimal = Field(
        max_digits=14, decimal_places=2, description="Costs under stress"
    )
    stressed_profit: Decimal = Field(
        max_digits=14, decimal_places=2, description="Profit under stress"
    )
    stressed_pd: float = Field(description="Probability of default under stress")
    profit_impact_pct: float = Field(
        description="Percentage change in profit vs baseline"
    )
    survives: bool = Field(description="True if stressed profit > 0")


class StressTestResult(BaseModel):
    """Aggregated result across all stress scenarios."""

    scenarios: list[ScenarioResult] = Field(description="Per-scenario results")
    scenarios_survived: int = Field(
        description="Count of scenarios where the borrower survives"
    )
    worst_case_profit: Decimal = Field(
        description="Minimum stressed profit across all scenarios"
    )
    overall_resilient: bool = Field(
        description="True if survived >= 3 out of 5 scenarios"
    )


# ---------------------------------------------------------------------------
# Stress test function
# ---------------------------------------------------------------------------


def run_stress_tests(
    revenue: Decimal,
    total_costs: Decimal,
    net_profit: Decimal,
    credit_score: int,
) -> StressTestResult:
    """Run stress test scenarios against a borrower's financial position.

    Applies revenue shocks and cost increases from each configured scenario,
    computes stressed profit and probability of default, and determines
    overall financial resilience.

    **Critical:** Shocks are applied as ``original * (1 + shock)``, not
    ``original * shock``.

    Args:
        revenue: Baseline annual revenue.
        total_costs: Baseline total costs.
        net_profit: Baseline net profit.
        credit_score: Credit score (0--100) for PD calculation.

    Returns:
        StressTestResult with per-scenario details and resilience summary.
    """
    cfg = ConfigLoader().scoring().stress_test

    # Base probability of default from credit score
    base_pd = 0.50 - (credit_score / 100) * 0.49

    scenario_results: list[ScenarioResult] = []

    for scenario in cfg.scenarios:
        # Apply shocks: original * (1 + shock)
        stressed_revenue = (
            revenue * Decimal(str(1 + scenario.revenue_shock))
        ).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

        stressed_costs = (
            total_costs * Decimal(str(1 + scenario.cost_increase))
        ).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

        stressed_profit = (stressed_revenue - stressed_costs).quantize(
            TWO_PLACES, rounding=ROUND_HALF_UP
        )

        # Stressed probability of default
        stressed_pd = min(1.0, base_pd * scenario.default_rate_multiplier)

        # Profit impact percentage
        if net_profit != 0:
            profit_impact_pct = round(
                float((stressed_profit - net_profit) / abs(net_profit)) * 100, 2
            )
        else:
            profit_impact_pct = 0.0

        survives = stressed_profit > 0

        scenario_results.append(
            ScenarioResult(
                scenario=scenario.name,
                description=scenario.description,
                stressed_revenue=stressed_revenue,
                stressed_costs=stressed_costs,
                stressed_profit=stressed_profit,
                stressed_pd=stressed_pd,
                profit_impact_pct=profit_impact_pct,
                survives=survives,
            )
        )

        logger.debug(
            "Scenario %s: revenue=%s, costs=%s, profit=%s, pd=%.4f, survives=%s",
            scenario.name,
            stressed_revenue,
            stressed_costs,
            stressed_profit,
            stressed_pd,
            survives,
        )

    # Aggregate summary
    scenarios_survived = sum(1 for s in scenario_results if s.survives)
    worst_case_profit = min(s.stressed_profit for s in scenario_results)
    overall_resilient = scenarios_survived >= 3

    logger.info(
        "Stress test complete: %d/%d survived, worst profit=%s, resilient=%s",
        scenarios_survived,
        len(scenario_results),
        worst_case_profit,
        overall_resilient,
    )

    return StressTestResult(
        scenarios=scenario_results,
        scenarios_survived=scenarios_survived,
        worst_case_profit=worst_case_profit,
        overall_resilient=overall_resilient,
    )
