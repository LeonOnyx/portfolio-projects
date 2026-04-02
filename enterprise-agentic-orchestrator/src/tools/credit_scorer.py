"""Credit scoring engine with factor normalisation and weighted scoring.

Accepts financial metrics as plain dicts and returns a CreditScoreResult
with a 0-100 score, 8-factor breakdown, and CCJ penalty details.
Weights are loaded from scoring.yaml via ConfigLoader.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.config.settings import ConfigLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class CreditScoreResult(BaseModel):
    """Output of the credit scoring function."""

    score: int = Field(ge=0, le=100, description="Final clamped credit score")
    factor_breakdown: dict[str, float] = Field(
        description="Factor name -> weighted contribution on 0-100 scale"
    )
    ccj_penalty: int = Field(description="Total CCJ penalty applied")
    raw_score: float = Field(description="Weighted sum before CCJ penalty and clamping")


# ---------------------------------------------------------------------------
# Normalisation helpers (each returns float in [0, 1])
# ---------------------------------------------------------------------------


def _normalise_profit_margin(margin: float) -> float:
    """Higher margin is better; cap excellent at 30%."""
    return max(0.0, min(1.0, margin / 0.30))


def _normalise_debt_ratio(ratio: float) -> float:
    """Lower debt-to-asset ratio is better."""
    return max(0.0, min(1.0, 1.0 - ratio))


def _normalise_revenue_trend(financials: list[dict[str, Any]]) -> float:
    """YoY revenue growth of most recent pair, normalised to [0, 1].

    With fewer than 2 years of data, returns 0.5 (neutral).
    Growth of +50% maps to 1.0, decline of -50% maps to 0.0, linear between.
    """
    if len(financials) < 2:
        return 0.5

    sorted_fins = sorted(financials, key=lambda f: f["year"])
    previous = sorted_fins[-2]["revenue"]
    recent = sorted_fins[-1]["revenue"]

    if previous == 0:
        return 0.5

    growth = (recent - previous) / abs(previous)
    # Map [-0.50, +0.50] -> [0.0, 1.0]
    normalised = (growth + 0.50) / 1.0
    return max(0.0, min(1.0, normalised))


def _normalise_cash_coverage(cash_balance: float, total_liabilities: float) -> float:
    """Cash-to-liabilities ratio capped at 1.0."""
    if total_liabilities <= 0:
        return 1.0
    return max(0.0, min(1.0, cash_balance / total_liabilities))


def _normalise_years_trading(years: int) -> float:
    """Longer trading history is better; cap at 20 years."""
    return min(1.0, years / 20)


def _normalise_sector_outlook(outlook: str) -> float:
    """Map sector outlook string to normalised score."""
    mapping = {
        "positive": 0.9,
        "stable": 0.6,
        "cautious": 0.3,
        "negative": 0.1,
    }
    return mapping.get(outlook.lower().strip(), 0.5)


def _normalise_ccj_history(ccj_count: int) -> float:
    """0 CCJs = 1.0, each CCJ reduces by 0.2."""
    return max(0.0, 1.0 - ccj_count * 0.2)


def _normalise_security_coverage(security_value: float, loan_amount: float) -> float:
    """Security-to-loan ratio capped at 1.5, mapped to [0, 1]."""
    if loan_amount <= 0:
        return 0.0
    ratio = security_value / loan_amount
    capped = min(ratio, 1.5)
    return capped / 1.5


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------


def calculate_credit_score(
    financials: list[dict[str, Any]],
    years_trading: int,
    sector_outlook: str,
    ccj_count: int = 0,
    security_value: float = 0.0,
    loan_amount: float = 0.0,
) -> CreditScoreResult:
    """Calculate a credit score from financial metrics and contextual data.

    Args:
        financials: List of dicts with keys including year, revenue,
            profit_margin, debt_to_asset_ratio, cash_balance, total_liabilities.
        years_trading: Number of years the business has been trading.
        sector_outlook: One of positive/stable/cautious/negative.
        ccj_count: Number of County Court Judgments on record.
        security_value: Value of collateral offered.
        loan_amount: Requested loan amount.

    Returns:
        CreditScoreResult with score, factor breakdown, CCJ penalty, and raw score.
    """
    cfg = ConfigLoader().scoring().credit_scoring
    weights = cfg.weights

    # Extract most recent year's financials
    sorted_fins = sorted(financials, key=lambda f: f["year"])
    latest = sorted_fins[-1]

    # Compute normalised factor scores
    factors: dict[str, float] = {
        "profit_margin": _normalise_profit_margin(latest["profit_margin"]),
        "debt_to_asset_ratio": _normalise_debt_ratio(latest["debt_to_asset_ratio"]),
        "revenue_trend": _normalise_revenue_trend(financials),
        "cash_coverage": _normalise_cash_coverage(
            latest["cash_balance"], latest["total_liabilities"]
        ),
        "years_trading": _normalise_years_trading(years_trading),
        "sector_outlook": _normalise_sector_outlook(sector_outlook),
        "ccj_history": _normalise_ccj_history(ccj_count),
        "security_coverage": _normalise_security_coverage(security_value, loan_amount),
    }

    # Weight mapping from config
    weight_map: dict[str, float] = {
        "profit_margin": weights.profit_margin,
        "debt_to_asset_ratio": weights.debt_to_asset_ratio,
        "revenue_trend": weights.revenue_trend,
        "cash_coverage": weights.cash_coverage,
        "years_trading": weights.years_trading,
        "sector_outlook": weights.sector_outlook,
        "ccj_history": weights.ccj_history,
        "security_coverage": weights.security_coverage,
    }

    # Weighted sum -> raw score on 100 scale
    weighted_sum = sum(factors[k] * weight_map[k] for k in factors)
    raw_score = weighted_sum * 100

    # Factor breakdown on 100 scale
    factor_breakdown = {
        k: round(factors[k] * weight_map[k] * 100, 2) for k in factors
    }

    # CCJ penalty
    ccj_penalty = ccj_count * cfg.ccj_penalty_per_count

    # Final clamped score
    score = int(round(max(0, min(100, raw_score - ccj_penalty))))

    logger.info(
        "Credit score calculated: %d (raw=%.2f, ccj_penalty=%d)",
        score,
        raw_score,
        ccj_penalty,
    )

    return CreditScoreResult(
        score=score,
        factor_breakdown=factor_breakdown,
        ccj_penalty=ccj_penalty,
        raw_score=raw_score,
    )
