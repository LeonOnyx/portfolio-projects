"""Risk calculator producing PD, LGD, EAD, and Expected Loss metrics.

All monetary calculations use Decimal arithmetic to maintain precision
required for regulated financial reporting.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class RiskCalculationResult(BaseModel):
    """Output of the risk metrics calculation."""

    probability_of_default: float = Field(
        ge=0.0, le=1.0, description="Probability of default"
    )
    loss_given_default: float = Field(
        ge=0.0, le=1.0, description="Loss given default"
    )
    exposure_at_default: Decimal = Field(
        max_digits=12, decimal_places=2, description="Exposure at default"
    )
    expected_loss: Decimal = Field(
        max_digits=12, decimal_places=2, description="Expected loss"
    )
    risk_rating: str = Field(description="Risk category: LOW, MEDIUM, HIGH, VERY_HIGH")


# ---------------------------------------------------------------------------
# Risk rating derivation
# ---------------------------------------------------------------------------

_RISK_THRESHOLDS: list[tuple[float, str]] = [
    (0.05, "LOW"),
    (0.15, "MEDIUM"),
    (0.30, "HIGH"),
]


def _derive_risk_rating(pd: float) -> str:
    """Derive risk rating from probability of default."""
    for threshold, rating in _RISK_THRESHOLDS:
        if pd < threshold:
            return rating
    return "VERY_HIGH"


# ---------------------------------------------------------------------------
# Main calculation function
# ---------------------------------------------------------------------------


def calculate_risk_metrics(
    credit_score: int,
    loan_amount: Decimal,
    security_value: Decimal | None = None,
    security_type: str = "unsecured",
) -> RiskCalculationResult:
    """Calculate risk metrics from credit score and loan details.

    Args:
        credit_score: Credit score 0-100 from the credit scorer.
        loan_amount: Requested loan amount as Decimal.
        security_value: Value of collateral as Decimal, or None if unsecured.
        security_type: Type of security (e.g. 'property', 'unsecured').

    Returns:
        RiskCalculationResult with PD, LGD, EAD, Expected Loss, and risk rating.
    """
    # PD: score 100 -> 0.01, score 0 -> 0.50
    pd = 0.50 - (credit_score / 100) * 0.49
    pd = max(0.01, min(0.50, pd))

    # LGD: secured loans get lower LGD
    if security_value is not None and security_value > 0:
        coverage = min(float(security_value / loan_amount), 1.5)
        lgd = max(0.10, 0.60 - (coverage * 0.40))
    else:
        lgd = 0.60
    lgd = max(0.10, min(0.60, lgd))

    # EAD: full exposure
    ead = loan_amount

    # Expected Loss: Decimal arithmetic throughout
    expected_loss = Decimal(str(pd)) * Decimal(str(lgd)) * ead
    expected_loss = expected_loss.quantize(Decimal("0.01"))

    # Risk rating
    risk_rating = _derive_risk_rating(pd)

    logger.info(
        "Risk metrics: PD=%.4f, LGD=%.4f, EAD=%s, EL=%s, Rating=%s",
        pd,
        lgd,
        ead,
        expected_loss,
        risk_rating,
    )

    return RiskCalculationResult(
        probability_of_default=pd,
        loss_given_default=lgd,
        exposure_at_default=ead,
        expected_loss=expected_loss,
        risk_rating=risk_rating,
    )
