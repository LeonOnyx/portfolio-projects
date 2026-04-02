"""Portfolio concentration risk checker.

Validates single-name and sector exposure limits against configurable
thresholds loaded from scoring.yaml via ConfigLoader.  Accepts portfolio
context as explicit parameters so the function remains pure and testable.
"""

from __future__ import annotations

import logging
from decimal import Decimal

from pydantic import BaseModel, Field

from src.config.settings import ConfigLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class ConcentrationResult(BaseModel):
    """Output of the concentration risk check."""

    single_name_exposure_pct: float = Field(
        description="Exposure percentage for this borrower"
    )
    single_name_limit_pct: float = Field(
        description="Single-name limit from config (e.g. 5.0)"
    )
    single_name_breach: bool = Field(
        description="True if single-name exposure exceeds limit"
    )
    sector_exposure_pct: float = Field(
        description="Exposure percentage for this sector"
    )
    sector_limit_pct: float = Field(
        description="Sector limit from config (e.g. 25.0)"
    )
    sector_breach: bool = Field(
        description="True if sector exposure exceeds limit"
    )
    overall_pass: bool = Field(
        description="True only if BOTH limits are respected"
    )


# ---------------------------------------------------------------------------
# Checker function
# ---------------------------------------------------------------------------


def check_concentration(
    loan_amount: Decimal,
    borrower_name: str,
    sector: str,
    portfolio_total: Decimal,
    existing_exposures_by_name: dict[str, Decimal],
    existing_exposures_by_sector: dict[str, Decimal],
) -> ConcentrationResult:
    """Check whether a proposed loan breaches portfolio concentration limits.

    Computes single-name and sector exposure percentages *including* the
    proposed loan, and compares against limits from ``scoring.yaml``.

    Args:
        loan_amount: Proposed loan amount.
        borrower_name: Name of the borrower.
        sector: Business sector of the borrower.
        portfolio_total: Current total portfolio value (before this loan).
        existing_exposures_by_name: Current exposure per borrower name.
        existing_exposures_by_sector: Current exposure per sector.

    Returns:
        ConcentrationResult with exposure percentages, breach flags, and
        overall pass/fail.
    """
    cfg = ConfigLoader().scoring().concentration

    # Proposed exposures (existing + new loan)
    current_name_exposure = existing_exposures_by_name.get(borrower_name, Decimal("0"))
    proposed_name_exposure = current_name_exposure + loan_amount

    current_sector_exposure = existing_exposures_by_sector.get(sector, Decimal("0"))
    proposed_sector_exposure = current_sector_exposure + loan_amount

    # New portfolio total includes the proposed loan
    new_portfolio_total = portfolio_total + loan_amount

    # Compute percentages (as ratios first)
    if new_portfolio_total > 0:
        name_ratio = float(proposed_name_exposure / new_portfolio_total)
        sector_ratio = float(proposed_sector_exposure / new_portfolio_total)
    else:
        name_ratio = 0.0
        sector_ratio = 0.0

    # Convert to percentage values
    name_pct = round(name_ratio * 100, 2)
    sector_pct = round(sector_ratio * 100, 2)

    # Limits as percentage values
    name_limit_pct = cfg.single_name_limit * 100
    sector_limit_pct = cfg.sector_limit * 100

    # Breach checks (compare ratios against config limits which are ratios)
    single_name_breach = name_ratio > cfg.single_name_limit
    sector_breach = sector_ratio > cfg.sector_limit

    overall_pass = not single_name_breach and not sector_breach

    logger.info(
        "Concentration check for %s (%s): name=%.2f%% (limit %.1f%%), "
        "sector=%.2f%% (limit %.1f%%), pass=%s",
        borrower_name,
        sector,
        name_pct,
        name_limit_pct,
        sector_pct,
        sector_limit_pct,
        overall_pass,
    )

    return ConcentrationResult(
        single_name_exposure_pct=name_pct,
        single_name_limit_pct=name_limit_pct,
        single_name_breach=single_name_breach,
        sector_exposure_pct=sector_pct,
        sector_limit_pct=sector_limit_pct,
        sector_breach=sector_breach,
        overall_pass=overall_pass,
    )
