"""Loan application generator with risk-correlated financial data.

Produces realistic UK SME loan applications with correlated financials,
credit scores, and CCJ history across three risk profiles (healthy,
stressed, distressed).
"""

from __future__ import annotations

import random as _random_module

from src.generators.base import (
    LOAN_PURPOSES,
    SECURITY_TYPES,
    RiskProfile,
    assign_risk_profiles,
    create_seeded_faker,
    create_seeded_random,
    to_decimal,
)
from src.models.loan import (
    ApplicantDetails,
    FinancialSummary,
    LoanApplication,
    LoanDetails,
    SectorType,
)

_CONTACT_ROLES = [
    "Director",
    "Finance Director",
    "CEO",
    "Managing Director",
    "CFO",
]

_ALL_SECTORS = list(SectorType)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _generate_applicant(
    index: int,
    sector: SectorType,
    profile: RiskProfile,
    fake,  # noqa: ANN001 -- Faker instance
    rng: _random_module.Random,
) -> ApplicantDetails:
    """Build an ApplicantDetails instance correlated to *profile*."""
    if profile == RiskProfile.HEALTHY:
        years_trading = rng.randint(5, 30)
        employee_count = rng.randint(10, 500)
    elif profile == RiskProfile.STRESSED:
        years_trading = rng.randint(2, 10)
        employee_count = rng.randint(5, 100)
    else:  # DISTRESSED
        years_trading = rng.randint(1, 5)
        employee_count = rng.randint(1, 30)

    return ApplicantDetails(
        company_name=fake.company(),
        company_number=str(rng.randint(10_000_000, 99_999_999)),
        sector=sector,
        years_trading=years_trading,
        employee_count=employee_count,
        contact_name=fake.name(),
        contact_role=rng.choice(_CONTACT_ROLES),
    )


def _generate_financials(
    profile: RiskProfile,
    rng: _random_module.Random,
) -> list[FinancialSummary]:
    """Generate 3 years (2022-2024) of financial summaries correlated to *profile*."""
    years = [2022, 2023, 2024]
    summaries: list[FinancialSummary] = []

    # Base revenue for first year
    if profile == RiskProfile.HEALTHY:
        base_revenue = rng.uniform(500_000, 10_000_000)
    elif profile == RiskProfile.STRESSED:
        base_revenue = rng.uniform(100_000, 2_000_000)
    else:
        base_revenue = rng.uniform(50_000, 500_000)

    revenue = base_revenue

    for year in years:
        # Year-over-year growth
        if year > 2022:
            if profile == RiskProfile.HEALTHY:
                growth = rng.uniform(0.05, 0.15)
            elif profile == RiskProfile.STRESSED:
                growth = rng.uniform(-0.10, 0.00)
            else:
                growth = rng.uniform(-0.30, -0.10)
            revenue = revenue * (1 + growth)

        # Ensure revenue is non-negative (ge=0 constraint)
        revenue = max(revenue, 0.0)

        # Gross profit margin
        if profile == RiskProfile.HEALTHY:
            gp_margin = rng.uniform(0.30, 0.45)
        elif profile == RiskProfile.STRESSED:
            gp_margin = rng.uniform(0.20, 0.35)
        else:
            gp_margin = rng.uniform(0.20, 0.30)
        gross_profit = revenue * gp_margin

        # Net profit margin
        if profile == RiskProfile.HEALTHY:
            np_margin = rng.uniform(0.05, 0.20)
        elif profile == RiskProfile.STRESSED:
            np_margin = rng.uniform(-0.05, 0.05)
        else:
            np_margin = rng.uniform(-0.20, -0.02)
        net_profit = revenue * np_margin

        # Total assets
        asset_multiplier = rng.uniform(0.5, 3.0)
        total_assets = revenue * asset_multiplier
        total_assets = max(total_assets, 0.0)

        # Total liabilities
        if profile == RiskProfile.HEALTHY:
            liability_ratio = rng.uniform(0.20, 0.50)
        elif profile == RiskProfile.STRESSED:
            liability_ratio = rng.uniform(0.50, 0.80)
        else:
            liability_ratio = rng.uniform(0.70, 1.20)
        total_liabilities = total_assets * liability_ratio
        total_liabilities = max(total_liabilities, 0.0)

        # Cash balance
        if profile == RiskProfile.HEALTHY:
            cash_ratio = rng.uniform(0.05, 0.20)
        elif profile == RiskProfile.STRESSED:
            cash_ratio = rng.uniform(0.01, 0.05)
        else:
            cash_ratio = rng.uniform(0.00, 0.02)
        cash_balance = revenue * cash_ratio

        summaries.append(
            FinancialSummary(
                year=year,
                revenue=to_decimal(revenue),
                gross_profit=to_decimal(gross_profit),
                net_profit=to_decimal(net_profit),
                total_assets=to_decimal(total_assets),
                total_liabilities=to_decimal(total_liabilities),
                cash_balance=to_decimal(cash_balance),
            )
        )

    return summaries


def _generate_loan(
    profile: RiskProfile,
    sector: SectorType,
    latest_revenue: float,
    rng: _random_module.Random,
) -> LoanDetails:
    """Build a LoanDetails instance correlated to *profile*."""
    # Amount: 10-30% of latest revenue
    amount_ratio = rng.uniform(0.10, 0.30)
    amount = latest_revenue * amount_ratio
    # Ensure amount is positive (gt=0 constraint)
    amount = max(amount, 100.0)

    # Term
    if profile == RiskProfile.HEALTHY:
        term = rng.randint(12, 120)
    elif profile == RiskProfile.STRESSED:
        term = rng.randint(6, 60)
    else:
        term = rng.randint(6, 36)

    # Purpose
    purposes = LOAN_PURPOSES[sector]
    purpose = rng.choice(purposes)

    # Security
    if profile == RiskProfile.HEALTHY:
        secured_prob = 0.60
    elif profile == RiskProfile.STRESSED:
        secured_prob = 0.40
    else:
        secured_prob = 0.20

    if rng.random() < secured_prob:
        security_type = rng.choice(
            [s for s in SECURITY_TYPES if s != "unsecured"]
        )
        security_value = to_decimal(amount * rng.uniform(1.0, 1.5))
    else:
        security_type = "unsecured"
        security_value = None

    return LoanDetails(
        amount_requested=to_decimal(amount),
        term_months=term,
        purpose=purpose,
        security_type=security_type,
        security_value=security_value,
        currency="GBP",
    )


def _generate_credit_and_ccj(
    profile: RiskProfile,
    rng: _random_module.Random,
) -> tuple[int, int]:
    """Return (credit_score, ccj_count) correlated to *profile*."""
    if profile == RiskProfile.HEALTHY:
        return rng.randint(65, 95), 0
    elif profile == RiskProfile.STRESSED:
        return rng.randint(35, 64), rng.randint(0, 2)
    else:
        return rng.randint(10, 34), rng.randint(1, 5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_loan_applications(
    n: int = 50,
    seed: int = 42,
) -> list[LoanApplication]:
    """Generate *n* realistic UK SME loan applications.

    All random operations use seeded instances so output is deterministic
    (except ``application_id`` which uses ``uuid4``).

    Args:
        n: Number of applications to generate (default 50).
        seed: Random seed for reproducibility (default 42).

    Returns:
        List of validated ``LoanApplication`` Pydantic model instances.
    """
    fake = create_seeded_faker(seed)
    rng = create_seeded_random(seed)

    profiles = assign_risk_profiles(n, rng)

    # Sector assignment: first 10 get one of each (shuffled), rest random
    sectors: list[SectorType] = list(_ALL_SECTORS)
    rng.shuffle(sectors)
    for _ in range(n - len(_ALL_SECTORS)):
        sectors.append(rng.choice(_ALL_SECTORS))

    applications: list[LoanApplication] = []

    for i in range(n):
        profile = profiles[i]
        sector = sectors[i]

        applicant = _generate_applicant(i, sector, profile, fake, rng)
        financials = _generate_financials(profile, rng)
        latest_revenue = float(financials[-1].revenue)
        loan = _generate_loan(profile, sector, latest_revenue, rng)
        credit_score, ccj_count = _generate_credit_and_ccj(profile, rng)

        app = LoanApplication(
            applicant=applicant,
            loan=loan,
            financials=financials,
            credit_score=credit_score,
            ccj_count=ccj_count,
        )
        applications.append(app)

    return applications
