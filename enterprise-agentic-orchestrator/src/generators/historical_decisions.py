"""Historical lending decision generator for RAG ingestion.

Produces historical lending decisions that cross-reference generated loan
applications, with realistic outcome distributions adjusted by borrower
risk profile (based on credit score at decision time).
"""

from __future__ import annotations

import random as _random_module
from datetime import date, timedelta
from uuid import uuid4

from src.models.loan import SectorType

# ---------------------------------------------------------------------------
# Risk-factor pools by sector
# ---------------------------------------------------------------------------

_SECTOR_RISK_FACTORS: dict[str, list[str]] = {
    SectorType.CONSTRUCTION.value: [
        "Material cost escalation risk",
        "Planning permission delays",
        "Skilled labour shortage",
        "Fixed-price contract exposure",
        "Building Safety Act compliance costs",
        "Project overrun history",
        "Subcontractor dependency",
    ],
    SectorType.HOSPITALITY.value: [
        "Seasonal cash flow variability",
        "Consumer discretionary spend sensitivity",
        "Energy cost exposure",
        "Staff retention challenges",
        "Licensing and regulatory compliance",
        "Location-dependent footfall risk",
    ],
    SectorType.RETAIL.value: [
        "E-commerce displacement pressure",
        "High street vacancy rate exposure",
        "Thin operating margins",
        "Consumer confidence dependency",
        "Business rates burden",
        "Inventory obsolescence risk",
    ],
    SectorType.TECHNOLOGY.value: [
        "Talent acquisition cost inflation",
        "Customer concentration risk",
        "Rapid technology obsolescence",
        "Intellectual property protection",
        "Revenue recognition complexity",
        "Scaling cost uncertainty",
    ],
    SectorType.MANUFACTURING.value: [
        "Supply chain disruption exposure",
        "Energy-intensive production costs",
        "Post-Brexit trade friction",
        "Currency volatility on imports",
        "Equipment obsolescence cycle",
        "Quality control compliance",
    ],
    SectorType.HEALTHCARE.value: [
        "CQC regulatory compliance burden",
        "Workforce shortage and agency costs",
        "Government policy change risk",
        "Professional indemnity exposure",
        "Technology adoption requirement",
        "Patient volume concentration",
    ],
    SectorType.LOGISTICS.value: [
        "Fuel cost volatility",
        "Driver shortage and wage pressure",
        "Last-mile delivery margin pressure",
        "Clean air zone compliance costs",
        "Fleet maintenance capital requirements",
        "Customer volume concentration",
    ],
    SectorType.PROFESSIONAL_SERVICES.value: [
        "Key person dependency",
        "Fee rate pressure from procurement",
        "Professional indemnity cost inflation",
        "Client concentration risk",
        "Regulatory compliance complexity",
        "Talent retention competition",
    ],
    SectorType.AGRICULTURE.value: [
        "Subsidy transition income uncertainty",
        "Climate and weather event exposure",
        "Input cost inflation exceeding output prices",
        "Generational succession risk",
        "Trade policy and import competition",
        "Land value volatility",
    ],
    SectorType.ENERGY.value: [
        "Commodity price volatility",
        "Grid connection delay risk",
        "Regulatory and planning uncertainty",
        "Stranded asset exposure",
        "Capital-intensive project execution",
        "Technology maturity uncertainty",
    ],
}

# Fallback if sector not found
_DEFAULT_RISK_FACTORS = [
    "Market cyclicality",
    "Competitive pressure",
    "Regulatory change exposure",
    "Working capital strain",
    "Management capacity constraints",
    "Economic downturn sensitivity",
]

# ---------------------------------------------------------------------------
# Outcome-specific lessons learned templates
# ---------------------------------------------------------------------------

_LESSONS_TEMPLATES: dict[str, list[str]] = {
    "performing": [
        "Strong initial underwriting assessment validated by consistent repayment performance. "
        "Credit score and financial metrics at origination were reliable indicators of borrower resilience.",
        "Borrower demonstrated effective cash flow management throughout the facility term. "
        "Sector conditions remained supportive and the business executed well against its plan.",
        "Adequate security coverage and conservative loan-to-value provided comfortable downside protection. "
        "Regular covenant monitoring confirmed continued compliance with no waivers required.",
        "Diversified revenue streams and experienced management team contributed to stable performance. "
        "No material adverse events during the facility term.",
    ],
    "arrears": [
        "Early warning signals were present in quarterly management accounts but not escalated promptly. "
        "Enhanced monitoring triggers should have been activated when profit margins first declined.",
        "Sector-wide headwinds created temporary cash flow pressure that led to arrears. "
        "Forbearance measures (payment holiday, interest-only period) successfully stabilised the position.",
        "Working capital mismatch between receivables collection and facility repayment dates contributed "
        "to arrears. Restructuring to align repayment with cash conversion cycle resolved the issue.",
        "Unexpected cost escalation compressed margins and reduced debt service capacity. "
        "Arrears were cured following management action to reduce overheads and renegotiate supplier terms.",
    ],
    "default": [
        "Borrower's concentration on a single customer or contract created catastrophic revenue risk "
        "when that relationship deteriorated. Diversification requirements should be a covenant condition.",
        "Sector conditions deteriorated beyond stress-test scenarios. The combination of revenue "
        "decline and fixed cost base made debt service unsustainable despite management efforts.",
        "Management quality proved inadequate for the challenges faced. Leadership changes came too "
        "late to prevent default. Pre-lending assessment should weight management track record more heavily.",
        "Fraud indicators were missed during origination. Financial statements contained material "
        "misrepresentations that were only identified during post-default investigation.",
    ],
    "written_off": [
        "Total loss event with no recoverable assets. Security proved insufficient due to prior "
        "charges and asset deterioration. Valuation at origination was overstated relative to forced-sale reality.",
        "Protracted administration and liquidation process yielded negligible recovery. Legal costs "
        "consumed a significant proportion of recovered amounts. Earlier enforcement action may have "
        "preserved recoverable value.",
        "Business model proved fundamentally unviable in changed market conditions. The sector shift "
        "was foreseeable but underweighted in the original credit assessment. Scenario analysis should "
        "incorporate structural disruption risks.",
        "Connected exposures amplified losses. The borrower's linked entities were also in distress, "
        "creating cascading defaults across the group. Connected party exposure limits should have "
        "constrained total group lending.",
    ],
}


# ---------------------------------------------------------------------------
# Outcome distribution by risk profile
# ---------------------------------------------------------------------------

_OUTCOME_WEIGHTS: dict[str, dict[str, float]] = {
    "healthy": {
        "performing": 0.80,
        "arrears": 0.10,
        "default": 0.07,
        "written_off": 0.03,
    },
    "stressed": {
        "performing": 0.55,
        "arrears": 0.20,
        "default": 0.15,
        "written_off": 0.10,
    },
    "distressed": {
        "performing": 0.30,
        "arrears": 0.25,
        "default": 0.25,
        "written_off": 0.20,
    },
}


def _classify_risk(credit_score: int) -> str:
    """Classify risk profile from credit score at decision time."""
    if credit_score >= 65:
        return "healthy"
    elif credit_score >= 35:
        return "stressed"
    else:
        return "distressed"


def _pick_outcome(risk_profile: str, rng: _random_module.Random) -> str:
    """Select a performance outcome weighted by risk profile."""
    weights = _OUTCOME_WEIGHTS[risk_profile]
    outcomes = list(weights.keys())
    probs = list(weights.values())
    return rng.choices(outcomes, weights=probs, k=1)[0]


def _random_date(
    start: date,
    end: date,
    rng: _random_module.Random,
) -> date:
    """Return a random date between start and end (inclusive)."""
    delta_days = (end - start).days
    return start + timedelta(days=rng.randint(0, delta_days))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_historical_decisions(
    applications: list,
    n: int = 200,
    seed: int = 42,
) -> list[dict]:
    """Generate *n* historical lending decisions cross-referencing *applications*.

    Each decision links to an existing loan application via ``application_id``
    and includes the original loan terms, a performance outcome drawn from a
    risk-adjusted distribution, and retrospective lessons learned.

    Args:
        applications: List of ``LoanApplication`` Pydantic instances to
            cross-reference. Decisions cycle through them.
        n: Number of decisions to generate (default 200).
        seed: Random seed for reproducibility (default 42).

    Returns:
        List of decision dictionaries suitable for JSON serialisation
        and Weaviate RAG ingestion.
    """
    rng = _random_module.Random(seed)

    decision_start = date(2020, 1, 1)
    decision_end = date(2024, 6, 30)

    decisions: list[dict] = []

    for i in range(n):
        app = applications[i % len(applications)]

        # Extract fields from the Pydantic model
        app_id: str = app.application_id
        company_name: str = app.applicant.company_name
        sector: str = app.applicant.sector.value
        loan_amount: float = float(app.loan.amount_requested)
        credit_score: int = app.credit_score if app.credit_score is not None else 50

        # Risk classification and outcome
        risk_profile = _classify_risk(credit_score)
        outcome = _pick_outcome(risk_profile, rng)

        # Decision date
        decision_date = _random_date(decision_start, decision_end, rng)

        # Months to outcome
        months_ranges = {
            "performing": (12, 48),
            "arrears": (6, 24),
            "default": (9, 36),
            "written_off": (12, 48),
        }
        lo, hi = months_ranges[outcome]
        months_to_outcome = rng.randint(lo, hi)

        # Loss amount
        if outcome == "performing":
            loss_amount = None
        elif outcome == "arrears":
            loss_pct = rng.uniform(0.05, 0.15)
            loss_amount = round(loan_amount * loss_pct, 2)
        elif outcome == "default":
            loss_pct = rng.uniform(0.30, 0.60)
            loss_amount = round(loan_amount * loss_pct, 2)
        else:  # written_off
            loss_pct = rng.uniform(0.70, 1.00)
            loss_amount = round(loan_amount * loss_pct, 2)

        # Risk factors at decision time (2-4 from sector pool)
        factor_pool = _SECTOR_RISK_FACTORS.get(sector, _DEFAULT_RISK_FACTORS)
        num_factors = rng.randint(2, min(4, len(factor_pool)))
        risk_factors = rng.sample(factor_pool, num_factors)

        # Lessons learned
        lessons = rng.choice(_LESSONS_TEMPLATES[outcome])

        decision = {
            "decision_id": str(uuid4()),
            "document_type": "historical_decision",
            "application_id": app_id,
            "company_name": company_name,
            "sector": sector,
            "loan_amount": loan_amount,
            "credit_score_at_decision": credit_score,
            "decision_date": decision_date.isoformat(),
            "original_decision": "approved",
            "performance_outcome": outcome,
            "months_to_outcome": months_to_outcome,
            "loss_amount": loss_amount,
            "risk_factors_at_decision": risk_factors,
            "lessons_learned": lessons,
            "generated_at": decision_date.isoformat(),
        }
        decisions.append(decision)

    return decisions
