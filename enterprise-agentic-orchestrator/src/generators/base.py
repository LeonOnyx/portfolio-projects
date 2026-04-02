"""Shared utilities for synthetic data generators.

Provides seeded random instances, risk profile distribution,
Decimal conversion, and reference data used across all generators.
"""

from __future__ import annotations

import random
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum

from faker import Faker

from src.models.loan import SectorType


class RiskProfile(str, Enum):
    """Risk profile categories for synthetic data generation."""

    HEALTHY = "healthy"
    STRESSED = "stressed"
    DISTRESSED = "distressed"


RISK_DISTRIBUTION: dict[RiskProfile, float] = {
    RiskProfile.HEALTHY: 0.50,
    RiskProfile.STRESSED: 0.30,
    RiskProfile.DISTRESSED: 0.20,
}


def create_seeded_faker(seed: int) -> Faker:
    """Return a Faker instance localised to en_GB with deterministic seed."""
    fake = Faker("en_GB")
    Faker.seed(seed)
    return fake


def create_seeded_random(seed: int) -> random.Random:
    """Return a seeded Random instance for deterministic generation."""
    return random.Random(seed)


def to_decimal(value: float, places: int = 2) -> Decimal:
    """Convert a float to Decimal with exact decimal places using ROUND_HALF_UP.

    Critical for Pydantic models that enforce max_digits and decimal_places.
    """
    quantize_str = "0." + "0" * places
    return Decimal(str(value)).quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def assign_risk_profiles(n: int, rng: random.Random) -> list[RiskProfile]:
    """Assign risk profiles to *n* items following RISK_DISTRIBUTION.

    Uses the provided *rng* instance (never the global random module)
    so results are deterministic for a given seed.
    """
    profiles = list(RISK_DISTRIBUTION.keys())
    weights = list(RISK_DISTRIBUTION.values())
    return rng.choices(profiles, weights=weights, k=n)


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

LOAN_PURPOSES: dict[SectorType, list[str]] = {
    SectorType.CONSTRUCTION: [
        "Purchase of excavation equipment",
        "Site development funding",
        "Construction materials bulk order",
        "Commercial building refurbishment",
    ],
    SectorType.HOSPITALITY: [
        "Restaurant fit-out and renovation",
        "Hotel expansion project",
        "Kitchen equipment upgrade",
        "Event venue modernisation",
    ],
    SectorType.RETAIL: [
        "Stock purchase for seasonal demand",
        "Store refurbishment programme",
        "E-commerce platform development",
        "Point-of-sale system upgrade",
    ],
    SectorType.TECHNOLOGY: [
        "Software product development",
        "Cloud infrastructure scaling",
        "Cybersecurity enhancement programme",
        "AI research and development",
        "Data centre expansion",
    ],
    SectorType.MANUFACTURING: [
        "Production line modernisation",
        "Raw materials procurement",
        "Quality control system upgrade",
        "Warehouse expansion project",
    ],
    SectorType.HEALTHCARE: [
        "Medical equipment acquisition",
        "Clinic premises expansion",
        "Digital health platform build",
        "Regulatory compliance upgrade",
    ],
    SectorType.LOGISTICS: [
        "Fleet vehicle acquisition",
        "Warehouse automation systems",
        "Route optimisation technology",
        "Cold chain infrastructure",
    ],
    SectorType.PROFESSIONAL_SERVICES: [
        "Office premises expansion",
        "Practice management software",
        "Staff recruitment and training",
        "International office opening",
    ],
    SectorType.AGRICULTURE: [
        "Farm machinery purchase",
        "Irrigation system installation",
        "Livestock acquisition programme",
        "Crop diversification project",
    ],
    SectorType.ENERGY: [
        "Solar panel installation project",
        "Wind turbine acquisition",
        "Battery storage infrastructure",
        "Grid connection upgrade",
        "Energy efficiency retrofit",
    ],
}

SECURITY_TYPES: list[str] = [
    "unsecured",
    "property",
    "equipment",
    "debtors",
    "inventory",
]
