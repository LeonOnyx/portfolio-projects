"""Loan domain models for the Enterprise Agentic Orchestrator.

Defines Pydantic v2 models for loan applications, applicant details,
financial summaries, and loan terms used across the credit risk pipeline.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, model_validator


class SectorType(str, Enum):
    """UK commercial lending sector classifications."""

    CONSTRUCTION = "construction"
    HOSPITALITY = "hospitality"
    RETAIL = "retail"
    TECHNOLOGY = "technology"
    MANUFACTURING = "manufacturing"
    HEALTHCARE = "healthcare"
    LOGISTICS = "logistics"
    PROFESSIONAL_SERVICES = "professional_services"
    AGRICULTURE = "agriculture"
    ENERGY = "energy"


class ApplicantDetails(BaseModel):
    """Company and contact information for a loan applicant."""

    company_name: str = Field(min_length=2, max_length=200)
    company_number: str = Field(pattern=r"^\d{8}$")  # UK Companies House format
    sector: SectorType
    years_trading: int = Field(ge=0, le=200)
    employee_count: int = Field(ge=1)
    contact_name: str = Field(min_length=2)
    contact_role: str


class FinancialSummary(BaseModel):
    """Annual financial summary for creditworthiness assessment."""

    year: int = Field(ge=2000, le=2030)
    revenue: Decimal = Field(max_digits=14, decimal_places=2, ge=0)
    gross_profit: Decimal = Field(max_digits=14, decimal_places=2)
    net_profit: Decimal = Field(max_digits=14, decimal_places=2)
    total_assets: Decimal = Field(max_digits=14, decimal_places=2, ge=0)
    total_liabilities: Decimal = Field(max_digits=14, decimal_places=2, ge=0)
    cash_balance: Decimal = Field(max_digits=14, decimal_places=2)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def profit_margin(self) -> float:
        """Net profit margin as a ratio. Returns 0.0 when revenue is zero."""
        if self.revenue == 0:
            return 0.0
        return float(self.net_profit / self.revenue)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def debt_to_asset_ratio(self) -> float:
        """Debt-to-asset ratio. Returns 0.0 when total assets is zero."""
        if self.total_assets == 0:
            return 0.0
        return float(self.total_liabilities / self.total_assets)


class LoanDetails(BaseModel):
    """Loan terms and security information."""

    amount_requested: Decimal = Field(max_digits=12, decimal_places=2, gt=0)
    term_months: int = Field(ge=6, le=360)
    purpose: str = Field(min_length=5, max_length=500)
    security_type: str = Field(default="unsecured")
    security_value: Optional[Decimal] = Field(
        default=None, max_digits=12, decimal_places=2, ge=0
    )
    currency: str = Field(default="GBP", pattern=r"^[A-Z]{3}$")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def loan_to_value(self) -> Optional[float]:
        """Loan-to-value ratio. None when unsecured or no security value."""
        if self.security_value is None or self.security_value == 0:
            return None
        return float(self.amount_requested / self.security_value)

    @model_validator(mode="after")
    def secured_loans_must_have_security_value(self) -> LoanDetails:
        """Secured loans must specify a security value."""
        if self.security_type != "unsecured" and self.security_value is None:
            raise ValueError(
                "Secured loans must have a security_value specified"
            )
        return self


class LoanApplication(BaseModel):
    """Complete loan application combining applicant, financial, and loan data."""

    application_id: str = Field(default_factory=lambda: str(uuid4()))
    applicant: ApplicantDetails
    loan: LoanDetails
    financials: list[FinancialSummary] = Field(min_length=1, max_length=5)
    credit_score: Optional[int] = Field(default=None, ge=0, le=100)
    ccj_count: int = Field(default=0, ge=0)
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def no_duplicate_financial_years(self) -> LoanApplication:
        """Ensure no two financial summaries share the same year."""
        years = [f.year for f in self.financials]
        if len(years) != len(set(years)):
            raise ValueError("Duplicate financial years are not allowed")
        return self
