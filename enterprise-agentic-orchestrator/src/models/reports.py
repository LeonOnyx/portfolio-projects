"""Report and decision models for the Enterprise Agentic Orchestrator.

Defines Pydantic v2 models for analysis reports, review reports, compliance
reports, and final decisions produced by the agent pipeline.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class Recommendation(str, Enum):
    """Agent recommendation outcomes."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    REFER_TO_UNDERWRITER = "REFER_TO_UNDERWRITER"


class ConfidenceLevel(str, Enum):
    """Confidence classification for review assessments."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class DecisionOutcome(str, Enum):
    """Final decision outcomes for loan applications."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    REFERRED_TO_UNDERWRITER = "REFERRED_TO_UNDERWRITER"
    ESCALATED = "ESCALATED"


class RiskMetrics(BaseModel):
    """Quantitative risk measures for credit assessment."""

    probability_of_default: float = Field(ge=0.0, le=1.0)
    loss_given_default: float = Field(ge=0.0, le=1.0)
    exposure_at_default: Decimal = Field(max_digits=12, decimal_places=2)
    expected_loss: Decimal = Field(max_digits=12, decimal_places=2)


class AnalysisReport(BaseModel):
    """Primary credit analysis report produced by the analyst agent."""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    application_id: str
    credit_score: int = Field(ge=0, le=100)
    risk_metrics: RiskMetrics
    sector_outlook: str
    recommendation: Recommendation
    reasoning: str
    source_citations: list[dict] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ReviewReport(BaseModel):
    """Peer review report produced by the reviewer agent."""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    application_id: str
    analyst_report_id: str
    agrees_with_analyst: bool
    confidence_level: ConfidenceLevel
    quality_score: float = Field(ge=0.0, le=1.0)
    stress_test_results: list[dict] = Field(default_factory=list)
    issues_found: list[str] = Field(default_factory=list)
    reasoning: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ComplianceCheckResult(BaseModel):
    """Individual regulatory compliance check outcome."""

    check_name: str
    passed: bool
    regulation_cited: str
    details: str


class ComplianceReport(BaseModel):
    """Aggregated compliance report from the compliance agent."""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    application_id: str
    checks: list[ComplianceCheckResult]
    overall_passed: bool
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def overall_passed_requires_all_checks(self) -> ComplianceReport:
        """overall_passed must be True only if ALL individual checks passed."""
        all_passed = all(check.passed for check in self.checks)
        if self.overall_passed and not all_passed:
            raise ValueError(
                "overall_passed cannot be True when individual checks have failed"
            )
        return self


class Decision(BaseModel):
    """Final lending decision produced by the orchestrator."""

    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    application_id: str
    outcome: DecisionOutcome
    reasoning: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    conditions: list[str] = Field(default_factory=list)
    decided_at: datetime = Field(default_factory=datetime.utcnow)
