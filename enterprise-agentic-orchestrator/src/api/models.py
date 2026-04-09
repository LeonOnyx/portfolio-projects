"""Pydantic v2 request/response models for the credit-risk REST API.

Request models mirror the domain ``LoanApplication`` shape but exclude
server-generated fields (``application_id``, ``submitted_at``).  Response
models extract relevant fields from the orchestrator state dict without
exposing internal pipeline state.

The helper :func:`build_assessment_response` centralises status-code
routing so endpoint handlers stay thin.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# Reuse domain validation types directly -- no duplication of field
# constraints (min_length, patterns, decimal_places, etc.).
from src.models.loan import ApplicantDetails, FinancialSummary, LoanDetails


# =====================================================================
# Request models
# =====================================================================


class LoanApplicationRequest(BaseModel):
    """Incoming loan application payload.

    Mirrors :class:`src.models.loan.LoanApplication` but omits
    ``application_id`` and ``submitted_at`` which are server-generated.
    Nested types are imported directly from the domain layer so field
    validation rules are shared, not copied.
    """

    model_config = ConfigDict(from_attributes=True)

    applicant: ApplicantDetails
    loan: LoanDetails
    financials: list[FinancialSummary] = Field(min_length=1, max_length=5)
    credit_score: Optional[int] = Field(default=None, ge=0, le=100)
    ccj_count: int = Field(default=0, ge=0)


# =====================================================================
# Response models
# =====================================================================


class AssessmentResponse(BaseModel):
    """200 -- assessment completed with a decision."""

    model_config = ConfigDict(from_attributes=True)

    request_id: str
    application_id: str
    decision: str
    confidence_score: float
    reasoning: str
    audit_trail: list[dict] = Field(default_factory=list)
    grounding_scores: list[dict] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)


class EscalationResponse(BaseModel):
    """202 -- application escalated to human review."""

    model_config = ConfigDict(from_attributes=True)

    request_id: str
    application_id: str
    status: str = "escalated"
    message: str = "Application has been escalated to human review"
    reasoning: str
    escalation_triggers: list[str] = Field(default_factory=list)
    audit_trail: list[dict] = Field(default_factory=list)


class ValidationErrorResponse(BaseModel):
    """400 -- intake validation failure."""

    request_id: str
    error: str = "validation_error"
    detail: str
    errors: list[dict] = Field(default_factory=list)


class DecisionResponse(BaseModel):
    """GET /decisions/{id} -- same shape as AssessmentResponse."""

    model_config = ConfigDict(from_attributes=True)

    request_id: str
    application_id: str
    decision: str
    confidence_score: float
    reasoning: str
    audit_trail: list[dict] = Field(default_factory=list)
    grounding_scores: list[dict] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)


class ExplainResponse(BaseModel):
    """GET /decisions/{id}/explain -- explainability breakdown."""

    model_config = ConfigDict(from_attributes=True)

    request_id: str
    application_id: str
    decision: str
    confidence_score: float
    reasoning: str
    analysis_summary: dict | None = None
    review_summary: dict | None = None
    compliance_summary: dict | None = None
    grounding_scores: list[dict] = Field(default_factory=list)


class AuditResponse(BaseModel):
    """GET /decisions/{id}/audit -- full audit trail."""

    request_id: str
    entry_count: int
    audit_trail: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """GET /health -- service health check."""

    status: str
    services: dict = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Generic 500 error response."""

    error: str
    detail: str


# =====================================================================
# Response builder
# =====================================================================


def _extract_escalation_triggers(audit_trail: list[dict]) -> list[str]:
    """Pull escalation-related entries from the audit trail.

    Looks for audit entries whose ``action`` field contains the word
    "escalation" (case-insensitive) and returns a deduplicated list of
    those action strings.
    """
    triggers: list[str] = []
    seen: set[str] = set()
    for entry in audit_trail:
        if not isinstance(entry, dict):
            continue
        action = entry.get("action", "")
        if "escalation" in action.lower() and action not in seen:
            triggers.append(action)
            seen.add(action)
    return triggers


def build_assessment_response(result: dict) -> tuple[int, dict]:
    """Route an orchestrator result dict to the correct status code and response body.

    Returns
    -------
    tuple[int, dict]
        ``(status_code, response_dict)`` ready for the endpoint to return.

    Routing logic:

    * ``final_decision == "ERROR"`` with any error at ``stage == "intake"``
      -> 400 :class:`ValidationErrorResponse`
    * ``final_decision == "ERROR"`` (non-intake)
      -> 500 :class:`ErrorResponse`
    * ``requires_escalation`` is truthy
      -> 202 :class:`EscalationResponse`
    * Otherwise
      -> 200 :class:`AssessmentResponse`
    """
    final_decision = result.get("final_decision", "")
    errors = result.get("errors", []) or []
    audit_trail = result.get("audit_trail", []) or []
    grounding_scores = result.get("grounding_scores", []) or []
    request_id = result.get("request_id", "")
    application = result.get("application", {}) or {}
    application_id = application.get("application_id", "")
    reasoning = result.get("reasoning_trace", "") or ""

    # --- ERROR path -------------------------------------------------------
    if final_decision == "ERROR":
        # Check for intake-stage errors (validation failures -> 400)
        intake_errors = [
            e for e in errors
            if isinstance(e, dict) and e.get("stage") == "intake"
        ]
        if intake_errors:
            detail = "; ".join(
                e.get("error", "Unknown validation error")
                for e in intake_errors
            )
            resp = ValidationErrorResponse(
                request_id=request_id,
                detail=detail,
                errors=intake_errors,
            )
            return 400, resp.model_dump()

        # Non-intake errors -> 500
        detail = "; ".join(
            e.get("error", "Internal error") if isinstance(e, dict) else str(e)
            for e in errors
        ) or reasoning or "Unknown error"
        resp_500 = ErrorResponse(
            error="internal_error",
            detail=detail,
        )
        return 500, resp_500.model_dump()

    # --- ESCALATION path --------------------------------------------------
    if result.get("requires_escalation", False):
        triggers = _extract_escalation_triggers(audit_trail)
        resp_esc = EscalationResponse(
            request_id=request_id,
            application_id=application_id,
            reasoning=reasoning,
            escalation_triggers=triggers,
            audit_trail=audit_trail,
        )
        return 202, resp_esc.model_dump()

    # --- DECIDED path (happy path) ----------------------------------------
    resp_ok = AssessmentResponse(
        request_id=request_id,
        application_id=application_id,
        decision=final_decision,
        confidence_score=result.get("confidence_score", 0.0) or 0.0,
        reasoning=reasoning,
        audit_trail=audit_trail,
        grounding_scores=grounding_scores,
        errors=[e for e in errors if isinstance(e, dict)],
    )
    return 200, resp_ok.model_dump()
