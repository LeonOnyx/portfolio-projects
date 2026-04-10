"""GET endpoints for retrieving previous assessment results.

Provides three read-only endpoints:

- ``GET /decisions/{request_id}`` -- full decision with audit trail
- ``GET /decisions/{request_id}/explain`` -- explainability breakdown
  with analysis, review, and compliance summaries plus grounding scores
- ``GET /decisions/{request_id}/audit`` -- standalone audit trail export

All endpoints return 404 when the requested ``request_id`` has no
persisted assessment in :class:`AssessmentStorage`.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_storage
from src.api.models import AuditResponse, DecisionResponse, ExplainResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["decisions"])


@router.get(
    "/decisions/{request_id}",
    response_model=DecisionResponse,
    summary="Retrieve a previous assessment decision",
    responses={404: {"description": "Request ID not found"}},
)
async def get_decision(request_id: str, storage=Depends(get_storage)):
    """Look up a previously assessed decision by its ``request_id``.

    Returns the same shape as :class:`AssessmentResponse` so consumers
    can use a single model for both POST and GET flows.
    """
    try:
        result = await storage.get(request_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request_id format")
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Decision not found for request_id: {request_id}",
        )

    return DecisionResponse(
        request_id=result.get("request_id", request_id),
        application_id=result.get("application", {}).get("application_id", ""),
        decision=result.get("final_decision", "UNKNOWN"),
        confidence_score=result.get("confidence_score", 0.0) or 0.0,
        reasoning=result.get("reasoning_trace", ""),
        audit_trail=result.get("audit_trail", []),
        grounding_scores=result.get("grounding_scores", []),
        errors=result.get("errors", []),
    )


@router.get(
    "/decisions/{request_id}/explain",
    response_model=ExplainResponse,
    summary="Retrieve explainability report for an assessment",
    responses={404: {"description": "Request ID not found"}},
)
async def get_explain(request_id: str, storage=Depends(get_storage)):
    """Return an explainability breakdown for a completed assessment.

    Includes per-stage summaries (analysis, review, compliance) and
    grounding scores so consumers can understand *why* the decision
    was made and how well each claim was grounded.
    """
    try:
        result = await storage.get(request_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request_id format")
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Decision not found for request_id: {request_id}",
        )

    return ExplainResponse(
        request_id=result.get("request_id", request_id),
        application_id=result.get("application", {}).get("application_id", ""),
        decision=result.get("final_decision", "UNKNOWN"),
        confidence_score=result.get("confidence_score", 0.0) or 0.0,
        reasoning=result.get("reasoning_trace", ""),
        analysis_summary=result.get("analysis_result"),
        review_summary=result.get("review_result"),
        compliance_summary=result.get("compliance_result"),
        grounding_scores=result.get("grounding_scores", []),
    )


@router.get(
    "/decisions/{request_id}/audit",
    response_model=AuditResponse,
    summary="Retrieve full audit trail for an assessment",
    responses={404: {"description": "Request ID not found"}},
)
async def get_audit(request_id: str, storage=Depends(get_storage)):
    """Return the complete audit trail for a given assessment.

    Each entry records an action taken during the orchestration
    pipeline (intake validation, agent execution, grounding checks,
    decision matrix evaluation, etc.) with timestamps and metadata
    for regulatory traceability.
    """
    try:
        result = await storage.get(request_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid request_id format")
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Decision not found for request_id: {request_id}",
        )

    audit_entries = result.get("audit_trail", [])
    return AuditResponse(
        request_id=result.get("request_id", request_id),
        entry_count=len(audit_entries),
        audit_trail=audit_entries,
    )
