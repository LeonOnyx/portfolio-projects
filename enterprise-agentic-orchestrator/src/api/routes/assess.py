"""POST /api/v1/assess -- Submit a loan application for credit risk assessment.

Returns:
    200: Decision rendered (AssessmentResponse)
    202: Escalated to human review (EscalationResponse)
    400: Invalid application (ValidationErrorResponse)
    500: Internal error (ErrorResponse)
"""

import logging
from uuid import uuid4

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from src.api.dependencies import get_langfuse_handler, get_orchestrator, get_storage
from src.api.models import (
    AssessmentResponse,
    EscalationResponse,
    ErrorResponse,
    LoanApplicationRequest,
    ValidationErrorResponse,
    build_assessment_response,
)
from src.observability.metrics import record_assessment_metrics, track_assessment_latency
from src.observability.tracing import traced_orchestrator_run

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["assessment"])


@router.post(
    "/assess",
    response_model=AssessmentResponse,
    responses={
        200: {"model": AssessmentResponse, "description": "Decision rendered"},
        202: {"model": EscalationResponse, "description": "Escalated to human review"},
        400: {
            "model": ValidationErrorResponse,
            "description": "Invalid application",
        },
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    summary="Submit a loan application for credit risk assessment",
)
async def assess_application(
    application: LoanApplicationRequest,
    orchestrator=Depends(get_orchestrator),
    storage=Depends(get_storage),
    langfuse_handler=Depends(get_langfuse_handler),
):
    """Accept a loan application JSON body, run the credit risk orchestrator,
    persist the result, and return the decision with appropriate status code.

    The orchestrator call is wrapped in Langfuse tracing (when available) and
    Prometheus latency tracking.  The result is persisted to JSON file storage
    before the response is returned so that downstream consumers
    (GET /decisions/{id}) can retrieve it immediately.
    """
    request_id = str(uuid4())
    app_dict = application.model_dump(mode="json")

    logger.info("Received assessment request %s", request_id)

    try:
        with track_assessment_latency():
            result = await traced_orchestrator_run(
                orchestrator=orchestrator,
                application=app_dict,
                request_id=request_id,
                user_role="api",
                langfuse_handler=langfuse_handler,
            )

        # Record domain metrics (decision counter, grounding histogram, etc.)
        record_assessment_metrics(result)

        # Persist before responding so GET /decisions/{id} is immediately available
        await storage.save(request_id, result)

        status_code, response_data = build_assessment_response(result)
        return JSONResponse(status_code=status_code, content=response_data)

    except Exception as exc:
        logger.exception("Assessment failed for request %s: %s", request_id, exc)
        return JSONResponse(
            status_code=500,
            content={
                "request_id": request_id,
                "error": "internal_error",
                "detail": f"Assessment pipeline failed: {type(exc).__name__}",
            },
        )
