"""Health check and Prometheus metrics endpoints.

- ``GET /health`` -- reports overall system health and per-service
  availability (Weaviate vector DB, Azure OpenAI LLM provider).
- ``GET /metrics`` -- returns Prometheus-compatible text exposition
  of all registered metrics (domain counters, histograms, and
  HTTP-level instrumentation when available).
"""

import logging
import os

from fastapi import APIRouter
from fastapi.responses import Response

from src.api.models import HealthResponse
from src.observability.metrics import get_metrics_text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["operations"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check system health and service availability",
)
async def health_check():
    """Probe Weaviate and LLM availability, returning aggregate status.

    Status values:

    - **healthy** -- all services reachable
    - **degraded** -- at least one service healthy, at least one not
    - **unhealthy** -- no service reachable
    """
    services: dict[str, dict] = {}
    services["weaviate"] = await _check_weaviate()
    services["llm"] = await _check_llm()

    statuses = [s.get("status") for s in services.values()]
    if all(s == "healthy" for s in statuses):
        overall = "healthy"
    elif any(s == "healthy" for s in statuses):
        overall = "degraded"
    else:
        overall = "unhealthy"

    return HealthResponse(status=overall, services=services)


async def _check_weaviate() -> dict:
    """Attempt a local Weaviate connection and readiness check."""
    try:
        import weaviate

        client = weaviate.connect_to_local()
        ready = client.is_ready()
        client.close()
        return {"status": "healthy" if ready else "unhealthy", "ready": ready}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


async def _check_llm() -> dict:
    """Check Azure OpenAI reachability via a lightweight models.list call."""
    try:
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            return {
                "status": "unconfigured",
                "error": "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY not set",
            }
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version="2024-12-01-preview",
        )
        client.models.list()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@router.get(
    "/metrics",
    summary="Prometheus-compatible metrics",
    include_in_schema=True,
)
async def metrics():
    """Serve all Prometheus metrics in text exposition format.

    Returns the output of ``prometheus_client.generate_latest()``
    with the correct ``text/plain; version=0.0.4; charset=utf-8``
    content type header.
    """
    metrics_bytes, content_type = get_metrics_text()
    return Response(content=metrics_bytes, media_type=content_type)
