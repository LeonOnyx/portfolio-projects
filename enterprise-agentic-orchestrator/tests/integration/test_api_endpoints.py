"""API endpoint integration tests using httpx.AsyncClient with ASGITransport.

Tests all FastAPI routes with mocked orchestrator, storage, and Langfuse
dependencies via FastAPI's dependency_overrides mechanism.

Coverage
--------
- POST /api/v1/assess  (200 decided, 202 escalated, 422 validation error)
- GET /api/v1/decisions/{id}  (200 found, 404 not found)
- GET /api/v1/decisions/{id}/explain  (200 found)
- GET /api/v1/decisions/{id}/audit  (200 found)
- GET /api/v1/health  (200 with service status)
- GET /api/v1/metrics  (200 Prometheus text)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Valid request body matching LoanApplicationRequest schema
# ---------------------------------------------------------------------------

_VALID_REQUEST_BODY = {
    "applicant": {
        "company_name": "API Test Corp",
        "company_number": "11223344",
        "sector": "technology",
        "years_trading": 5,
        "employee_count": 30,
        "contact_name": "John Smith",
        "contact_role": "Director",
    },
    "loan": {
        "amount_requested": 150000.00,
        "term_months": 24,
        "purpose": "Equipment purchase for expansion",
        "security_type": "unsecured",
        "currency": "GBP",
    },
    "financials": [
        {
            "year": 2024,
            "revenue": 800000.00,
            "gross_profit": 320000.00,
            "net_profit": 160000.00,
            "total_assets": 1500000.00,
            "total_liabilities": 600000.00,
            "cash_balance": 100000.00,
        },
    ],
    "credit_score": 70,
    "ccj_count": 0,
}

# Simulated orchestrator result dicts for different scenarios

_DECIDED_RESULT = {
    "request_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    "application": {"application_id": "APP-API-001"},
    "final_decision": "APPROVED",
    "confidence_score": 0.85,
    "reasoning_trace": "Analyst: APPROVE. Reviewer agrees. Compliance passed.",
    "audit_trail": [
        {"stage": "intake", "action": "input_received", "details": {}},
        {"stage": "decision", "action": "decision_rendered", "details": {}},
    ],
    "grounding_scores": [
        {"checkpoint": "post_analyst", "score": 0.95, "is_grounded": True},
    ],
    "errors": [],
    "requires_escalation": False,
    "analysis_result": {"credit_score": 72, "recommendation": "APPROVE"},
    "review_result": {"quality_score": 0.85, "agrees_with_analyst": True},
    "compliance_result": {"overall_passed": True},
}

_ESCALATED_RESULT = {
    "request_id": "aaaaaaaa-bbbb-cccc-dddd-ffffffffffff",
    "application": {"application_id": "APP-API-002"},
    "final_decision": "ESCALATED",
    "confidence_score": 0.40,
    "reasoning_trace": "Escalated due to compliance failure.",
    "audit_trail": [
        {"stage": "decision", "action": "escalation_triggered", "details": {}},
    ],
    "grounding_scores": [],
    "errors": [],
    "requires_escalation": True,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
async def api_client():
    """Yield an httpx.AsyncClient wired to the FastAPI app with mocked deps.

    Patches:
    - init_dependencies -> no-op (prevents real orchestrator/storage creation)
    - setup_instrumentator -> no-op (avoids Prometheus conflicts)
    - flush_langfuse -> no-op
    - Dependency overrides for get_orchestrator, get_storage, get_langfuse_handler
    """
    mock_orchestrator = MagicMock()
    mock_orchestrator.run = AsyncMock(return_value=_DECIDED_RESULT)

    mock_storage = MagicMock()
    mock_storage.save = AsyncMock()
    mock_storage.get = AsyncMock(return_value=_DECIDED_RESULT)

    with (
        patch("src.api.dependencies.init_dependencies"),
        patch("src.observability.metrics.setup_instrumentator"),
        patch("src.observability.tracing.flush_langfuse"),
    ):
        from src.api.app import app
        from src.api.dependencies import (
            get_langfuse_handler,
            get_orchestrator,
            get_storage,
        )

        app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
        app.dependency_overrides[get_storage] = lambda: mock_storage
        app.dependency_overrides[get_langfuse_handler] = lambda: None

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            yield client, mock_orchestrator, mock_storage

        app.dependency_overrides.clear()


# ===========================================================================
# POST /api/v1/assess tests
# ===========================================================================


@pytest.mark.asyncio
async def test_post_assess_returns_200(api_client):
    """POST /assess with valid body -> 200 for a decided case."""
    client, mock_orch, mock_storage = api_client
    mock_orch.run = AsyncMock(return_value=_DECIDED_RESULT)

    response = await client.post("/api/v1/assess", json=_VALID_REQUEST_BODY)

    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert data["decision"] == "APPROVED"
    assert "request_id" in data
    assert "confidence_score" in data


@pytest.mark.asyncio
async def test_post_assess_returns_202_escalated(api_client):
    """POST /assess when orchestrator returns ESCALATED -> 202."""
    client, mock_orch, mock_storage = api_client
    mock_orch.run = AsyncMock(return_value=_ESCALATED_RESULT)

    response = await client.post("/api/v1/assess", json=_VALID_REQUEST_BODY)

    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "escalated"


@pytest.mark.asyncio
async def test_post_assess_returns_422_invalid(api_client):
    """POST /assess with invalid body -> 422 (FastAPI validation error)."""
    client, mock_orch, mock_storage = api_client

    # Empty body -- missing required fields
    response = await client.post("/api/v1/assess", json={})

    assert response.status_code == 422


# ===========================================================================
# GET /api/v1/decisions tests
# ===========================================================================


@pytest.mark.asyncio
async def test_get_decision_returns_200(api_client):
    """GET /decisions/{id} with known ID -> 200."""
    client, mock_orch, mock_storage = api_client
    mock_storage.get = AsyncMock(return_value=_DECIDED_RESULT)

    response = await client.get(
        "/api/v1/decisions/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    )

    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "APPROVED"
    assert "request_id" in data


@pytest.mark.asyncio
async def test_get_decision_returns_404(api_client):
    """GET /decisions/{id} with unknown ID -> 404."""
    client, mock_orch, mock_storage = api_client
    mock_storage.get = AsyncMock(return_value=None)

    response = await client.get(
        "/api/v1/decisions/00000000-0000-0000-0000-000000000000"
    )

    assert response.status_code == 404


# ===========================================================================
# GET /api/v1/decisions/{id}/explain
# ===========================================================================


@pytest.mark.asyncio
async def test_get_explain_returns_200(api_client):
    """GET /decisions/{id}/explain -> 200 with explainability data."""
    client, mock_orch, mock_storage = api_client
    mock_storage.get = AsyncMock(return_value=_DECIDED_RESULT)

    response = await client.get(
        "/api/v1/decisions/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/explain"
    )

    assert response.status_code == 200
    data = response.json()
    assert "decision" in data
    assert "grounding_scores" in data
    assert "analysis_summary" in data
    assert "review_summary" in data
    assert "compliance_summary" in data


# ===========================================================================
# GET /api/v1/decisions/{id}/audit
# ===========================================================================


@pytest.mark.asyncio
async def test_get_audit_returns_200(api_client):
    """GET /decisions/{id}/audit -> 200 with audit trail."""
    client, mock_orch, mock_storage = api_client
    mock_storage.get = AsyncMock(return_value=_DECIDED_RESULT)

    response = await client.get(
        "/api/v1/decisions/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/audit"
    )

    assert response.status_code == 200
    data = response.json()
    assert "audit_trail" in data
    assert isinstance(data["audit_trail"], list)
    assert data["entry_count"] == len(data["audit_trail"])


# ===========================================================================
# GET /api/v1/health
# ===========================================================================


@pytest.mark.asyncio
async def test_get_health(api_client):
    """GET /health returns status with service checks."""
    client, mock_orch, mock_storage = api_client

    with (
        patch(
            "src.api.routes.health._check_weaviate",
            new_callable=AsyncMock,
            return_value={"status": "healthy"},
        ),
        patch(
            "src.api.routes.health._check_llm",
            new_callable=AsyncMock,
            return_value={"status": "unconfigured"},
        ),
    ):
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    # One healthy + one unconfigured = degraded
    assert data["status"] == "degraded"
    assert "services" in data


# ===========================================================================
# GET /api/v1/metrics
# ===========================================================================


@pytest.mark.asyncio
async def test_get_metrics(api_client):
    """GET /metrics returns Prometheus text format."""
    client, mock_orch, mock_storage = api_client

    response = await client.get("/api/v1/metrics")

    assert response.status_code == 200
    # Prometheus exposition format uses text/plain
    content_type = response.headers.get("content-type", "")
    assert "text/plain" in content_type or "text/plain" in str(
        response.headers
    )
    # Body should contain at least one Prometheus metric name
    body = response.text
    assert len(body) > 0
