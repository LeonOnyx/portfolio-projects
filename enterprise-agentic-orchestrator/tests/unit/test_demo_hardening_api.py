"""Tests for Phase 11 API hardening fixes (P0-1, P0-2, P0-9, P0-10).

Covers:
- P0-1: CORS credentials disabled with wildcard origins
- P0-2: assess_application has try/except for structured errors
- P0-9: Dependency guards raise RuntimeError, not AssertionError
- P0-10: Health aggregation reports degraded when LLM is unconfigured
"""

import ast
import inspect

import pytest


# -----------------------------------------------------------------------
# P0-1: CORS must not combine wildcard origins with credentials=True
# -----------------------------------------------------------------------


def test_cors_credentials_disabled():
    """P0-1: CORS must not combine wildcard origins with credentials=True."""
    from src.api.app import app

    cors_mw = None
    for mw in app.user_middleware:
        if "CORSMiddleware" in str(mw.cls):
            cors_mw = mw
            break
    assert cors_mw is not None, "CORSMiddleware not found"
    assert cors_mw.kwargs.get("allow_credentials") is False, (
        "allow_credentials must be False when using wildcard origins"
    )


# -----------------------------------------------------------------------
# P0-9: Dependency guards must raise RuntimeError, not AssertionError
# -----------------------------------------------------------------------


def test_get_orchestrator_raises_runtime_error():
    """P0-9: get_orchestrator must raise RuntimeError, not AssertionError."""
    from src.api import dependencies

    original = dependencies._orchestrator
    dependencies._orchestrator = None
    try:
        with pytest.raises(RuntimeError, match="not initialised"):
            dependencies.get_orchestrator()
    finally:
        dependencies._orchestrator = original


def test_get_storage_raises_runtime_error():
    """P0-9: get_storage must raise RuntimeError, not AssertionError."""
    from src.api import dependencies

    original = dependencies._storage
    dependencies._storage = None
    try:
        with pytest.raises(RuntimeError, match="not initialised"):
            dependencies.get_storage()
    finally:
        dependencies._storage = original


# -----------------------------------------------------------------------
# P0-2: assess_application must have try/except for structured errors
# -----------------------------------------------------------------------


def test_assess_endpoint_has_exception_handler():
    """P0-2: assess_application must have try/except for structured errors."""
    from src.api.routes.assess import assess_application

    source = inspect.getsource(assess_application)
    tree = ast.parse(source)
    handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
    assert len(handlers) >= 1, "assess_application needs at least one except handler"


# -----------------------------------------------------------------------
# P0-10: Health aggregation must report degraded when LLM unconfigured
# -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_degraded_when_llm_unconfigured():
    """P0-10: Overall health must be degraded when LLM is unconfigured."""
    from unittest.mock import AsyncMock, patch

    from src.api.routes.health import health_check

    with (
        patch(
            "src.api.routes.health._check_weaviate", new_callable=AsyncMock
        ) as mock_wv,
        patch(
            "src.api.routes.health._check_llm", new_callable=AsyncMock
        ) as mock_llm,
    ):
        mock_wv.return_value = {"status": "healthy", "ready": True}
        mock_llm.return_value = {
            "status": "unconfigured",
            "error": "creds missing",
        }
        result = await health_check()
        assert result.status == "degraded", (
            f"Expected 'degraded' when LLM is unconfigured, got '{result.status}'"
        )


@pytest.mark.asyncio
async def test_health_unhealthy_when_all_services_down():
    """P0-10: Overall health must be unhealthy when all services are down."""
    from unittest.mock import AsyncMock, patch

    from src.api.routes.health import health_check

    with (
        patch(
            "src.api.routes.health._check_weaviate", new_callable=AsyncMock
        ) as mock_wv,
        patch(
            "src.api.routes.health._check_llm", new_callable=AsyncMock
        ) as mock_llm,
    ):
        mock_wv.return_value = {"status": "unhealthy", "error": "unreachable"}
        mock_llm.return_value = {"status": "unhealthy", "error": "unreachable"}
        result = await health_check()
        assert result.status == "unhealthy", (
            f"Expected 'unhealthy' when all services down, got '{result.status}'"
        )
