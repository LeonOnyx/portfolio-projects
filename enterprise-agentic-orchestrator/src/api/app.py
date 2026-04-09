"""Credit Risk Assessment API.

FastAPI application exposing the CreditRiskOrchestrator as REST endpoints
with Langfuse tracing and Prometheus metrics.

Usage:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Path shim (same pattern as orchestrator.py)
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.api.dependencies import init_dependencies
from src.observability.metrics import setup_instrumentator
from src.observability.tracing import flush_langfuse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup, clean up on shutdown."""
    logger.info("Starting Credit Risk Assessment API...")
    init_dependencies()
    logger.info("Dependencies initialised")
    yield
    logger.info("Shutting down Credit Risk Assessment API...")
    flush_langfuse()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Credit Risk Assessment API",
    description=(
        "Enterprise Agentic Orchestrator -- governed multi-agent credit risk "
        "assessment with explainability, grounding verification, and audit trails."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware -- permissive for demo, tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus HTTP instrumentation (excludes health and metrics endpoints)
setup_instrumentator(app)

# =====================================================================
# Mount route modules
# =====================================================================

from src.api.routes.assess import router as assess_router  # noqa: E402

app.include_router(assess_router)

# decisions and health routers are created in Plan 08-04 -- import with
# graceful fallback so the app starts with just the assess endpoint.
try:
    from src.api.routes.decisions import router as decisions_router  # noqa: E402

    app.include_router(decisions_router)
except ImportError:
    logger.info("decisions router not yet available")

try:
    from src.api.routes.health import router as health_router  # noqa: E402

    app.include_router(health_router)
except ImportError:
    logger.info("health router not yet available")
