"""FastAPI dependency injection factories.

Provides singleton instances of the :class:`CreditRiskOrchestrator`,
:class:`AssessmentStorage`, and (optionally) a Langfuse callback handler.

Call :func:`init_dependencies` once at application startup (in the
``lifespan`` context manager or ``@app.on_event("startup")``).  The
``get_*`` functions are designed for use with ``Depends()`` in route
signatures.

Usage in a route::

    @router.post("/assess")
    async def assess(
        orch: CreditRiskOrchestrator = Depends(get_orchestrator),
        storage: AssessmentStorage = Depends(get_storage),
    ):
        ...
"""

from __future__ import annotations

import logging

from src.api.storage import AssessmentStorage
from src.orchestrator import CreditRiskOrchestrator

logger = logging.getLogger(__name__)

# =====================================================================
# Module-level singletons (set by init_dependencies)
# =====================================================================

_orchestrator: CreditRiskOrchestrator | None = None
_storage: AssessmentStorage | None = None
_langfuse_handler = None  # type: ignore[assignment]


# =====================================================================
# Initialisation
# =====================================================================


def init_dependencies(data_dir: str = "data/assessments") -> None:
    """Create shared resource singletons.

    Must be called exactly once at application startup.  Subsequent
    calls overwrite previous instances (useful in tests).

    Parameters
    ----------
    data_dir:
        Filesystem path passed through to :class:`AssessmentStorage`.
    """
    global _orchestrator, _storage, _langfuse_handler

    _orchestrator = CreditRiskOrchestrator()
    _storage = AssessmentStorage(data_dir=data_dir)

    # Langfuse is optional -- if env vars (LANGFUSE_PUBLIC_KEY,
    # LANGFUSE_SECRET_KEY, LANGFUSE_HOST) are missing or the package
    # is not installed, observability simply degrades gracefully.
    try:
        from langfuse.langchain import CallbackHandler

        _langfuse_handler = CallbackHandler()
        logger.info("Langfuse callback handler initialised")
    except Exception as exc:
        _langfuse_handler = None
        logger.info(
            "Langfuse handler not available (%s: %s) -- "
            "observability will run without Langfuse tracing",
            type(exc).__name__,
            exc,
        )

    logger.info(
        "API dependencies initialised (orchestrator=%s, storage=%s, langfuse=%s)",
        type(_orchestrator).__name__,
        data_dir,
        "enabled" if _langfuse_handler else "disabled",
    )


# =====================================================================
# Dependency getters (for FastAPI Depends())
# =====================================================================


def get_orchestrator() -> CreditRiskOrchestrator:
    """Return the shared CreditRiskOrchestrator instance.

    Raises :class:`RuntimeError` if :func:`init_dependencies` has
    not been called -- this is a programming error, not a runtime
    condition.
    """
    if _orchestrator is None:
        raise RuntimeError(
            "CreditRiskOrchestrator not initialised -- "
            "call init_dependencies() at app startup"
        )
    return _orchestrator


def get_storage() -> AssessmentStorage:
    """Return the shared AssessmentStorage instance."""
    if _storage is None:
        raise RuntimeError(
            "AssessmentStorage not initialised -- "
            "call init_dependencies() at app startup"
        )
    return _storage


def get_langfuse_handler():
    """Return the Langfuse callback handler, or ``None`` if unavailable.

    Routes should treat a ``None`` return as "no observability" and
    continue without tracing -- never fail a request because Langfuse
    is down.
    """
    return _langfuse_handler
