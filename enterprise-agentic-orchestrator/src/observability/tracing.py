"""Langfuse tracing integration for the credit risk orchestrator.

Uses Langfuse v4 SDK (pinned 4.0.6). Key API:
- get_client() for singleton client
- CallbackHandler for LangGraph/LangChain auto-tracing
- propagate_attributes() for per-request trace metadata

IMPORTANT v4 changes (do NOT use v2/v3 API):
- No langfuse_context (removed)
- No update_trace param on CallbackHandler (removed)
- Use propagate_attributes() context manager instead
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_langfuse_handler():
    """Create a reusable Langfuse CallbackHandler for LangGraph tracing.

    Returns None if Langfuse env vars (LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY)
    are not set. The API should work without Langfuse -- tracing is optional.
    """
    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()
        logger.info("Langfuse CallbackHandler created successfully")
        return handler
    except Exception as exc:
        logger.warning(
            "Langfuse CallbackHandler creation failed (%s); tracing disabled",
            exc,
        )
        return None


async def traced_orchestrator_run(
    orchestrator,
    application: dict,
    request_id: str,
    user_role: str = "default",
    langfuse_handler=None,
) -> dict:
    """Run the orchestrator with Langfuse tracing wrapped around it.

    If langfuse_handler is provided, wraps the call in propagate_attributes()
    to inject trace metadata (trace_name, session_id, user_id, tags) and
    passes the handler via config={"callbacks": [handler]} to the orchestrator.

    If langfuse_handler is None, calls orchestrator.run() without tracing.
    """
    if langfuse_handler is None:
        return await orchestrator.run(
            application, request_id=request_id, user_role=user_role
        )

    try:
        from langfuse import propagate_attributes

        with propagate_attributes(
            trace_name="credit-risk-assessment",
            user_id=user_role,
            session_id=request_id,
            metadata={
                "application_id": application.get("application_id", ""),
                "endpoint": "/api/v1/assess",
            },
            tags=["credit-risk", "api"],
        ):
            config = {"callbacks": [langfuse_handler]}
            result = await orchestrator.run(
                application,
                request_id=request_id,
                user_role=user_role,
                config=config,
            )
        return result
    except Exception as exc:
        logger.warning(
            "Langfuse tracing failed (%s); running without tracing", exc
        )
        return await orchestrator.run(
            application, request_id=request_id, user_role=user_role
        )


def flush_langfuse() -> None:
    """Flush and shutdown Langfuse client. Called on app shutdown."""
    try:
        from langfuse import get_client

        client = get_client()
        client.flush()
        client.shutdown()
        logger.info("Langfuse client flushed and shut down")
    except Exception as exc:
        logger.warning("Langfuse shutdown failed (%s)", exc)
