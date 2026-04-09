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


# ---------------------------------------------------------------------------
# Langfuse v4 span helper functions (gap closure: OBS-01/02/03)
# ---------------------------------------------------------------------------
#
# API surface validated via inspect.signature on langfuse 4.0.6:
#
#   Langfuse.start_observation(name: str, as_type: Literal['span','agent',
#       'tool','chain','retriever','evaluator','guardrail','generation',
#       'embedding'], trace_context: Optional[TraceContext], metadata: Any,
#       usage_details: Optional[Dict[str,int]], cost_details: Optional[Dict[str,float]], ...)
#   LangfuseSpan.update(metadata, usage_details, cost_details, output, ...)
#   LangfuseSpan.end(end_time: Optional[int])
#   LangfuseSpan.score(name: str, value: Union[float,str], ...)
#   TraceContext = TypedDict('TraceContext', trace_id: str, parent_span_id: NotRequired[str])
#
# CrewAI Crew model_fields (crewai installed version):
#   Callback-related: step_callback, task_callback, before_kickoff_callbacks,
#       after_kickoff_callbacks
#   NO native 'langfuse' parameter exists on Crew.
#   Instrumentation must use explicit span wrapping (same approach as AutoGen).
#   Plan 08-06 should wrap agent calls with start_observation/end rather than
#   passing a Langfuse handler to CrewAI's constructor.
# ---------------------------------------------------------------------------


def create_agent_span(
    agent_name: str, trace_id: str | None = None
) -> "Any | None":
    """Create a Langfuse observation span for an agent execution.

    Uses Langfuse v4 ``start_observation()`` with ``as_type='agent'``.
    If *trace_id* is provided it is attached via a ``TraceContext`` dict so the
    span nests under the correct trace.

    Returns the ``LangfuseSpan`` object, or ``None`` when Langfuse is
    unavailable (missing env vars, import errors, SDK errors).
    """
    try:
        from langfuse import get_client

        client = get_client()

        trace_context = None
        if trace_id is not None:
            trace_context = {"trace_id": trace_id}

        span = client.start_observation(
            name=f"agent:{agent_name}",
            as_type="agent",
            metadata={"agent_name": agent_name},
            trace_context=trace_context,
        )
        return span
    except ImportError:
        logger.warning(
            "langfuse not installed; create_agent_span returning None"
        )
        return None
    except Exception as exc:
        logger.warning(
            "create_agent_span failed (%s); returning None", exc
        )
        return None


def end_agent_span(span, agent_response_or_dict: dict) -> None:
    """Attach usage metadata to an agent span and end it.

    Expects *agent_response_or_dict* to contain any of:
      - ``tokens_used`` (int) -- split 60/40 into input/output token estimates
      - ``latency_ms`` (float)
      - ``agent_name`` (str)

    If *span* is ``None`` (graceful degradation path) this is a no-op.
    """
    if span is None:
        return

    try:
        tokens_used = agent_response_or_dict.get("tokens_used", 0)
        latency_ms = agent_response_or_dict.get("latency_ms")
        agent_name = agent_response_or_dict.get("agent_name", "unknown")

        # usage_details values must be int per Langfuse v4 API
        input_tokens = int(tokens_used * 0.6) if tokens_used else 0
        output_tokens = int(tokens_used * 0.4) if tokens_used else 0

        usage_details: dict[str, int] = {}
        if tokens_used:
            usage_details["input"] = input_tokens
            usage_details["output"] = output_tokens
            usage_details["total"] = int(tokens_used)

        update_metadata: dict = {"agent_name": agent_name}
        if latency_ms is not None:
            update_metadata["latency_ms"] = latency_ms

        span.update(
            metadata=update_metadata,
            usage_details=usage_details if usage_details else None,
            output=agent_response_or_dict,
        )
        span.end()
    except Exception as exc:
        logger.warning("end_agent_span failed (%s); span may be incomplete", exc)


def create_grounding_span(
    checkpoint_name: str, trace_id: str | None = None
) -> "Any | None":
    """Create a Langfuse observation span for a grounding checkpoint.

    Uses ``as_type='evaluator'`` which is the closest semantic match for a
    grounding verification step in Langfuse v4's type vocabulary.

    Returns the ``LangfuseSpan`` object, or ``None`` on failure.
    """
    try:
        from langfuse import get_client

        client = get_client()

        trace_context = None
        if trace_id is not None:
            trace_context = {"trace_id": trace_id}

        span = client.start_observation(
            name=f"grounding:{checkpoint_name}",
            as_type="evaluator",
            metadata={"checkpoint_name": checkpoint_name},
            trace_context=trace_context,
        )
        return span
    except ImportError:
        logger.warning(
            "langfuse not installed; create_grounding_span returning None"
        )
        return None
    except Exception as exc:
        logger.warning(
            "create_grounding_span failed (%s); returning None", exc
        )
        return None


def end_grounding_span(
    span,
    score: float,
    is_grounded: bool,
    checkpoint_name: str,
) -> None:
    """Attach grounding results to a span and end it.

    Records *score* and *is_grounded* as both span metadata and a Langfuse
    score (``span.score()``) so grounding quality is queryable in the
    Langfuse dashboard.

    If *span* is ``None`` (graceful degradation path) this is a no-op.
    """
    if span is None:
        return

    try:
        span.update(
            metadata={
                "checkpoint_name": checkpoint_name,
                "grounding_score": score,
                "is_grounded": is_grounded,
            },
            output={
                "score": score,
                "is_grounded": is_grounded,
                "checkpoint_name": checkpoint_name,
            },
        )
        # Record a numeric score for Langfuse analytics
        span.score(
            name=f"grounding:{checkpoint_name}",
            value=score,
        )
        span.end()
    except Exception as exc:
        logger.warning(
            "end_grounding_span failed (%s); span may be incomplete", exc
        )
