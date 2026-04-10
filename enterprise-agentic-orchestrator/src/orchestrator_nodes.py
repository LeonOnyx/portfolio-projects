"""Pipeline node functions for the LangGraph credit-risk orchestration graph.

This module defines every async node that the Phase 7 LangGraph ``StateGraph``
will execute. Each node follows the same contract:

* Signature: ``async def xxx_node(state: dict) -> dict``
* Returns a *partial* dict (delta) -- never the full state, never mutated.
* Accumulating fields (``audit_trail``, ``grounding_scores``, ``errors``,
  ``retrieved_documents``) are returned as flat lists so the TypedDict
  ``operator.add`` reducer can append them to the existing state.

Import strategy
---------------
All imports of agents, guardrails, governance, models, config, and tool
adapters are **lazy** -- performed inside the function bodies. This avoids
import-time circular chains between ``orchestrator_nodes`` and the agents
it wraps (CrewAI/AutoGen pull in heavy dependencies that must not be
required at module import time).

The only top-level imports are stdlib (``json``, ``logging``) plus the
lightweight ``WorkflowStage`` enum from ``src.state`` (no transitive
external deps).

Graceful degradation (ORCH-06)
------------------------------
Agent wrapper nodes wrap their ``execute()`` calls in try/except: on
exception they log the error and return an error delta instead of
re-raising. The orchestrator decides whether to escalate or continue.

Grounding adapter nodes additionally apply a per-checkpoint circuit
breaker: after N consecutive Weaviate/embedding failures they stop
calling the vector store and return a zero-score result with
``circuit_broken=True`` until a successful call resets the counter.
"""

from __future__ import annotations

import json
import logging

from src.state import WorkflowStage

logger = logging.getLogger(__name__)


# ===========================================================================
# Grounding node factory (lazy, module-scoped singleton)
# ===========================================================================

_grounding_nodes: dict | None = None


def _get_grounding_nodes():
    """Lazily create the three grounding checkpoint nodes on first use.

    Instantiation is deferred so that importing ``orchestrator_nodes``
    does not pull in the grounding stack (which loads config and will
    eventually connect to Weaviate / the embedding provider).

    Returns
    -------
    dict
        Mapping from checkpoint position name
        (``post_analyst`` / ``post_reviewer`` / ``post_compliance``)
        to a :class:`GroundingCheckpointNode` instance sharing a
        single underlying :class:`GroundingChecker`.
    """
    global _grounding_nodes
    if _grounding_nodes is None:
        from src.guardrails.grounding_node import create_grounding_checkpoints

        _grounding_nodes = create_grounding_checkpoints()
    return _grounding_nodes


# ===========================================================================
# Circuit breaker (ORCH-06)
# ===========================================================================
#
# A lightweight in-process circuit breaker tracks consecutive Weaviate /
# embedding failures per checkpoint. After ``_CIRCUIT_BREAKER_THRESHOLD``
# consecutive failures the breaker trips: subsequent calls return a
# zero-score grounding result with ``circuit_broken=True`` until a
# successful call resets the counter.
#
# State is module-level (one process == one breaker) which is appropriate
# for a single-worker orchestrator. Multi-worker deployments would need
# to promote this to a shared store (e.g. Redis) but that is out of
# scope for Phase 7 Plan 1.

_circuit_breaker_failures: dict[str, int] = {}
_CIRCUIT_BREAKER_THRESHOLD = 3  # consecutive failures before tripping


def _circuit_breaker_record_failure(checkpoint_name: str) -> bool:
    """Increment the failure counter for *checkpoint_name*.

    Returns
    -------
    bool
        ``True`` if the breaker is now tripped (threshold reached),
        ``False`` otherwise.
    """
    _circuit_breaker_failures[checkpoint_name] = (
        _circuit_breaker_failures.get(checkpoint_name, 0) + 1
    )
    return _circuit_breaker_failures[checkpoint_name] >= _CIRCUIT_BREAKER_THRESHOLD


def _circuit_breaker_record_success(checkpoint_name: str) -> None:
    """Reset the failure counter for *checkpoint_name* on a successful call."""
    _circuit_breaker_failures[checkpoint_name] = 0


def _circuit_breaker_is_open(checkpoint_name: str) -> bool:
    """Return ``True`` if the breaker is currently tripped for *checkpoint_name*."""
    return (
        _circuit_breaker_failures.get(checkpoint_name, 0)
        >= _CIRCUIT_BREAKER_THRESHOLD
    )


# ===========================================================================
# Intake node (ORCH-02)
# ===========================================================================


async def intake_node(state: dict) -> dict:
    """Validate the incoming loan application, scan for PII, and start the audit trail.

    Steps
    -----
    1. Attempt to parse ``state["application"]`` into a
       :class:`LoanApplication`. On ``ValidationError`` return an error
       delta; the graph's routing will send the request to ESCALATE.
    2. Run :class:`PIIDetector` against the JSON-serialised application
       text (detect-only -- PII in input is recorded, not blocked).
    3. Create an :class:`AuditTrail` for the request and log two
       entries: ``input_received`` and ``pii_scan_complete``.

    Returns
    -------
    dict
        Partial state delta with ``current_stage``, ``pii_detected``,
        and ``audit_trail`` (flat list of entry dicts). On validation
        failure: an ``errors`` entry is also included.
    """
    logger.info("intake_node: entering")

    # Lazy imports keep the module import-time footprint minimal and
    # avoid circular chains with Pydantic models / guardrail config.
    from pydantic import ValidationError

    from src.governance.audit import AuditTrail, compute_content_hash
    from src.guardrails.pii import PIIDetector
    from src.models.loan import LoanApplication

    application_dict = state.get("application", {}) or {}
    request_id = state.get("request_id", "unknown")

    # -- 1. Validate LoanApplication ---------------------------------------
    try:
        app = LoanApplication(**application_dict)
    except ValidationError as exc:
        logger.error("intake_node: LoanApplication validation failed: %s", exc)
        return {
            "errors": [
                {
                    "stage": WorkflowStage.INTAKE.value,
                    "error": str(exc),
                }
            ],
            "current_stage": WorkflowStage.INTAKE.value,
        }

    # -- 2. PII scan (detect-only) -----------------------------------------
    application_json = json.dumps(application_dict, default=str)
    detector = PIIDetector()
    scan_result = detector.scan(application_json)

    if scan_result.pii_found:
        logger.warning(
            "intake_node: PII detected in application input: types=%s",
            scan_result.pii_types_detected,
        )

    # -- 3. Audit trail ----------------------------------------------------
    trail = AuditTrail(request_id)
    trail.add_entry(
        stage=WorkflowStage.INTAKE.value,
        action="input_received",
        details={"application_id": app.application_id},
        input_hash=compute_content_hash(application_json),
    )
    trail.add_entry(
        stage=WorkflowStage.INTAKE.value,
        action="pii_scan_complete",
        details={
            "pii_found": scan_result.pii_found,
            "pii_types": scan_result.pii_types_detected,
        },
    )

    logger.info(
        "intake_node: completed (application_id=%s, pii_found=%s)",
        app.application_id,
        scan_result.pii_found,
    )

    return {
        "current_stage": WorkflowStage.INTAKE.value,
        "pii_detected": scan_result.pii_found,
        "audit_trail": [entry.model_dump(mode="json") for entry in trail.entries],
    }


# ===========================================================================
# Agent wrapper nodes (ORCH-03 / ORCH-04 / ORCH-05)
# ===========================================================================


async def analysis_node(state: dict) -> dict:
    """Run the AnalystAgent on the current application.

    Wraps :class:`AnalystAgent.execute` in a try/except so that agent
    failures do not bring down the pipeline -- on exception we return
    an error delta instead of re-raising (ORCH-06 graceful degradation).
    """
    logger.info("analysis_node: entering")

    # Lazy imports
    from src.agents.analyst import AnalystAgent
    from src.agents.tools_adapter import get_analyst_tools
    from src.config.settings import ConfigLoader

    try:
        agent_config = ConfigLoader().agents().analyst.model_dump()
        agent = AnalystAgent(config=agent_config)
        tools = get_analyst_tools()

        context = {
            "application": state.get("application", {}),
            "retrieved_documents": state.get("retrieved_documents", []),
        }

        response = await agent.execute(context, tools)

        audit_entry = {
            "stage": WorkflowStage.ANALYSIS.value,
            "action": "analysis_complete",
            **agent.to_audit_entry(response),
        }

        logger.info(
            "analysis_node: completed (confidence=%.2f, sources=%d)",
            response.confidence,
            len(response.sources_used),
        )

        return {
            "analysis_result": response.output,
            "current_stage": WorkflowStage.ANALYSIS.value,
            "audit_trail": [audit_entry],
            "retrieved_documents": response.sources_used,
        }

    except Exception as exc:
        logger.error("analysis_node: AnalystAgent failed: %s", exc, exc_info=True)
        return {
            "errors": [
                {
                    "stage": WorkflowStage.ANALYSIS.value,
                    "error": str(exc),
                    "agent": "financial_analyst",
                }
            ],
            "current_stage": WorkflowStage.ANALYSIS.value,
        }


async def review_node(state: dict) -> dict:
    """Run the ReviewerAgent on the analyst's output.

    Follows the same graceful-degradation pattern as
    :func:`analysis_node`. The context passes the analyst report under
    both ``analysis_result`` (the plan's state-mirroring key) and
    ``analyst_report`` (the key the ReviewerAgent actually reads in
    ``reviewer.py::execute``).
    """
    logger.info("review_node: entering")

    # Lazy imports
    from src.agents.reviewer import ReviewerAgent
    from src.agents.tools_adapter import get_reviewer_tools
    from src.config.settings import ConfigLoader

    try:
        agent_config = ConfigLoader().agents().reviewer.model_dump()
        agent = ReviewerAgent(config=agent_config)
        tools = get_reviewer_tools()

        analyst_result = state.get("analysis_result", {})
        context = {
            "application": state.get("application", {}),
            "analysis_result": analyst_result,
            # ReviewerAgent.execute() reads context["analyst_report"]
            "analyst_report": analyst_result,
            "retrieved_documents": state.get("retrieved_documents", []),
        }

        response = await agent.execute(context, tools)

        audit_entry = {
            "stage": WorkflowStage.REVIEW.value,
            "action": "review_complete",
            **agent.to_audit_entry(response),
        }

        logger.info(
            "review_node: completed (confidence=%.2f, sources=%d)",
            response.confidence,
            len(response.sources_used),
        )

        return {
            "review_result": response.output,
            "current_stage": WorkflowStage.REVIEW.value,
            "audit_trail": [audit_entry],
            "retrieved_documents": response.sources_used,
        }

    except Exception as exc:
        logger.error("review_node: ReviewerAgent failed: %s", exc, exc_info=True)
        return {
            "errors": [
                {
                    "stage": WorkflowStage.REVIEW.value,
                    "error": str(exc),
                    "agent": "independent_reviewer",
                }
            ],
            "current_stage": WorkflowStage.REVIEW.value,
        }


async def compliance_node(state: dict) -> dict:
    """Run the ComplianceAgent on the analyst + reviewer output.

    Follows the same graceful-degradation pattern as
    :func:`analysis_node`. The context passes the upstream reports under
    both the plan's state-mirroring keys (``analysis_result`` /
    ``review_result``) and the keys the ComplianceAgent actually reads
    in ``compliance.py::_build_task`` (``analyst_report`` /
    ``reviewer_report``).
    """
    logger.info("compliance_node: entering")

    # Lazy imports
    from src.agents.compliance import ComplianceAgent
    from src.agents.tools_adapter import get_compliance_tools_autogen
    from src.config.settings import ConfigLoader

    try:
        agent_config = ConfigLoader().agents().compliance.model_dump()
        agent = ComplianceAgent(config=agent_config)
        tools = get_compliance_tools_autogen()

        analyst_result = state.get("analysis_result", {})
        reviewer_result = state.get("review_result", {})
        context = {
            "application": state.get("application", {}),
            "analysis_result": analyst_result,
            "review_result": reviewer_result,
            # ComplianceAgent._build_task() reads these keys:
            "analyst_report": analyst_result,
            "reviewer_report": reviewer_result,
            "retrieved_documents": state.get("retrieved_documents", []),
        }

        response = await agent.execute(context, tools)

        audit_entry = {
            "stage": WorkflowStage.COMPLIANCE.value,
            "action": "compliance_complete",
            **agent.to_audit_entry(response),
        }

        logger.info(
            "compliance_node: completed (confidence=%.2f, sources=%d)",
            response.confidence,
            len(response.sources_used),
        )

        return {
            "compliance_result": response.output,
            "current_stage": WorkflowStage.COMPLIANCE.value,
            "audit_trail": [audit_entry],
            "retrieved_documents": response.sources_used,
        }

    except Exception as exc:
        logger.error(
            "compliance_node: ComplianceAgent failed: %s", exc, exc_info=True
        )
        return {
            "errors": [
                {
                    "stage": WorkflowStage.COMPLIANCE.value,
                    "error": str(exc),
                    "agent": "compliance_officer",
                }
            ],
            "current_stage": WorkflowStage.COMPLIANCE.value,
        }


# ===========================================================================
# Grounding adapter nodes (bridge OrchestratorState <-> GroundingCheckpointNode)
# ===========================================================================
#
# Interface mismatch the adapters bridge
# --------------------------------------
# GroundingCheckpointNode expects (per grounding_node.py::__call__):
#     state["agent_output"]       : str
#     state["source_documents"]   : list
#
# OrchestratorState stores agent outputs as dicts under:
#     state["analysis_result"]    : dict   (AnalysisReport)
#     state["review_result"]      : dict   (ReviewReport)
#     state["compliance_result"]  : dict   (ComplianceReport)
# and retrieved documents under:
#     state["retrieved_documents"] : list
#
# Each adapter extracts the relevant result dict, serialises it to JSON
# (so GroundingChecker has text to score against sources), builds a
# private dict matching GroundingCheckpointNode's expected shape, calls
# the checkpoint, and translates the result back to OrchestratorState
# fields (``grounding_scores`` / ``audit_trail``).
#
# All three adapters share ``_run_grounding_adapter`` to guarantee
# consistent circuit-breaker and error-handling behaviour.


async def _run_grounding_adapter(
    state: dict,
    checkpoint_name: str,
    result_field: str,
    stage: str,
) -> dict:
    """Shared adapter logic for every grounding checkpoint node.

    Parameters
    ----------
    state:
        The current OrchestratorState.
    checkpoint_name:
        One of ``"post_analyst"``, ``"post_reviewer"``, ``"post_compliance"``
        -- the key used by :func:`create_grounding_checkpoints`.
    result_field:
        The OrchestratorState field containing the agent output dict
        to verify (``"analysis_result"`` / ``"review_result"`` /
        ``"compliance_result"``).
    stage:
        The :class:`WorkflowStage` string value to record against
        the audit entry.

    Returns
    -------
    dict
        Partial state delta with ``current_stage``, ``grounding_scores``
        (flat list with a single dict -- the reducer will append it),
        and ``audit_trail``.

    Notes
    -----
    Implements the ORCH-06 circuit breaker. Before calling the grounding
    checkpoint we check whether the breaker is open for this checkpoint
    name -- if so we short-circuit with a zero-score result flagged
    ``circuit_broken=True`` and skip the Weaviate / embedding call
    entirely. Every successful call resets the per-checkpoint failure
    counter; every exception increments it, tripping the breaker once
    the threshold is reached.
    """
    logger.info(
        "grounding adapter '%s': entering (result_field=%s)",
        checkpoint_name,
        result_field,
    )

    # Lazy import of Langfuse grounding span helpers (graceful degradation)
    from src.observability.tracing import create_grounding_span, end_grounding_span

    grounding_span = None

    # -- Fast path: breaker already tripped -------------------------------
    if _circuit_breaker_is_open(checkpoint_name):
        logger.warning(
            "grounding adapter '%s': circuit breaker OPEN -- returning "
            "zero-score without calling vector DB",
            checkpoint_name,
        )
        cb_span = create_grounding_span(checkpoint_name)
        end_grounding_span(cb_span, score=0.0, is_grounded=False, checkpoint_name=checkpoint_name)
        return {
            "current_stage": stage,
            "grounding_scores": [
                {
                    "checkpoint": checkpoint_name,
                    "score": 0.0,
                    "is_grounded": False,
                    "circuit_broken": True,
                    "error": (
                        f"Circuit breaker open after "
                        f"{_CIRCUIT_BREAKER_THRESHOLD} consecutive failures"
                    ),
                }
            ],
            "audit_trail": [
                {
                    "stage": stage,
                    "action": "grounding_check",
                    "details": {
                        "checkpoint": checkpoint_name,
                        "score": 0.0,
                        "circuit_broken": True,
                    },
                }
            ],
        }

    # -- Normal path: call the grounding checkpoint -----------------------
    try:
        grounding_span = create_grounding_span(checkpoint_name)
        nodes = _get_grounding_nodes()
        grounding_node = nodes[checkpoint_name]

        # Serialise the agent output dict to JSON so GroundingChecker
        # has text to split into claims and embed.
        output_dict = state.get(result_field, {}) or {}
        output_text = json.dumps(output_dict, default=str)
        sources = state.get("retrieved_documents", []) or []

        adapter_dict: dict = {
            "agent_output": output_text,
            "source_documents": sources,
        }

        # GroundingCheckpointNode returns the same dict enriched with
        # ``grounding_results[checkpoint_name]`` and possibly
        # ``needs_reprompt``.
        result = await grounding_node(adapter_dict)

        result_data = (
            result.get("grounding_results", {}).get(checkpoint_name, {}) or {}
        )
        # GroundingResult.model_dump uses the field name ``grounding_score``
        # (see src/models/governance.py) -- fall back to ``score`` for
        # forward compatibility with any alternative serialisation shape.
        grounding_score = result_data.get(
            "grounding_score", result_data.get("score", 0.0)
        )
        is_grounded = result_data.get("is_grounded", False)

        end_grounding_span(grounding_span, score=grounding_score, is_grounded=is_grounded, checkpoint_name=checkpoint_name)

        # Success -- reset the per-checkpoint failure counter
        _circuit_breaker_record_success(checkpoint_name)

        logger.info(
            "grounding adapter '%s': score=%.2f is_grounded=%s",
            checkpoint_name,
            grounding_score,
            is_grounded,
        )

        return {
            "current_stage": stage,
            "grounding_scores": [
                {
                    "checkpoint": checkpoint_name,
                    "score": grounding_score,
                    "is_grounded": is_grounded,
                }
            ],
            "audit_trail": [
                {
                    "stage": stage,
                    "action": "grounding_check",
                    "details": {
                        "checkpoint": checkpoint_name,
                        "score": grounding_score,
                        "is_grounded": is_grounded,
                    },
                }
            ],
        }

    except Exception as exc:
        logger.error(
            "grounding adapter '%s' failed: %s",
            checkpoint_name,
            exc,
            exc_info=True,
        )

        end_grounding_span(grounding_span, score=0.0, is_grounded=False, checkpoint_name=checkpoint_name)

        # Record failure; may trip the breaker.
        circuit_broken = _circuit_breaker_record_failure(checkpoint_name)
        if circuit_broken:
            logger.warning(
                "grounding adapter '%s': circuit breaker TRIPPED after "
                "%d consecutive failures",
                checkpoint_name,
                _CIRCUIT_BREAKER_THRESHOLD,
            )

        return {
            "current_stage": stage,
            "grounding_scores": [
                {
                    "checkpoint": checkpoint_name,
                    "score": 0.0,
                    "is_grounded": False,
                    "error": str(exc),
                    "circuit_broken": circuit_broken,
                }
            ],
            "audit_trail": [
                {
                    "stage": stage,
                    "action": "grounding_check",
                    "details": {
                        "checkpoint": checkpoint_name,
                        "score": 0.0,
                        "error": str(exc),
                        "circuit_broken": circuit_broken,
                    },
                }
            ],
        }


async def grounding_analysis_node(state: dict) -> dict:
    """Grounding checkpoint after the AnalystAgent (``post_analyst``).

    Verifies the analyst report's factual claims against the documents
    retrieved so far. On vector-DB failure records a failure with the
    ORCH-06 circuit breaker and returns a zero-score delta.
    """
    return await _run_grounding_adapter(
        state,
        checkpoint_name="post_analyst",
        result_field="analysis_result",
        stage=WorkflowStage.GROUNDING_ANALYSIS.value,
    )


async def grounding_review_node(state: dict) -> dict:
    """Grounding checkpoint after the ReviewerAgent (``post_reviewer``).

    Verifies the reviewer report's factual claims against the accumulated
    retrieved documents. On vector-DB failure records a failure with the
    ORCH-06 circuit breaker and returns a zero-score delta.
    """
    return await _run_grounding_adapter(
        state,
        checkpoint_name="post_reviewer",
        result_field="review_result",
        stage=WorkflowStage.GROUNDING_REVIEW.value,
    )


async def grounding_compliance_node(state: dict) -> dict:
    """Grounding checkpoint after the ComplianceAgent (``post_compliance``).

    Verifies the compliance report's regulatory citations against the
    accumulated retrieved documents. On vector-DB failure records a
    failure with the ORCH-06 circuit breaker and returns a zero-score
    delta.
    """
    return await _run_grounding_adapter(
        state,
        checkpoint_name="post_compliance",
        result_field="compliance_result",
        stage=WorkflowStage.GROUNDING_COMPLIANCE.value,
    )
