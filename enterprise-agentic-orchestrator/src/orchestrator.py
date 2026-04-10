"""Credit Risk Orchestrator -- LangGraph State Machine.

Complete rewrite of the original dataclass-based orchestrator. Uses
LangGraph :class:`StateGraph` over the :class:`OrchestratorState` TypedDict
(``src.state``), with conditional routing for grounding retry loops, a
LangGraph :class:`RetryPolicy` for transient LLM/embedding errors, and a
deterministic decision matrix terminating the pipeline.

Pipeline
--------
``INTAKE -> ANALYSIS -> GROUNDING_ANALYSIS -> REVIEW -> GROUNDING_REVIEW
-> COMPLIANCE -> GROUNDING_COMPLIANCE -> DECISION`` (with ``ESCALATE``
as a terminal off-ramp from intake validation, every grounding
retry-exhaustion path, and the decision node when escalation triggers
fire).

Phase 7 (ORCH-01 through ORCH-06).
"""

from __future__ import annotations

import logging
import os
import sys
from uuid import uuid4

# ---------------------------------------------------------------------------
# Path shim: when this module is run as a script (``python src/orchestrator.py``)
# Python sets sys.path[0] to the script's directory (``src/``), so the
# absolute imports below (``from src.orchestrator_decision import ...``)
# would fail. Insert the project root onto sys.path before any first-party
# imports happen. Imported as a module (``import src.orchestrator``), the
# package is already resolvable, so the second insert is a no-op against the
# existing sys.path entries -- no risk of duplicate / shadowing.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from langgraph.graph import END, START, StateGraph  # noqa: E402

from src.orchestrator_decision import decision_node, escalation_node  # noqa: E402
from src.orchestrator_nodes import (  # noqa: E402
    analysis_node,
    compliance_node,
    grounding_analysis_node,
    grounding_compliance_node,
    grounding_review_node,
    intake_node,
    review_node,
)
from src.state import OrchestratorState, WorkflowStage  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# RetryPolicy import (version-tolerant)
# ---------------------------------------------------------------------------
#
# The :class:`RetryPolicy` location has moved between LangGraph minor
# releases. We try the historical locations in order and fall back to
# ``None`` (no retry) if every import fails. The agent / grounding nodes
# already have their own try/except graceful-degradation logic, so the
# absence of LangGraph-level retries only weakens transient-error
# resilience -- it does not break the pipeline.
RetryPolicy = None  # type: ignore[assignment]
for _retry_module in (
    "langgraph.types",
    "langgraph.pregel",
    "langgraph.pregel.retry",
):
    try:
        _mod = __import__(_retry_module, fromlist=["RetryPolicy"])
        RetryPolicy = getattr(_mod, "RetryPolicy")  # type: ignore[assignment]
        logger.debug("RetryPolicy imported from %s", _retry_module)
        break
    except (ImportError, AttributeError):
        continue

if RetryPolicy is None:
    logger.warning(
        "RetryPolicy not available in installed langgraph version -- "
        "LLM nodes will run without LangGraph-level retry. Internal "
        "node try/except still provides graceful degradation."
    )


# ---------------------------------------------------------------------------
# Grounding retry budget (loaded once at module import)
# ---------------------------------------------------------------------------

try:
    from src.config.settings import ConfigLoader

    _MAX_GROUNDING_RETRIES = int(
        ConfigLoader().guardrails().grounding.max_retries
    )
except Exception as _exc:  # pragma: no cover -- defensive
    logger.warning(
        "Could not load grounding.max_retries from guardrails.yaml (%s); "
        "falling back to 2.",
        _exc,
    )
    _MAX_GROUNDING_RETRIES = 2


# ===========================================================================
# Routing functions (module-level so they are easy to unit-test)
# ===========================================================================


def route_after_intake(state: dict) -> str:
    """Route after the intake node.

    If the intake node emitted any error (LoanApplication validation
    failed), the application is unprocessable and we route straight to
    ESCALATE. Otherwise we proceed to the analyst.
    """
    errors = state.get("errors", []) or []
    for err in errors:
        if isinstance(err, dict) and err.get("stage") == WorkflowStage.INTAKE.value:
            return "escalate"
        # String errors emitted by other nodes occasionally end up here;
        # check for the literal "intake" prefix as well.
        if isinstance(err, str) and err.lower().startswith("intake"):
            return "escalate"
    return "analysis"


def _route_after_grounding(
    state: dict,
    checkpoint_name: str,
    proceed_to: str,
    retry_to: str,
) -> str:
    """Shared routing logic for grounding-checkpoint outputs.

    Inspects ``state["grounding_scores"]`` for entries matching
    ``checkpoint_name`` and decides whether to:

    * proceed to the next stage (``proceed_to``) -- latest entry is
      grounded;
    * retry the previous agent (``retry_to``) -- not yet grounded but
      retry budget remaining;
    * escalate to human review -- retry budget exhausted.
    """
    scores = state.get("grounding_scores", []) or []
    if not isinstance(scores, list):
        return retry_to

    matching = [
        entry
        for entry in scores
        if isinstance(entry, dict) and entry.get("checkpoint") == checkpoint_name
    ]

    if not matching:
        # Grounding node never produced an entry for this checkpoint --
        # we cannot prove the output is grounded, so escalate.
        logger.warning(
            "_route_after_grounding(%s): no entries -- escalating",
            checkpoint_name,
        )
        return "escalate"

    latest = matching[-1]

    # Circuit-breaker open results count as failures, but they should
    # never be retried (the breaker exists precisely because retries are
    # not helping). Escalate immediately on a circuit-broken entry.
    if latest.get("circuit_broken"):
        logger.warning(
            "_route_after_grounding(%s): circuit breaker open -- escalating",
            checkpoint_name,
        )
        return "escalate"

    if latest.get("is_grounded"):
        return proceed_to

    attempts = len(matching)
    if attempts >= _MAX_GROUNDING_RETRIES:
        logger.warning(
            "_route_after_grounding(%s): retries exhausted (%d/%d) -- escalating",
            checkpoint_name,
            attempts,
            _MAX_GROUNDING_RETRIES,
        )
        return "escalate"

    logger.info(
        "_route_after_grounding(%s): not grounded, retry %d/%d -- back to %s",
        checkpoint_name,
        attempts,
        _MAX_GROUNDING_RETRIES,
        retry_to,
    )
    return retry_to


def route_after_grounding_analysis(state: dict) -> str:
    """Conditional edge after the post-analyst grounding checkpoint."""
    return _route_after_grounding(
        state,
        checkpoint_name="post_analyst",
        proceed_to="review",
        retry_to="analysis",
    )


def route_after_grounding_review(state: dict) -> str:
    """Conditional edge after the post-reviewer grounding checkpoint."""
    return _route_after_grounding(
        state,
        checkpoint_name="post_reviewer",
        proceed_to="compliance",
        retry_to="review",
    )


def route_after_grounding_compliance(state: dict) -> str:
    """Conditional edge after the post-compliance grounding checkpoint."""
    return _route_after_grounding(
        state,
        checkpoint_name="post_compliance",
        proceed_to="decision",
        retry_to="compliance",
    )


def route_after_decision(state: dict) -> str:
    """Conditional edge after the decision node.

    If ``decision_node`` flagged the request for escalation (any
    escalation trigger fired, or the matrix produced an ESCALATED
    outcome), divert to the escalation terminal node. Otherwise the
    pipeline ends with the synthesised Decision.
    """
    if state.get("requires_escalation", False):
        return "escalate"
    # END is the literal string "__end__"; using the same literal in the
    # routing map keeps the conditional-edges definition self-documenting.
    return "__end__"


# ===========================================================================
# CreditRiskOrchestrator
# ===========================================================================


class CreditRiskOrchestrator:
    """LangGraph state machine for governed multi-agent credit risk assessment.

    Wires nine nodes into a single compiled :class:`StateGraph`:

    1. ``intake`` -- LoanApplication validation, PII scan, audit-trail init
       (:func:`src.orchestrator_nodes.intake_node`).
    2. ``analysis`` -- AnalystAgent execute wrapper.
    3. ``grounding_analysis`` -- post-analyst grounding checkpoint adapter.
    4. ``review`` -- ReviewerAgent execute wrapper.
    5. ``grounding_review`` -- post-reviewer grounding checkpoint adapter.
    6. ``compliance`` -- ComplianceAgent execute wrapper (AutoGen).
    7. ``grounding_compliance`` -- post-compliance grounding checkpoint
       adapter.
    8. ``decision`` -- deterministic decision matrix + escalation triggers
       (:func:`src.orchestrator_decision.decision_node`).
    9. ``escalate`` -- terminal human-review off-ramp.

    Conditional edges implement:

    * intake validation routing (proceed or escalate);
    * grounding retry loops at all 3 checkpoints (proceed / retry / escalate)
      with the retry budget loaded from ``guardrails.yaml::grounding.max_retries``;
    * post-decision routing (END or escalate).

    LangGraph :class:`RetryPolicy` is applied to every LLM-/embedding-calling
    node so that transient timeouts and connection errors are retried with
    exponential backoff before falling through to the in-node try/except.
    """

    def __init__(self) -> None:
        self.graph: StateGraph = self._build_graph()
        self._compiled = self.graph.compile()
        logger.info(
            "CreditRiskOrchestrator initialised "
            "(max_grounding_retries=%d, retry_policy=%s)",
            _MAX_GROUNDING_RETRIES,
            "enabled" if RetryPolicy is not None else "disabled",
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """Construct (but do not compile) the LangGraph StateGraph.

        Compilation happens in ``__init__`` after this returns. Splitting
        the two phases makes it easier to introspect the graph in tests
        and to add post-construction hooks (e.g. checkpointers) later.
        """
        graph: StateGraph = StateGraph(OrchestratorState)

        # ------------------------------------------------------------------
        # Build the LLM RetryPolicy from config (best-effort)
        # ------------------------------------------------------------------
        llm_retry = self._build_llm_retry_policy()

        # Helper that quietly drops the retry kwarg when RetryPolicy is
        # unavailable, so the same call site works on every LangGraph
        # version.
        def add_llm_node(name: str, func) -> None:
            if llm_retry is not None:
                graph.add_node(name, func, retry_policy=llm_retry)
            else:
                graph.add_node(name, func)

        # ------------------------------------------------------------------
        # 1. Register nodes
        # ------------------------------------------------------------------

        # No-LLM nodes (no retry needed)
        graph.add_node("intake", intake_node)

        # Agent nodes -- call LLMs, may timeout
        add_llm_node("analysis", analysis_node)
        add_llm_node("review", review_node)
        add_llm_node("compliance", compliance_node)

        # Grounding nodes -- call embedding APIs and Weaviate
        add_llm_node("grounding_analysis", grounding_analysis_node)
        add_llm_node("grounding_review", grounding_review_node)
        add_llm_node("grounding_compliance", grounding_compliance_node)

        # Pure-logic nodes (deterministic, no LLM calls)
        graph.add_node("decision", decision_node)
        graph.add_node("escalate", escalation_node)

        # ------------------------------------------------------------------
        # 2. Edges
        # ------------------------------------------------------------------

        # Entry point
        graph.add_edge(START, "intake")

        # After intake: validate or proceed
        graph.add_conditional_edges(
            "intake",
            route_after_intake,
            {
                "analysis": "analysis",
                "escalate": "escalate",
            },
        )

        # Linear: analysis -> grounding_analysis
        # (Grounding ALWAYS runs after every agent -- see GOV-03 note below.)
        graph.add_edge("analysis", "grounding_analysis")

        # After grounding_analysis: proceed, retry, or escalate
        graph.add_conditional_edges(
            "grounding_analysis",
            route_after_grounding_analysis,
            {
                "review": "review",
                "analysis": "analysis",  # retry loop
                "escalate": "escalate",
            },
        )

        # Linear: review -> grounding_review
        # NOTE (ORCH-03 / GOV-03 design choice): This is intentionally a
        # linear (unconditional) edge, NOT a conditional edge. GOV-03
        # mandates grounding verification at all 3 checkpoints
        # (post-analyst, post-reviewer, post-compliance). Grounding MUST
        # run after every agent -- it is never skipped on a "review fail"
        # short-circuit. The "review routing (pass/fail)" outcome from
        # ORCH-03 is implemented as the grounding routing (pass / retry /
        # escalate) via the conditional edges AFTER the grounding nodes
        # (route_after_grounding_review). The review/compliance pass/fail
        # outcome is resolved downstream by the deterministic decision
        # matrix (ORCH-04 apply_decision_matrix).
        graph.add_edge("review", "grounding_review")

        # After grounding_review: proceed, retry, or escalate
        graph.add_conditional_edges(
            "grounding_review",
            route_after_grounding_review,
            {
                "compliance": "compliance",
                "review": "review",  # retry loop
                "escalate": "escalate",
            },
        )

        # Linear: compliance -> grounding_compliance
        # NOTE: Same GOV-03 design rationale as review -> grounding_review
        # above. GOV-03 requires grounding after every agent; the
        # compliance pass/fail signal is resolved by the decision matrix
        # (ORCH-04), not by skipping grounding. Grounding always runs.
        graph.add_edge("compliance", "grounding_compliance")

        # After grounding_compliance: proceed, retry, or escalate
        graph.add_conditional_edges(
            "grounding_compliance",
            route_after_grounding_compliance,
            {
                "decision": "decision",
                "compliance": "compliance",  # retry loop
                "escalate": "escalate",
            },
        )

        # After decision: end or escalate
        graph.add_conditional_edges(
            "decision",
            route_after_decision,
            {
                "__end__": END,
                "escalate": "escalate",
            },
        )

        # Escalate is terminal
        graph.add_edge("escalate", END)

        return graph

    # ------------------------------------------------------------------
    # RetryPolicy construction (best-effort, version-tolerant)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_llm_retry_policy():
        """Build a LangGraph :class:`RetryPolicy` from ``config.yaml``.

        Returns ``None`` if RetryPolicy is unavailable (the import block
        at the top of this module will have already logged a warning) or
        if the config cannot be loaded for any reason -- pipelines must
        not refuse to start because of a transient config issue.
        """
        if RetryPolicy is None:
            return None

        try:
            from src.config.settings import ConfigLoader

            processing = ConfigLoader().app().processing
            attempts = int(processing.llm_retry_attempts)
            backoff = float(processing.llm_retry_backoff_seconds)
        except Exception as exc:  # pragma: no cover -- defensive
            logger.warning(
                "Could not load LLM retry config (%s); using defaults "
                "max_attempts=3, initial_interval=2.0",
                exc,
            )
            attempts = 3
            backoff = 2.0

        try:
            return RetryPolicy(
                max_attempts=attempts,
                initial_interval=backoff,
                backoff_factor=2.0,
            )
        except TypeError:
            # Older RetryPolicy signatures may not accept the same kwargs.
            try:
                return RetryPolicy(max_attempts=attempts)
            except Exception as exc:  # pragma: no cover -- defensive
                logger.warning(
                    "Could not instantiate RetryPolicy (%s); proceeding "
                    "without LangGraph-level retry.",
                    exc,
                )
                return None

    # ------------------------------------------------------------------
    # Public async entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        application: dict,
        request_id: str | None = None,
        user_role: str = "default",
        config: dict | None = None,
    ) -> dict:
        """Execute the full credit-risk pipeline for one loan application.

        This is the **sole** async entry point on the orchestrator. The
        Phase 8 API layer is expected to call ``await orch.run(app_dict)``
        and treat the returned dict as the final state.

        Parameters
        ----------
        application:
            JSON-serialisable LoanApplication dict (the same shape that
            ``LoanApplication.model_dump(mode="json")`` produces). Validation
            happens inside :func:`intake_node`; an invalid payload routes
            the request to the escalation node rather than raising.
        request_id:
            Optional caller-supplied request id. A UUID4 is generated when
            omitted so every run is uniquely traceable in logs / audit.
        user_role:
            The role of the requesting user (used downstream for RLS /
            permission decisions). Defaults to ``"default"``.
        config:
            Optional LangGraph config dict forwarded to ``ainvoke()``.
            Typically carries ``{"callbacks": [langfuse_handler]}`` for
            Langfuse tracing. Defaults to ``{}`` when omitted -- fully
            backward-compatible with existing callers.

        Returns
        -------
        dict
            The final LangGraph state dict containing (typically)
            ``final_decision``, ``confidence_score``, ``reasoning_trace``,
            ``audit_trail``, ``grounding_scores``, ``errors`` and
            ``current_stage``. On unhandled exception a partial dict with
            ``final_decision="ERROR"`` is returned -- this method **never**
            raises so that the API layer always has a dict to serialise.
        """
        request_id = request_id or str(uuid4())

        initial_state: dict = {
            "request_id": request_id,
            "application": application,
            "user_role": user_role,
            "current_stage": WorkflowStage.INTAKE.value,
            "audit_trail": [],
            "grounding_scores": [],
            "errors": [],
            "retrieved_documents": [],
            "pii_detected": False,
            "requires_escalation": False,
            "final_decision": "",
            "confidence_score": 0.0,
            "reasoning_trace": "",
        }

        logger.info(
            "Starting credit risk assessment for request %s", request_id
        )

        try:
            result = await self._compiled.ainvoke(
                initial_state, config=config or {}
            )
            logger.info(
                "Assessment complete for %s: %s (confidence: %.2f)",
                request_id,
                result.get("final_decision"),
                result.get("confidence_score", 0.0) or 0.0,
            )
            return result

        except Exception as exc:
            # Never raise out of run() -- the Phase 8 API layer depends on
            # always receiving a dict (which it can serialise to JSON and
            # return as a 5xx with structured error context).
            logger.exception(
                "CreditRiskOrchestrator.run failed for request %s: %s",
                request_id,
                exc,
            )
            return {
                "request_id": request_id,
                "current_stage": "error",
                "final_decision": "ERROR",
                "confidence_score": 0.0,
                "reasoning_trace": (
                    f"Orchestrator failed before completion: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "errors": [
                    {
                        "stage": "orchestrator",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    }
                ],
                "audit_trail": [],
                "grounding_scores": [],
                "retrieved_documents": [],
                "pii_detected": False,
                "requires_escalation": True,
            }


# ===========================================================================
# __main__ smoke test
# ===========================================================================

if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Minimal mock application -- only used to demonstrate that the
    # orchestrator can be constructed without contacting any external
    # service. We do NOT call orch.run() here because that would require
    # a live LLM provider (Azure OpenAI) and a running Weaviate instance.
    mock_app = {
        "application_id": "TEST-001",
        "applicant": {
            "company_name": "Test Corp Ltd",
            "company_number": "12345678",
            "sector": "technology",
            "years_trading": 5,
            "employee_count": 50,
            "contact_name": "Jane Smith",
            "contact_role": "Director",
        },
        "loan": {
            "amount_requested": 250000.00,
            "term_months": 36,
            "purpose": "Working capital expansion and equipment purchase",
            "security_type": "unsecured",
            "currency": "GBP",
        },
        "financials": [
            {
                "year": 2025,
                "revenue": 1500000.00,
                "gross_profit": 600000.00,
                "net_profit": 150000.00,
                "total_assets": 800000.00,
                "total_liabilities": 300000.00,
                "cash_balance": 200000.00,
            }
        ],
        "credit_score": 72,
        "ccj_count": 0,
    }

    orch = CreditRiskOrchestrator()
    print("Orchestrator compiled successfully.")
    print(f"Registered nodes: {sorted(orch.graph.nodes.keys())}")
    print("Ready to process applications via: await orch.run(application_dict)")
    print(
        "(Full pipeline execution requires LLM providers and Weaviate -- "
        "this smoke test only validates graph compilation and structure.)"
    )
