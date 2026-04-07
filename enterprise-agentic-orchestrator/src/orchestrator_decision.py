"""Deterministic decision matrix, escalation rules and LangGraph node functions.

This module implements ORCH-04 (deterministic decision matrix) and ORCH-05
(escalation trigger evaluation) as pure business logic, plus two LangGraph
nodes (``decision_node``, ``escalation_node``) that synthesise all upstream
agent outputs into a final lending Decision and route cases to human review.

Design principles
-----------------
* **Pure sync functions** for matrix and trigger checks -- no side effects,
  trivially unit-testable.
* **Async nodes** for LangGraph compatibility; return *partial dict deltas*
  only, never mutate the incoming state.
* **Lazy imports** for every heavy dependency (Pydantic models, ConfigLoader,
  AuditTrail) inside function bodies to keep module import cost ~zero and
  avoid circular import chains during package initialisation.
* **Config-driven escalation** -- trigger *names* come from
  ``config/guardrails.yaml`` via ``ConfigLoader().guardrails()``; each
  name is mapped to an evaluator and wrapped in try/except so a missing
  or malformed state field skips that trigger rather than crashing the
  pipeline.
* **Graceful degradation** -- the outer ``decision_node`` and
  ``escalation_node`` functions are wrapped in try/except that emits an
  error delta rather than propagating exceptions out of a LangGraph node.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ORCH-04: Deterministic decision matrix
# ---------------------------------------------------------------------------

def apply_decision_matrix(
    analyst_recommendation: str,
    reviewer_agrees: bool,
    compliance_passed: bool,
) -> str:
    """Apply the deterministic 9-row decision matrix.

    Maps the combination of analyst recommendation, reviewer agreement and
    compliance status to a final :class:`DecisionOutcome` value. The matrix
    is fully deterministic -- no LLM calls, no randomness -- which makes it
    regulator-defensible and unit-testable.

    Decision matrix (from SPECIFICATION, ORCH-04):

    ======================  ===============  =================  =======================
    Analyst Recommendation  Reviewer Agrees  Compliance Passed  Decision
    ======================  ===============  =================  =======================
    APPROVE                 True             True               APPROVED
    APPROVE                 True             False              REFERRED_TO_UNDERWRITER
    APPROVE                 False            True               REFERRED_TO_UNDERWRITER
    APPROVE                 False            False              REJECTED
    REJECT                  True             True               REJECTED
    REJECT                  True             False              REJECTED
    REJECT                  False            True               REFERRED_TO_UNDERWRITER
    REJECT                  False            False              REJECTED
    REFER_TO_UNDERWRITER    *                *                  REFERRED_TO_UNDERWRITER
    ======================  ===============  =================  =======================

    Parameters
    ----------
    analyst_recommendation:
        One of ``"APPROVE"``, ``"REJECT"``, ``"REFER_TO_UNDERWRITER"``.
        Case-insensitive -- normalised to upper before comparison.
    reviewer_agrees:
        Whether the reviewer agent agrees with the analyst's recommendation.
    compliance_passed:
        Whether the compliance agent's overall assessment passed.

    Returns
    -------
    str
        The string value of a :class:`DecisionOutcome` enum member
        (``"APPROVED"``, ``"REJECTED"`` or ``"REFERRED_TO_UNDERWRITER"``).

    Notes
    -----
    If ``analyst_recommendation`` is not a valid
    :class:`~src.models.reports.Recommendation` value, the function returns
    ``"REFERRED_TO_UNDERWRITER"`` as a safe default and logs a warning. This
    is a deliberate "fail safe" (not "fail open") -- unknown inputs route to
    a human underwriter rather than silently approving or rejecting.
    """
    # Lazy import -- keeps module import cost near-zero.
    from src.models.reports import DecisionOutcome, Recommendation

    # Normalise analyst recommendation to upper case for case-insensitive match.
    try:
        normalised = str(analyst_recommendation).strip().upper()
    except Exception:  # pragma: no cover -- defensive
        normalised = ""

    # Validate against the Recommendation enum. Unknown values route to
    # human review as a safe default.
    try:
        recommendation = Recommendation(normalised)
    except ValueError:
        logger.warning(
            "apply_decision_matrix: unknown analyst_recommendation=%r, "
            "routing to REFERRED_TO_UNDERWRITER as safe default",
            analyst_recommendation,
        )
        return DecisionOutcome.REFERRED_TO_UNDERWRITER.value

    # REFER_TO_UNDERWRITER is a wildcard: regardless of reviewer or compliance
    # state, this routes to human underwriter review.
    if recommendation is Recommendation.REFER_TO_UNDERWRITER:
        return DecisionOutcome.REFERRED_TO_UNDERWRITER.value

    # APPROVE branch -- 4 rows of the matrix.
    if recommendation is Recommendation.APPROVE:
        if reviewer_agrees and compliance_passed:
            return DecisionOutcome.APPROVED.value
        if reviewer_agrees and not compliance_passed:
            return DecisionOutcome.REFERRED_TO_UNDERWRITER.value
        if (not reviewer_agrees) and compliance_passed:
            return DecisionOutcome.REFERRED_TO_UNDERWRITER.value
        # not reviewer_agrees and not compliance_passed
        return DecisionOutcome.REJECTED.value

    # REJECT branch -- 4 rows of the matrix.
    # recommendation is Recommendation.REJECT at this point.
    if reviewer_agrees and compliance_passed:
        return DecisionOutcome.REJECTED.value
    if reviewer_agrees and not compliance_passed:
        return DecisionOutcome.REJECTED.value
    if (not reviewer_agrees) and compliance_passed:
        return DecisionOutcome.REFERRED_TO_UNDERWRITER.value
    # not reviewer_agrees and not compliance_passed
    return DecisionOutcome.REJECTED.value


# ---------------------------------------------------------------------------
# ORCH-05: Escalation trigger evaluation
# ---------------------------------------------------------------------------

# Reviewer confidence levels -> numeric scores used by the
# low_reviewer_confidence trigger. Kept at module level so tests can
# monkey-patch the mapping if the business rule changes.
_CONFIDENCE_LEVEL_SCORES: dict[str, float] = {
    "HIGH": 0.9,
    "MEDIUM": 0.7,
    "LOW": 0.3,
}


def _score_from_entry(entry: Any) -> float | None:
    """Extract a grounding score from a grounding_scores list entry.

    ``grounding_scores`` is accumulated from multiple producers (see
    ``OrchestratorState``); the exact field name has drifted between
    "score" and "grounding_score" in different iterations. This helper
    accepts both so the escalation logic never crashes on schema drift.
    """
    if not isinstance(entry, dict):
        return None
    for key in ("score", "grounding_score"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _entry_is_grounded(entry: Any) -> bool | None:
    """Return the is_grounded flag from a grounding_scores entry, if present."""
    if not isinstance(entry, dict):
        return None
    value = entry.get("is_grounded")
    if isinstance(value, bool):
        return value
    return None


def check_escalation_triggers(state: dict) -> list[str]:
    """Evaluate every escalation trigger from ``guardrails.yaml``.

    Walks the ``escalation.triggers`` list defined in ``guardrails.yaml``
    and runs the matching evaluator for each trigger *name*. The set of
    triggers checked is therefore **config-driven** -- adding a new entry
    to the YAML (plus an evaluator here) is all that's needed to extend
    the escalation logic.

    Each individual trigger check is wrapped in its own try/except so a
    missing or malformed state field skips that one trigger rather than
    crashing the entire evaluation.

    Parameters
    ----------
    state:
        The full LangGraph state dict (``OrchestratorState``). None of the
        fields are assumed to exist; every access is defensive.

    Returns
    -------
    list[str]
        Human-readable escalation reasons (empty list when no triggers
        fired). Each entry has the form
        ``"<trigger_name>: <explanatory message>"`` and is intended for
        inclusion in the audit trail and the final reasoning string.
    """
    from src.config.settings import ConfigLoader

    # Load the list of triggers from config. If config loading fails we
    # log and fall back to an empty list -- pipelines must not crash
    # because of a misconfigured guardrails file at runtime.
    try:
        guardrails = ConfigLoader().guardrails()
        configured_triggers = list(guardrails.escalation.triggers)
    except Exception as exc:  # pragma: no cover -- defensive
        logger.error(
            "check_escalation_triggers: failed to load guardrails config: %s",
            exc,
        )
        return []

    # Also load the grounding max_retries threshold up-front (once) so
    # the grounding_failure evaluator can reference it without re-reading
    # config for every trigger.
    try:
        max_retries = int(guardrails.grounding.max_retries)
    except Exception:  # pragma: no cover -- defensive
        max_retries = 2

    reasons: list[str] = []

    # ------------------------------------------------------------------
    # high_value_loan -- loan amount > 500_000
    # ------------------------------------------------------------------
    def _check_high_value_loan() -> str | None:
        try:
            amount_raw = (
                state["application"]["loan"]["amount_requested"]
            )
            amount = float(amount_raw)
            if amount > 500_000:
                return (
                    f"high_value_loan: Loan amount {amount:.0f} "
                    f"exceeds 500000 threshold"
                )
        except (KeyError, TypeError, ValueError):
            return None
        return None

    # ------------------------------------------------------------------
    # deteriorating_sector -- analysis_result.sector_outlook contains
    # "deteriorating" (case-insensitive substring match).
    # ------------------------------------------------------------------
    def _check_deteriorating_sector() -> str | None:
        try:
            outlook = (
                state.get("analysis_result", {}).get("sector_outlook", "")
            )
            if "deteriorating" in str(outlook).lower():
                return (
                    f"deteriorating_sector: Sector outlook '{outlook}' "
                    f"flagged as deteriorating"
                )
        except (AttributeError, TypeError):
            return None
        return None

    # ------------------------------------------------------------------
    # compliance_failure -- compliance_result.overall_passed is False
    # ------------------------------------------------------------------
    def _check_compliance_failure() -> str | None:
        try:
            passed = (
                state.get("compliance_result", {}).get("overall_passed")
            )
            if passed is False:
                return "compliance_failure: Compliance check failed"
        except (AttributeError, TypeError):
            return None
        return None

    # ------------------------------------------------------------------
    # low_reviewer_confidence -- reviewer confidence_level maps to < 0.5
    # ------------------------------------------------------------------
    def _check_low_reviewer_confidence() -> str | None:
        try:
            level = (
                state.get("review_result", {}).get("confidence_level", "HIGH")
            )
            level_key = str(level).upper()
            numeric = _CONFIDENCE_LEVEL_SCORES.get(level_key, 0.9)
            if numeric < 0.5:
                return (
                    f"low_reviewer_confidence: Reviewer confidence_level="
                    f"{level_key} (numeric={numeric:.1f}) below 0.5 threshold"
                )
        except (AttributeError, TypeError):
            return None
        return None

    # ------------------------------------------------------------------
    # grounding_failure -- any entry with is_grounded=False where the
    # total number of grounding attempts for that checkpoint has reached
    # max_retries from config.
    # ------------------------------------------------------------------
    def _check_grounding_failure() -> str | None:
        try:
            scores = state.get("grounding_scores", []) or []
            if not isinstance(scores, list):
                return None

            # Bucket entries by checkpoint name so we can count retries
            # per checkpoint. Falls back to a single unnamed bucket if
            # no checkpoint key is present.
            buckets: dict[str, list[dict]] = {}
            for entry in scores:
                if not isinstance(entry, dict):
                    continue
                key = (
                    entry.get("checkpoint")
                    or entry.get("checkpoint_name")
                    or entry.get("stage")
                    or "_default"
                )
                buckets.setdefault(str(key), []).append(entry)

            for checkpoint_name, entries in buckets.items():
                # Did the most recent attempt in this bucket fail
                # grounding, and have we exhausted retries?
                latest = entries[-1]
                grounded = _entry_is_grounded(latest)
                attempts = len(entries)
                if grounded is False and attempts >= max_retries:
                    return (
                        f"grounding_failure: Checkpoint '{checkpoint_name}' "
                        f"failed grounding after {attempts} attempts "
                        f"(max_retries={max_retries})"
                    )
        except (AttributeError, TypeError, IndexError):
            return None
        return None

    # ------------------------------------------------------------------
    # low_average_grounding -- mean of all score values < 0.75
    # ------------------------------------------------------------------
    def _check_low_average_grounding() -> str | None:
        try:
            scores = state.get("grounding_scores", []) or []
            if not isinstance(scores, list) or not scores:
                return None
            numeric_scores = [
                s for s in (_score_from_entry(e) for e in scores) if s is not None
            ]
            if not numeric_scores:
                return None
            average = sum(numeric_scores) / len(numeric_scores)
            if average < 0.75:
                return (
                    f"low_average_grounding: Average grounding score "
                    f"{average:.2f} below 0.75 threshold "
                    f"(n={len(numeric_scores)})"
                )
        except (AttributeError, TypeError, ZeroDivisionError):
            return None
        return None

    # Map trigger names (as they appear in guardrails.yaml) to their
    # evaluator functions. Unknown trigger names are logged and skipped.
    evaluators = {
        "high_value_loan": _check_high_value_loan,
        "deteriorating_sector": _check_deteriorating_sector,
        "compliance_failure": _check_compliance_failure,
        "low_reviewer_confidence": _check_low_reviewer_confidence,
        "grounding_failure": _check_grounding_failure,
        "low_average_grounding": _check_low_average_grounding,
    }

    for trigger in configured_triggers:
        name = getattr(trigger, "name", None)
        if not name:
            continue
        evaluator = evaluators.get(name)
        if evaluator is None:
            logger.warning(
                "check_escalation_triggers: no evaluator registered for "
                "trigger '%s' (defined in guardrails.yaml)",
                name,
            )
            continue
        try:
            message = evaluator()
        except Exception as exc:  # pragma: no cover -- defensive outer net
            logger.warning(
                "check_escalation_triggers: evaluator for '%s' raised %s",
                name,
                exc,
            )
            continue
        if message:
            logger.warning("Escalation trigger fired: %s", message)
            reasons.append(message)

    return reasons


# ---------------------------------------------------------------------------
# Internal helpers for node functions
# ---------------------------------------------------------------------------

def _clamp_unit(value: float) -> float:
    """Clamp *value* to the closed interval [0.0, 1.0]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _compute_confidence(
    analysis_result: dict,
    review_result: dict,
    compliance_result: dict,
    grounding_scores: list,
) -> float:
    """Compute the final decision confidence as an average of four signals.

    The four signals -- analyst credit score (normalised to [0,1]), reviewer
    quality score, compliance pass indicator and average grounding score --
    are averaged with equal weight and clamped to [0.0, 1.0]. Missing or
    malformed signals fall back to a neutral value of 0.5 so one bad field
    cannot drive the confidence to zero.
    """
    signals: list[float] = []

    # Analyst credit score (0-100 on the Pydantic model) -> [0,1]
    try:
        credit_score = float(analysis_result.get("credit_score", 50))
        signals.append(_clamp_unit(credit_score / 100.0))
    except (TypeError, ValueError):
        signals.append(0.5)

    # Reviewer quality score (already 0-1)
    try:
        quality = float(review_result.get("quality_score", 0.5))
        signals.append(_clamp_unit(quality))
    except (TypeError, ValueError):
        signals.append(0.5)

    # Compliance pass -> 1.0 / 0.0
    try:
        passed = bool(compliance_result.get("overall_passed", False))
        signals.append(1.0 if passed else 0.0)
    except (TypeError, ValueError):
        signals.append(0.0)

    # Average grounding score across all checkpoint entries
    try:
        if isinstance(grounding_scores, list) and grounding_scores:
            numeric = [
                s for s in (_score_from_entry(e) for e in grounding_scores)
                if s is not None
            ]
            if numeric:
                signals.append(_clamp_unit(sum(numeric) / len(numeric)))
            else:
                signals.append(0.5)
        else:
            signals.append(0.5)
    except (TypeError, ZeroDivisionError):
        signals.append(0.5)

    if not signals:
        return 0.0
    return _clamp_unit(sum(signals) / len(signals))


# ---------------------------------------------------------------------------
# LangGraph node: decision_node
# ---------------------------------------------------------------------------

async def decision_node(state: dict) -> dict:
    """Synthesise all agent outputs into the final lending Decision.

    This node is the deterministic end-of-pipeline synthesiser. It:

    1. Pulls the analyst, reviewer and compliance results from state.
    2. Runs the outputs through :func:`apply_decision_matrix` to obtain
       an initial outcome.
    3. Evaluates all escalation triggers (ORCH-05). If any fire, the
       outcome is overridden to ``"ESCALATED"`` and the
       ``requires_escalation`` flag is set.
    4. Computes a bounded confidence score averaged across four signals
       (analyst credit score, reviewer quality score, compliance pass
       and average grounding score).
    5. Builds a human-readable reasoning trace suitable for the audit
       trail and for regulator review.
    6. Creates a :class:`~src.models.reports.Decision` Pydantic model
       to validate the outcome and confidence against their enum / range
       constraints.
    7. Emits two or three audit entries (matrix applied, decision
       rendered and -- if escalated -- escalation triggered).

    The function returns a **partial dict delta** and never mutates the
    incoming state. Any exception inside the body is caught and turned
    into an error delta so a downstream LangGraph node can route to the
    escalation path instead of the workflow blowing up mid-run.

    Parameters
    ----------
    state:
        The current LangGraph state dict.

    Returns
    -------
    dict
        Partial state delta with ``final_decision``, ``confidence_score``,
        ``reasoning_trace``, ``requires_escalation``, ``current_stage``
        and ``audit_trail`` (list, consumed by the reducer in state.py).
    """
    # Lazy imports to keep module import cost minimal and avoid
    # circular dependencies during package init.
    from src.models.reports import Decision, DecisionOutcome
    from src.state import WorkflowStage

    try:
        # --- 1. Extract agent outputs ------------------------------------
        analysis_result: dict = state.get("analysis_result") or {}
        review_result: dict = state.get("review_result") or {}
        compliance_result: dict = state.get("compliance_result") or {}
        grounding_scores: list = state.get("grounding_scores") or []

        # --- 2. Matrix inputs --------------------------------------------
        analyst_recommendation = analysis_result.get(
            "recommendation", "REFER_TO_UNDERWRITER"
        )
        reviewer_agrees = bool(review_result.get("agrees_with_analyst", False))
        compliance_passed = bool(
            compliance_result.get("overall_passed", False)
        )

        # --- 3. Apply the deterministic decision matrix ------------------
        outcome = apply_decision_matrix(
            analyst_recommendation=analyst_recommendation,
            reviewer_agrees=reviewer_agrees,
            compliance_passed=compliance_passed,
        )

        # --- 4. Escalation override --------------------------------------
        triggers = check_escalation_triggers(state)
        requires_escalation = bool(triggers)
        if requires_escalation:
            outcome = DecisionOutcome.ESCALATED.value

        # --- 5. Confidence -----------------------------------------------
        confidence = _compute_confidence(
            analysis_result=analysis_result,
            review_result=review_result,
            compliance_result=compliance_result,
            grounding_scores=grounding_scores,
        )

        # --- 6. Reasoning trace ------------------------------------------
        if triggers:
            trigger_fragment = "Escalation triggers: " + ", ".join(triggers)
        else:
            trigger_fragment = "No escalation triggers."
        reasoning_trace = (
            f"Analyst recommendation: {analyst_recommendation}. "
            f"Reviewer agrees: {reviewer_agrees}. "
            f"Compliance passed: {compliance_passed}. "
            f"Decision: {outcome}. "
            f"Confidence: {confidence:.2f}. "
            f"{trigger_fragment}"
        )

        # --- 7. Create validated Decision model --------------------------
        application_id = (
            state.get("application", {}).get("application_id", "unknown")
        )
        decision = Decision(
            application_id=application_id,
            outcome=DecisionOutcome(outcome),
            reasoning=reasoning_trace,
            confidence_score=confidence,
            conditions=list(triggers) if triggers else [],
        )

        # --- 8. Build audit entries --------------------------------------
        audit_entries: list[dict] = []

        audit_entries.append(
            {
                "stage": WorkflowStage.DECISION.value,
                "action": "matrix_applied",
                "details": {
                    "recommendation": analyst_recommendation,
                    "reviewer_agrees": reviewer_agrees,
                    "compliance_passed": compliance_passed,
                    "outcome": outcome,
                },
            }
        )
        audit_entries.append(
            {
                "stage": WorkflowStage.DECISION.value,
                "action": "decision_rendered",
                "details": {
                    "decision_id": decision.decision_id,
                    "application_id": application_id,
                    "outcome": outcome,
                    "confidence": confidence,
                },
            }
        )
        if requires_escalation:
            audit_entries.append(
                {
                    "stage": WorkflowStage.DECISION.value,
                    "action": "escalation_triggered",
                    "details": {"triggers": list(triggers)},
                }
            )

        logger.info(
            "decision_node: application_id=%s outcome=%s confidence=%.2f "
            "triggers=%d",
            application_id,
            outcome,
            confidence,
            len(triggers),
        )

        # --- 9. Return partial delta -------------------------------------
        next_stage = (
            WorkflowStage.ESCALATE.value
            if requires_escalation
            else WorkflowStage.DECISION.value
        )
        return {
            "final_decision": outcome,
            "confidence_score": confidence,
            "reasoning_trace": reasoning_trace,
            "requires_escalation": requires_escalation,
            "current_stage": next_stage,
            "audit_trail": audit_entries,
        }

    except Exception as exc:
        logger.exception("decision_node: unhandled error: %s", exc)
        return {
            "final_decision": "ERROR",
            "requires_escalation": True,
            "current_stage": WorkflowStage.ESCALATE.value,
            "errors": [f"decision_node: {type(exc).__name__}: {exc}"],
            "audit_trail": [
                {
                    "stage": WorkflowStage.DECISION.value,
                    "action": "decision_node_error",
                    "details": {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                }
            ],
        }


# ---------------------------------------------------------------------------
# LangGraph node: escalation_node
# ---------------------------------------------------------------------------

async def escalation_node(state: dict) -> dict:
    """Terminal node that routes a request to human review.

    This node is invoked when ``decision_node`` (or any upstream node)
    flags the request for escalation. It collects every available
    escalation reason -- the pre-computed ``requires_escalation`` flag,
    any accumulated errors and a fresh re-run of
    :func:`check_escalation_triggers` as a belt-and-braces safety net --
    assembles a human-readable summary and emits a single audit entry.

    Like ``decision_node``, this function returns a partial dict delta
    and never mutates the incoming state. It is the terminal node of the
    escalation path in the LangGraph, so its return value is the final
    state observed by the caller.

    Parameters
    ----------
    state:
        The current LangGraph state dict at the point of escalation.

    Returns
    -------
    dict
        Partial delta setting ``current_stage=ESCALATE``,
        ``requires_escalation=True``, ``final_decision="ESCALATED"``,
        a reasoning_trace summary and one audit entry.
    """
    from src.state import WorkflowStage

    try:
        reasons: list[str] = []

        # Flag from an upstream node
        if state.get("requires_escalation"):
            reasons.append("upstream_flag: requires_escalation=True")

        # Any errors accumulated in the pipeline
        errors = state.get("errors") or []
        if isinstance(errors, list):
            for err in errors:
                if err:
                    reasons.append(f"error: {err}")

        # Re-run triggers as a safety net -- cheap and ensures we don't
        # lose context if decision_node didn't run (e.g. direct routing).
        try:
            trigger_reasons = check_escalation_triggers(state)
        except Exception:  # pragma: no cover -- defensive
            trigger_reasons = []
        for tr in trigger_reasons:
            if tr not in reasons:
                reasons.append(tr)

        if not reasons:
            reasons.append("unspecified: escalation node entered without reasons")

        partial_decision = state.get("final_decision")
        reasoning_trace = (
            f"Escalated to human review. Reasons: {'; '.join(reasons)}"
        )

        audit_entry = {
            "stage": WorkflowStage.ESCALATE.value,
            "action": "human_review_required",
            "details": {
                "reasons": list(reasons),
                "partial_decision": partial_decision,
                "application_id": state.get("application", {}).get(
                    "application_id", "unknown"
                ),
            },
        }

        logger.warning(
            "escalation_node: routing to human review (%d reasons)",
            len(reasons),
        )

        return {
            "current_stage": WorkflowStage.ESCALATE.value,
            "requires_escalation": True,
            "final_decision": "ESCALATED",
            "reasoning_trace": reasoning_trace,
            "audit_trail": [audit_entry],
        }

    except Exception as exc:
        logger.exception("escalation_node: unhandled error: %s", exc)
        return {
            "current_stage": WorkflowStage.ESCALATE.value,
            "requires_escalation": True,
            "final_decision": "ESCALATED",
            "reasoning_trace": (
                f"Escalated to human review after escalation_node error: "
                f"{type(exc).__name__}: {exc}"
            ),
            "errors": [f"escalation_node: {type(exc).__name__}: {exc}"],
            "audit_trail": [
                {
                    "stage": WorkflowStage.ESCALATE.value,
                    "action": "escalation_node_error",
                    "details": {
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                }
            ],
        }
