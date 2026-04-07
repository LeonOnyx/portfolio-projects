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
