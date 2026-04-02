"""
OrchestratorState -- LangGraph-compatible TypedDict with Annotated reducers.

This module defines the central state schema for the credit-risk orchestration
pipeline.  Every LangGraph node receives the full state and returns a *partial*
dict (delta) containing only the fields it wants to update.

Reducer semantics
-----------------
* **Annotated[list, operator.add]** -- the returned list is *appended* to the
  existing list (accumulating).  Nodes MUST return flat ``list`` values for
  these fields (not nested lists).

  Fields with reducers: ``audit_trail``, ``grounding_scores``, ``errors``,
  ``retrieved_documents``.

* **No reducer (bare type)** -- last-write-wins.  The value returned by the
  most-recent node simply replaces whatever was there before.

Pydantic model storage
----------------------
Agent output fields (``analysis_result``, ``review_result``,
``compliance_result``) and ``application`` store plain ``dict`` instances
produced by ``model.model_dump(mode="json")``.  This keeps the state fully
JSON-serialisable and avoids importing Pydantic models into the state module.
"""

from __future__ import annotations

from enum import Enum
from operator import add
from typing import Annotated

from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Workflow stages
# ---------------------------------------------------------------------------

class WorkflowStage(str, Enum):
    """Pipeline stages including grounding-verification passes.

    The three ``GROUNDING_*`` stages were added for Phase 7 (grounding
    verification).  They run immediately after their respective agent
    stages and validate factual claims against retrieved source data.
    """

    INTAKE = "intake"
    ANALYSIS = "analysis"
    GROUNDING_ANALYSIS = "grounding_analysis"
    REVIEW = "review"
    GROUNDING_REVIEW = "grounding_review"
    COMPLIANCE = "compliance"
    GROUNDING_COMPLIANCE = "grounding_compliance"
    DECISION = "decision"
    COMPLETE = "complete"
    ESCALATE = "escalate"


# ---------------------------------------------------------------------------
# Orchestrator state (LangGraph TypedDict)
# ---------------------------------------------------------------------------

class OrchestratorState(TypedDict, total=False):
    """Central state passed through the LangGraph state machine.

    ``total=False`` means every field is optional -- nodes return only
    the keys they want to update (partial dicts / deltas).

    See module docstring for reducer and storage conventions.
    """

    # -- Input fields (set once at intake, last-write-wins) ----------------
    request_id: str
    application: dict  # Serialised LoanApplication via model_dump(mode="json")
    user_role: str

    # -- Agent outputs (set once by respective agent, last-write-wins) -----
    analysis_result: dict   # Serialised AnalysisReport
    review_result: dict     # Serialised ReviewReport
    compliance_result: dict  # Serialised ComplianceReport

    # -- Accumulating fields (Annotated reducers -- see module docstring) --
    audit_trail: Annotated[list, add]
    grounding_scores: Annotated[list, add]
    errors: Annotated[list, add]
    retrieved_documents: Annotated[list, add]

    # -- Governance flags (last-write-wins) --------------------------------
    pii_detected: bool
    requires_escalation: bool

    # -- Decision (last-write-wins) ----------------------------------------
    final_decision: str
    confidence_score: float
    reasoning_trace: str

    # -- Control flow (last-write-wins) ------------------------------------
    current_stage: str
