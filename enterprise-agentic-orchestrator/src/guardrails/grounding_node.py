"""
Grounding Checkpoint Node -- LangGraph Interface Stub
======================================================
Provides a reusable ``GroundingCheckpointNode`` that Phase 7 will wire
into LangGraph's state-graph at three checkpoint positions:

* **post_analyst** -- verify analyst output against retrieved documents
* **post_reviewer** -- verify reviewer output against analyst evidence
* **post_compliance** -- verify compliance output against policy sources

This module does **not** depend on LangGraph.  It exposes a callable
``async __call__(state) -> state`` contract that LangGraph nodes accept
natively.  The full wiring is deferred to Phase 7 (orchestration layer).

See GOV-03 in ROADMAP.md.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# The three checkpoint positions required by GOV-03
_CHECKPOINT_POSITIONS: list[str] = [
    "post_analyst",
    "post_reviewer",
    "post_compliance",
]


class GroundingCheckpointNode:
    """LangGraph-compatible grounding checkpoint node.

    Wraps :class:`~src.guardrails.grounding.GroundingChecker` as an
    ``async __call__(state: dict) -> dict`` node that can be inserted
    directly into a LangGraph ``StateGraph``.

    Parameters
    ----------
    checker:
        An existing :class:`GroundingChecker` instance.  When *None*,
        a default instance is created on first use (lazy).
    checkpoint_name:
        Label for this checkpoint position (e.g. ``"post_analyst"``).
        Used as the key in ``state["grounding_results"]``.
    """

    def __init__(
        self,
        checker=None,
        checkpoint_name: str = "grounding_check",
    ) -> None:
        self._checker = checker
        self.checkpoint_name = checkpoint_name

    @property
    def checker(self):
        """Lazily create GroundingChecker on first access."""
        if self._checker is None:
            from src.guardrails.grounding import GroundingChecker

            self._checker = GroundingChecker()
        return self._checker

    async def __call__(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute grounding verification on the current pipeline state.

        Expects the following keys in *state*:

        * ``agent_output`` (str) -- the text to verify.
        * ``source_documents`` (list) -- the retrieved source documents.

        Adds / updates:

        * ``state["grounding_results"][checkpoint_name]`` -- the
          :class:`~src.models.governance.GroundingResult` (as dict).
        * ``state["needs_reprompt"]`` -- set to ``True`` when the output
          fails grounding.

        Parameters
        ----------
        state:
            Mutable pipeline state dict.

        Returns
        -------
        dict
            Updated state dict.
        """
        agent_output: str = state.get("agent_output", "")
        source_documents: list = state.get("source_documents", [])

        logger.info(
            "Grounding checkpoint '%s' -- verifying agent output (%d chars) "
            "against %d source documents",
            self.checkpoint_name,
            len(agent_output),
            len(source_documents),
        )

        result = self.checker.verify(agent_output, source_documents)

        # Store result in state under grounding_results
        if "grounding_results" not in state:
            state["grounding_results"] = {}
        state["grounding_results"][self.checkpoint_name] = result.model_dump(mode="json")

        # Flag for re-prompting if not grounded
        if not result.is_grounded:
            state["needs_reprompt"] = True
            logger.warning(
                "Checkpoint '%s' failed grounding -- needs_reprompt=True",
                self.checkpoint_name,
            )
        else:
            logger.info(
                "Checkpoint '%s' passed grounding (score: %.2f)",
                self.checkpoint_name,
                result.grounding_score,
            )

        return state

    @staticmethod
    def get_checkpoint_positions() -> list[str]:
        """Return the three GOV-03 checkpoint position names.

        These correspond to the positions where grounding verification
        must occur in the LangGraph pipeline:

        1. ``post_analyst`` -- after the analyst agent produces output
        2. ``post_reviewer`` -- after the reviewer agent produces output
        3. ``post_compliance`` -- after the compliance agent produces output
        """
        return list(_CHECKPOINT_POSITIONS)


def create_grounding_checkpoints() -> dict[str, GroundingCheckpointNode]:
    """Factory: create one ``GroundingCheckpointNode`` per GOV-03 position.

    All nodes share the same underlying ``GroundingChecker`` instance
    (created lazily on first use) to avoid redundant config loading.

    Returns
    -------
    dict[str, GroundingCheckpointNode]
        Mapping from checkpoint position name to node instance.
    """
    from src.guardrails.grounding import GroundingChecker

    checker = GroundingChecker()
    nodes: dict[str, GroundingCheckpointNode] = {}

    for position in _CHECKPOINT_POSITIONS:
        nodes[position] = GroundingCheckpointNode(
            checker=checker,
            checkpoint_name=position,
        )

    logger.info(
        "Created %d grounding checkpoint nodes: %s",
        len(nodes),
        list(nodes.keys()),
    )
    return nodes
