"""Immutable audit trail with SHA-256 hash chain for regulatory compliance.

Each audit entry is linked to the previous via a SHA-256 hash computed from
the previous entry's hash and the current entry's immutable fields. This
creates a tamper-evident chain: modifying any entry breaks verification
for all subsequent entries.

Design decisions:
    - GENESIS as initial previous_hash (well-known starting point)
    - All immutable fields (entry_id, timestamp, stage, action) AND details
      dict are included in hash computation
    - details dict IS included in hash via deterministic
      json.dumps(sort_keys=True, default=str)
    - sort_keys=True in json.dumps for deterministic hash computation
    - In-memory list storage; export_json() for regulatory submission
    - request_id managed at AuditTrail level, not per-entry
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from src.models.governance import AuditEntry

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of text content.

    Module-level utility for hashing input/output text to create
    content fingerprints for audit entries.

    Args:
        text: The text content to hash.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    return hashlib.sha256(text.encode()).hexdigest()


class AuditTrail:
    """Immutable audit trail with SHA-256 hash chain.

    Records 30-50 entries per lending request covering the full lifecycle
    from input_received to audit_trail_closed. Each entry is hash-chained
    to the previous, creating a tamper-evident record suitable for
    FCA/PRA regulatory submission and 7-year retention.

    Attributes:
        request_id: Unique identifier for the lending request.
        entries: Ordered list of audit entries in the chain.
    """

    def __init__(self, request_id: str) -> None:
        """Initialise an audit trail for a lending request.

        Args:
            request_id: Unique identifier for the request being audited.
        """
        self.request_id = request_id
        self.entries: list[AuditEntry] = []
        self._previous_hash: str = "GENESIS"

    def add_entry(
        self,
        stage: str,
        action: str,
        agent: Optional[str] = None,
        details: Optional[dict] = None,
        duration_ms: Optional[float] = None,
        token_count: Optional[int] = None,
        input_hash: Optional[str] = None,
        output_hash: Optional[str] = None,
    ) -> AuditEntry:
        """Add a new entry to the audit trail with hash chain linking.

        Creates an AuditEntry, computes its SHA-256 hash from the previous
        hash and immutable fields, and appends it to the chain.

        Args:
            stage: Pipeline stage (e.g. INTAKE, ANALYSIS, REVIEW).
            action: Specific action taken (e.g. input_received, analyst_started).
            agent: Name of the agent performing the action, if applicable.
            details: Additional context dict (included in hash via deterministic serialisation).
            duration_ms: Execution duration in milliseconds, if applicable.
            token_count: LLM token usage, if applicable.
            input_hash: SHA-256 hash of input content, if applicable.
            output_hash: SHA-256 hash of output content, if applicable.

        Returns:
            The created AuditEntry with hash set.
        """
        entry_details = details.copy() if details else {}

        if input_hash is not None:
            entry_details["input_hash"] = input_hash
        if output_hash is not None:
            entry_details["output_hash"] = output_hash

        entry = AuditEntry(
            stage=stage,
            agent=agent,
            action=action,
            details=entry_details,
            duration_ms=duration_ms,
            token_count=token_count,
        )

        # Compute hash from previous_hash + immutable fields + details.
        # details is pre-serialised with json.dumps(sort_keys=True, default=str)
        # to ensure deterministic string representation regardless of dict
        # insertion order or Decimal/datetime types.
        content = json.dumps(
            {
                "previous_hash": self._previous_hash,
                "entry_id": entry.entry_id,
                "timestamp": entry.timestamp.isoformat(),
                "stage": entry.stage,
                "action": entry.action,
                "details": json.dumps(entry_details, sort_keys=True, default=str),
            },
            sort_keys=True,
        )
        entry.hash = hashlib.sha256(content.encode()).hexdigest()

        self._previous_hash = entry.hash
        self.entries.append(entry)

        logger.debug(
            "Audit entry added: stage=%s action=%s hash=%s",
            stage,
            action,
            entry.hash[:12],
        )

        return entry

    def verify_chain(self) -> bool:
        """Verify hash chain integrity by recomputing all hashes.

        Walks the entire chain from GENESIS, recomputing each entry's
        expected hash from the previous hash and immutable fields. If
        any stored hash does not match the recomputed value, the chain
        has been tampered with.

        Returns:
            True if the entire chain is valid, False if any entry
            has been modified.
        """
        previous_hash = "GENESIS"

        for i, entry in enumerate(self.entries):
            content = json.dumps(
                {
                    "previous_hash": previous_hash,
                    "entry_id": entry.entry_id,
                    "timestamp": entry.timestamp.isoformat(),
                    "stage": entry.stage,
                    "action": entry.action,
                    "details": json.dumps(entry.details or {}, sort_keys=True, default=str),
                },
                sort_keys=True,
            )
            expected = hashlib.sha256(content.encode()).hexdigest()

            if entry.hash != expected:
                logger.warning(
                    "Hash chain broken at entry %d (stage=%s, action=%s): "
                    "stored=%s expected=%s",
                    i,
                    entry.stage,
                    entry.action,
                    entry.hash[:12],
                    expected[:12],
                )
                return False

            previous_hash = entry.hash

        return True

    def export_json(self) -> str:
        """Export the full audit trail as JSON for regulatory submission.

        Produces a JSON string containing the request_id and all entries
        with their fields serialised. Suitable for 7-year retention and
        FCA/PRA submission.

        Returns:
            JSON string with request metadata and all audit entries.
        """
        payload = {
            "request_id": self.request_id,
            "entry_count": len(self.entries),
            "chain_valid": self.verify_chain(),
            "entries": [
                entry.model_dump(mode="json") for entry in self.entries
            ],
        }
        return json.dumps(payload, indent=2, default=str)

    def __len__(self) -> int:
        """Return the number of entries in the audit trail."""
        return len(self.entries)

    def get_entries_by_stage(self, stage: str) -> list[AuditEntry]:
        """Filter audit entries by pipeline stage.

        Args:
            stage: The pipeline stage to filter by (e.g. ANALYSIS).

        Returns:
            List of entries matching the given stage.
        """
        return [entry for entry in self.entries if entry.stage == stage]

    def get_entries_by_agent(self, agent: str) -> list[AuditEntry]:
        """Filter audit entries by agent name.

        Args:
            agent: The agent name to filter by (e.g. analyst).

        Returns:
            List of entries matching the given agent.
        """
        return [entry for entry in self.entries if entry.agent == agent]
