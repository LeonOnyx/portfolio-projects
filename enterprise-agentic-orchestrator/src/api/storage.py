"""JSON file-based assessment persistence.

Assessment results from the orchestrator are serialised to individual
JSON files keyed by ``request_id``.  The storage directory is created
on first use and files survive server restarts (no in-memory cache).

This intentionally uses synchronous I/O wrapped in async methods --
JSON assessment files are small (< 100 KB) and the overhead of
thread-pool delegation would exceed the actual file I/O cost.  For
production deployments a database backend would replace this module
without changing the interface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AssessmentStorage:
    """Persist and retrieve credit-risk assessment results as JSON files.

    Each assessment is stored at ``{data_dir}/{request_id}.json``.

    Parameters
    ----------
    data_dir:
        Filesystem path for the assessments directory. Created
        automatically (including parents) if it does not exist.
    """

    def __init__(self, data_dir: str = "data/assessments") -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info("AssessmentStorage initialised at %s", self._dir.resolve())

    async def save(self, request_id: str, result: dict) -> None:
        """Write an assessment result to disk.

        Uses ``default=str`` to handle Decimal, datetime, UUID, and
        other types that the orchestrator state dict may contain.
        """
        path = self._dir / f"{request_id}.json"
        path.write_text(json.dumps(result, default=str, indent=2), encoding="utf-8")
        logger.debug("Saved assessment %s -> %s", request_id, path)

    async def get(self, request_id: str) -> dict | None:
        """Retrieve a previously saved assessment by request_id.

        Returns ``None`` if no file exists for the given id (rather
        than raising), so callers can map this to a 404 directly.
        """
        path = self._dir / f"{request_id}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    async def list_ids(self) -> list[str]:
        """Return all stored request_ids (filename stems, sorted)."""
        return sorted(p.stem for p in self._dir.glob("*.json"))
