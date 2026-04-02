"""Sector lookup tool wrapping the RAG pipeline for agent consumption.

Provides a thin wrapper around :func:`~src.tools.rag_tools.rag_sector_analysis`
that returns a strongly-typed :class:`SectorLookupResult` with outlook assessment,
risk level, summary, and source citations.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from src.tools.rag_tools import rag_sector_analysis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class SectorLookupResult(BaseModel):
    """Structured result from a sector lookup query."""

    sector: str = Field(description="The sector queried")
    outlook: str = Field(description="Outlook assessment (positive/stable/cautious/negative/unknown)")
    risk_level: str = Field(description="Risk level from sector analysis")
    summary: str = Field(description="Key findings summary (truncated to 500 chars)")
    source_count: int = Field(description="Number of sources found")
    citations: list[str] = Field(default_factory=list, description="Citation strings from RAG results")
    error: str | None = Field(default=None, description="Error message if lookup failed")


# ---------------------------------------------------------------------------
# Lookup function
# ---------------------------------------------------------------------------


def lookup_sector(sector: str) -> SectorLookupResult:
    """Look up sector outlook and risk assessment via RAG pipeline.

    Calls :func:`rag_sector_analysis` with a targeted query and returns a
    :class:`SectorLookupResult`.  If the RAG pipeline returns no results or
    encounters an error, the result will have ``outlook="unknown"`` and an
    error message.

    Args:
        sector: The sector to look up (e.g. ``"Technology"``, ``"Construction"``).

    Returns:
        SectorLookupResult with outlook, risk level, summary, and citations.
    """
    logger.info("Looking up sector: %s", sector)

    result = rag_sector_analysis(
        query=f"{sector} sector outlook and risk assessment",
        sector=sector,
        limit=3,
    )

    # Handle error or empty results
    if result.get("error") or result.get("result_count", 0) == 0:
        error_msg = result.get("error", "No sector data available")
        logger.warning("Sector lookup failed for %s: %s", sector, error_msg)
        return SectorLookupResult(
            sector=sector,
            outlook="unknown",
            risk_level="unknown",
            summary="No sector data available",
            source_count=0,
            citations=[],
            error=error_msg,
        )

    # Extract from first result
    results = result["results"]
    first = results[0]
    outlook = first.get("outlook", "unknown") or "unknown"
    risk_level = first.get("risk_level", "unknown") or "unknown"
    content = first.get("content", "")
    summary = content[:500] if content else "No content available"

    # Collect citations from all results
    citations = [r["citation"] for r in results if r.get("citation")]

    logger.info(
        "Sector lookup complete for %s: outlook=%s, risk_level=%s, sources=%d",
        sector,
        outlook,
        risk_level,
        len(results),
    )

    return SectorLookupResult(
        sector=sector,
        outlook=outlook,
        risk_level=risk_level,
        summary=summary,
        source_count=len(results),
        citations=citations,
    )
