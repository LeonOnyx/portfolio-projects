"""Tool registry for runtime discovery and invocation.

Provides :class:`ToolRegistry` for registering, querying, and filtering
tools by name or category.  The module-level :data:`registry` instance
is populated by :func:`register_all_tools` which lazily imports all
domain and RAG tool functions to avoid circular imports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolMetadata:
    """Descriptor for a registered tool."""

    name: str
    description: str
    callable: Callable[..., Any]
    parameters: dict[str, str] = field(default_factory=dict)
    category: str = "domain"


class ToolRegistry:
    """In-memory registry supporting registration and lookup of tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolMetadata] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, metadata: ToolMetadata) -> None:
        """Register a tool by its metadata name."""
        self._tools[metadata.name] = metadata
        logger.debug("Registered tool: %s [%s]", metadata.name, metadata.category)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolMetadata | None:
        """Return full metadata for *name*, or ``None``."""
        return self._tools.get(name)

    def get_callable(self, name: str) -> Callable[..., Any] | None:
        """Return just the callable for *name*, or ``None``."""
        meta = self._tools.get(name)
        return meta.callable if meta else None

    # ------------------------------------------------------------------
    # Listing / filtering
    # ------------------------------------------------------------------

    def list_tools(self, category: str | None = None) -> list[ToolMetadata]:
        """Return all tools, optionally filtered by *category*."""
        if category is None:
            return list(self._tools.values())
        return [m for m in self._tools.values() if m.category == category]

    def list_names(self, category: str | None = None) -> list[str]:
        """Return tool names, optionally filtered by *category*."""
        return [m.name for m in self.list_tools(category)]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# Module-level singleton
registry = ToolRegistry()


def register_all_tools() -> None:
    """Populate :data:`registry` with all domain and RAG tools.

    Imports are performed lazily inside this function to prevent
    circular import issues at module load time.
    """
    # -- Domain / analysis tools ----------------------------------------
    from src.tools.credit_scorer import calculate_credit_score
    from src.tools.risk_calculator import calculate_risk_metrics
    from src.tools.sector_lookup import lookup_sector
    from src.tools.concentration_checker import check_concentration
    from src.tools.stress_tester import run_stress_tests

    registry.register(ToolMetadata(
        name="credit_scorer",
        description="Score creditworthiness (0-100) from financial metrics with factor breakdown",
        callable=calculate_credit_score,
        parameters={
            "financials": "Dict of financial metrics (revenue, total_costs, net_profit, etc.)",
            "years_trading": "Number of years the business has been trading",
            "sector_outlook": "Sector outlook string (positive/stable/negative/uncertain)",
            "ccj_count": "Number of County Court Judgments",
            "security_value": "Value of security/collateral offered",
            "loan_amount": "Requested loan amount",
        },
        category="analysis",
    ))

    registry.register(ToolMetadata(
        name="risk_calculator",
        description="Compute PD, LGD, EAD, and Expected Loss from credit score and loan details",
        callable=calculate_risk_metrics,
        parameters={
            "credit_score": "Credit score (0-100)",
            "loan_amount": "Requested loan amount",
            "security_value": "Value of security/collateral",
            "security_type": "Type of security (property/debenture/personal_guarantee/unsecured)",
        },
        category="analysis",
    ))

    registry.register(ToolMetadata(
        name="sector_lookup",
        description="Retrieve sector analysis from knowledge base with outlook assessment",
        callable=lookup_sector,
        parameters={
            "sector": "Industry sector name to look up",
        },
        category="analysis",
    ))

    registry.register(ToolMetadata(
        name="concentration_checker",
        description="Check portfolio single-name and sector concentration limits",
        callable=check_concentration,
        parameters={
            "loan_amount": "Proposed loan amount",
            "borrower_name": "Name of the borrower",
            "sector": "Industry sector",
            "portfolio_total": "Total portfolio value",
            "existing_exposures_by_name": "Dict mapping borrower names to existing exposure amounts",
            "existing_exposures_by_sector": "Dict mapping sectors to existing exposure amounts",
        },
        category="analysis",
    ))

    registry.register(ToolMetadata(
        name="stress_tester",
        description="Run 5 adverse scenarios on financial position and assess resilience",
        callable=run_stress_tests,
        parameters={
            "revenue": "Annual revenue",
            "total_costs": "Annual total costs",
            "net_profit": "Annual net profit",
            "credit_score": "Current credit score (0-100)",
        },
        category="analysis",
    ))

    # -- RAG tools ------------------------------------------------------
    from src.tools.rag_tools import (
        rag_financial_lookup,
        rag_sector_analysis,
        rag_policy_lookup,
        historical_comparator,
    )

    registry.register(ToolMetadata(
        name="rag_financial_lookup",
        description="Search financial documents with metadata filtering",
        callable=rag_financial_lookup,
        parameters={
            "query": "Natural-language search query",
            "sector": "Optional sector filter",
            "financial_year": "Optional financial year filter",
            "company_name": "Optional company name filter",
            "limit": "Max results to return (default 5)",
            "alpha": "Hybrid search alpha 0-1 (0=keyword, 1=vector, default 0.7)",
        },
        category="rag",
    ))

    registry.register(ToolMetadata(
        name="rag_sector_analysis",
        description="Search sector analysis documents",
        callable=rag_sector_analysis,
        parameters={
            "query": "Natural-language search query",
            "sector": "Optional sector filter",
            "limit": "Max results to return (default 5)",
            "alpha": "Hybrid search alpha 0-1 (default 0.7)",
        },
        category="rag",
    ))

    registry.register(ToolMetadata(
        name="rag_policy_lookup",
        description="Search regulatory policy documents",
        callable=rag_policy_lookup,
        parameters={
            "query": "Natural-language search query",
            "policy_area": "Optional policy area filter",
            "limit": "Max results to return (default 5)",
            "alpha": "Hybrid search alpha 0-1 (default 0.7)",
        },
        category="rag",
    ))

    registry.register(ToolMetadata(
        name="historical_comparator",
        description="Search historical lending decisions",
        callable=historical_comparator,
        parameters={
            "query": "Natural-language search query",
            "sector": "Optional sector filter",
            "performance_outcome": "Optional outcome filter (performing/watch_list/default/write_off)",
            "limit": "Max results to return (default 5)",
            "alpha": "Hybrid search alpha 0-1 (default 0.7)",
        },
        category="rag",
    ))

    logger.info("Tool registry populated: %d tools registered", len(registry))
