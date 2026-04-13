"""Agent-callable RAG tool functions.

Each function manages its own Weaviate connection, calls the appropriate
retrieval function, and returns a structured dict with source citations
that agents can consume directly.
"""

from __future__ import annotations

import logging
import os

import weaviate

from src.rag.retrieval import (
    search_financial_documents,
    search_historical_decisions,
    search_regulatory_policies,
    search_sector_analyses,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weaviate client helper
# ---------------------------------------------------------------------------


def _get_weaviate_client() -> weaviate.WeaviateClient:
    """Return a Weaviate client connected to the local instance.

    Reads connection parameters from environment variables first (for Docker),
    then falls back to ``config/config.yaml`` via :class:`~src.config.ConfigLoader`
    when ``WEAVIATE_HOST`` is not set, and finally to built-in defaults
    (``localhost:8080``, gRPC port ``50051``).
    """
    host = os.environ.get("WEAVIATE_HOST", "localhost")
    port = int(os.environ.get("WEAVIATE_HTTP_PORT", "8080"))
    grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))

    # Only consult ConfigLoader when env vars are NOT set (local dev)
    if "WEAVIATE_HOST" not in os.environ:
        try:
            from src.config import ConfigLoader

            cfg = ConfigLoader()
            app_cfg = cfg.app()
            # weaviate_url is like "http://localhost:8080"
            url = app_cfg.providers.weaviate_url
            if "://" in url:
                url = url.split("://", 1)[1]
            if ":" in url:
                host, port_str = url.rsplit(":", 1)
                port = int(port_str)
            else:
                host = url
            grpc_port = app_cfg.providers.weaviate_grpc_port
        except Exception:
            logger.debug("ConfigLoader unavailable, using Weaviate defaults")

    return weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)


# ---------------------------------------------------------------------------
# Citation helper
# ---------------------------------------------------------------------------


def _make_citation(
    collection: str, document_id: str, label: str, score: float
) -> str:
    """Format a source citation string for agent output."""
    return f"[{collection}:{document_id}] {label} (score: {score:.3f})"


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------


def rag_financial_lookup(
    query: str,
    sector: str | None = None,
    financial_year: int | None = None,
    company_name: str | None = None,
    limit: int = 5,
    alpha: float = 0.7,
) -> dict:
    """Search financial documents and return structured results with citations.

    Parameters
    ----------
    query:
        Natural-language search query.
    sector:
        Optional sector filter (e.g. ``"Technology"``).
    financial_year:
        Optional financial year filter (e.g. ``2024``).
    company_name:
        Optional company name filter.
    limit:
        Maximum number of results (default 5).
    alpha:
        Hybrid search blend weight (0 = keyword, 1 = vector, 0.7 = default).

    Returns
    -------
    dict
        Structured tool output with ``query``, ``source_collection``,
        ``result_count``, and ``results`` list.
    """
    collection = "FinancialDocuments"
    try:
        with _get_weaviate_client() as client:
            raw = search_financial_documents(
                client,
                query=query,
                alpha=alpha,
                limit=limit,
                sector=sector,
                financial_year=financial_year,
                company_name=company_name,
            )
    except Exception as exc:
        logger.error("Weaviate connection failed for %s: %s", collection, exc)
        return {
            "query": query,
            "source_collection": collection,
            "result_count": 0,
            "results": [],
            "error": "Weaviate unavailable",
        }

    results = []
    for item in raw:
        doc_id = item.get("document_id", "unknown")
        comp = item.get("company_name", "Unknown")
        score = item.get("_score", 0.0)
        results.append(
            {
                "document_id": doc_id,
                "company_name": comp,
                "sector": item.get("sector", ""),
                "financial_year": item.get("financial_year"),
                "content": item.get("content", ""),
                "relevance_score": score,
                "citation": _make_citation(collection, doc_id, comp, score),
            }
        )

    return {
        "query": query,
        "source_collection": collection,
        "result_count": len(results),
        "results": results,
    }


def rag_sector_analysis(
    query: str,
    sector: str | None = None,
    limit: int = 5,
    alpha: float = 0.7,
) -> dict:
    """Search sector analysis documents and return structured results with citations.

    Parameters
    ----------
    query:
        Natural-language search query.
    sector:
        Optional sector filter.
    limit:
        Maximum number of results (default 5).
    alpha:
        Hybrid search blend weight.

    Returns
    -------
    dict
        Structured tool output with ``query``, ``source_collection``,
        ``result_count``, and ``results`` list.
    """
    collection = "SectorAnalysis"
    try:
        with _get_weaviate_client() as client:
            raw = search_sector_analyses(
                client,
                query=query,
                alpha=alpha,
                limit=limit,
                sector=sector,
            )
    except Exception as exc:
        logger.error("Weaviate connection failed for %s: %s", collection, exc)
        return {
            "query": query,
            "source_collection": collection,
            "result_count": 0,
            "results": [],
            "error": "Weaviate unavailable",
        }

    results = []
    for item in raw:
        doc_id = item.get("document_id", "unknown")
        title = item.get("title", "Untitled")
        score = item.get("_score", 0.0)
        results.append(
            {
                "document_id": doc_id,
                "sector": item.get("sector", ""),
                "title": title,
                "outlook": item.get("outlook", ""),
                "risk_level": item.get("risk_level", ""),
                "content": item.get("content", ""),
                "relevance_score": score,
                "citation": _make_citation(collection, doc_id, title, score),
            }
        )

    return {
        "query": query,
        "source_collection": collection,
        "result_count": len(results),
        "results": results,
    }


def rag_policy_lookup(
    query: str,
    policy_area: str | None = None,
    limit: int = 5,
    alpha: float = 0.7,
) -> dict:
    """Search regulatory policy documents and return structured results with citations.

    Parameters
    ----------
    query:
        Natural-language search query.
    policy_area:
        Optional policy area filter (e.g. ``"Capital Requirements"``).
    limit:
        Maximum number of results (default 5).
    alpha:
        Hybrid search blend weight.

    Returns
    -------
    dict
        Structured tool output with ``query``, ``source_collection``,
        ``result_count``, and ``results`` list.
    """
    collection = "RegulatoryPolicies"
    try:
        with _get_weaviate_client() as client:
            raw = search_regulatory_policies(
                client,
                query=query,
                alpha=alpha,
                limit=limit,
                policy_area=policy_area,
            )
    except Exception as exc:
        logger.error("Weaviate connection failed for %s: %s", collection, exc)
        return {
            "query": query,
            "source_collection": collection,
            "result_count": 0,
            "results": [],
            "error": "Weaviate unavailable",
        }

    results = []
    for item in raw:
        doc_id = item.get("document_id", "unknown")
        title = item.get("title", "Untitled")
        score = item.get("_score", 0.0)
        results.append(
            {
                "document_id": doc_id,
                "policy_area": item.get("policy_area", ""),
                "title": title,
                "regulation_reference": item.get("regulation_reference", ""),
                "content": item.get("content", ""),
                "relevance_score": score,
                "citation": _make_citation(collection, doc_id, title, score),
            }
        )

    return {
        "query": query,
        "source_collection": collection,
        "result_count": len(results),
        "results": results,
    }


def historical_comparator(
    query: str,
    sector: str | None = None,
    performance_outcome: str | None = None,
    limit: int = 5,
    alpha: float = 0.7,
) -> dict:
    """Search historical decisions and return structured results with citations.

    Parameters
    ----------
    query:
        Natural-language search query.
    sector:
        Optional sector filter.
    performance_outcome:
        Optional outcome filter (e.g. ``"default"``, ``"performing"``).
    limit:
        Maximum number of results (default 5).
    alpha:
        Hybrid search blend weight.

    Returns
    -------
    dict
        Structured tool output with ``query``, ``source_collection``,
        ``result_count``, and ``results`` list.
    """
    collection = "HistoricalDecisions"
    try:
        with _get_weaviate_client() as client:
            raw = search_historical_decisions(
                client,
                query=query,
                alpha=alpha,
                limit=limit,
                sector=sector,
                performance_outcome=performance_outcome,
            )
    except Exception as exc:
        logger.error("Weaviate connection failed for %s: %s", collection, exc)
        return {
            "query": query,
            "source_collection": collection,
            "result_count": 0,
            "results": [],
            "error": "Weaviate unavailable",
        }

    results = []
    for item in raw:
        decision_id = item.get("decision_id", item.get("document_id", "unknown"))
        comp = item.get("company_name", "Unknown")
        score = item.get("_score", 0.0)
        results.append(
            {
                "decision_id": decision_id,
                "company_name": comp,
                "sector": item.get("sector", ""),
                "loan_amount": item.get("loan_amount"),
                "performance_outcome": item.get("performance_outcome", ""),
                "content": item.get("content", ""),
                "relevance_score": score,
                "citation": _make_citation(collection, decision_id, comp, score),
            }
        )

    return {
        "query": query,
        "source_collection": collection,
        "result_count": len(results),
        "results": results,
    }
