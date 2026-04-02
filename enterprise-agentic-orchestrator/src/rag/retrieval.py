"""Hybrid retrieval module for the RAG pipeline.

Combines Weaviate vector similarity search with BM25 keyword search,
supporting configurable alpha blending and metadata filtering across
all four RAG collections.
"""

from __future__ import annotations

from weaviate.classes.query import Filter, MetadataQuery

from src.rag.embeddings import embed_texts


def hybrid_search(
    client,
    collection_name: str,
    query: str,
    alpha: float = 0.5,
    limit: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    """Execute a hybrid (vector + BM25) search against a Weaviate collection.

    Parameters
    ----------
    client:
        An open ``weaviate.WeaviateClient`` instance.
    collection_name:
        Name of the target Weaviate collection.
    query:
        Natural-language search query.
    alpha:
        Blend weight between keyword and vector search.
        ``0.0`` = pure BM25 keyword, ``1.0`` = pure vector similarity,
        ``0.5`` = balanced hybrid (default).
    limit:
        Maximum number of results to return.
    filters:
        Optional dict of property-name to value pairs for metadata
        filtering.  Values should use the correct Python type
        (``int`` for INT properties like ``financial_year``, ``str``
        for TEXT properties).

    Returns
    -------
    list[dict]
        Each dict contains all stored object properties plus a
        ``_score`` key with the hybrid relevance score.
    """
    # Embed the query text using the same model as ingestion
    query_vector = embed_texts([query])[0]

    # Build Weaviate filter from dict
    weaviate_filter = None
    if filters:
        conditions = [
            Filter.by_property(key).equal(value)
            for key, value in filters.items()
        ]
        if len(conditions) == 1:
            weaviate_filter = conditions[0]
        else:
            weaviate_filter = Filter.all_of(conditions)

    collection = client.collections.get(collection_name)
    response = collection.query.hybrid(
        query=query,
        vector=query_vector,
        alpha=alpha,
        limit=limit,
        filters=weaviate_filter,
        return_metadata=MetadataQuery(score=True),
    )

    results: list[dict] = []
    for obj in response.objects:
        item = {**obj.properties}
        item["_score"] = obj.metadata.score
        results.append(item)

    return results


# -------------------------------------------------------------------
# Collection-specific convenience functions
# -------------------------------------------------------------------


def _build_filters(**kwargs) -> dict | None:
    """Build a filters dict from non-None keyword arguments."""
    filters = {k: v for k, v in kwargs.items() if v is not None}
    return filters or None


def search_financial_documents(
    client,
    query: str,
    alpha: float = 0.5,
    limit: int = 5,
    sector: str | None = None,
    financial_year: int | None = None,
    company_name: str | None = None,
) -> list[dict]:
    """Search the FinancialDocuments collection with optional metadata filters."""
    return hybrid_search(
        client,
        collection_name="FinancialDocuments",
        query=query,
        alpha=alpha,
        limit=limit,
        filters=_build_filters(
            sector=sector,
            financial_year=financial_year,
            company_name=company_name,
        ),
    )


def search_sector_analyses(
    client,
    query: str,
    alpha: float = 0.5,
    limit: int = 5,
    sector: str | None = None,
    outlook: str | None = None,
) -> list[dict]:
    """Search the SectorAnalysis collection with optional metadata filters."""
    return hybrid_search(
        client,
        collection_name="SectorAnalysis",
        query=query,
        alpha=alpha,
        limit=limit,
        filters=_build_filters(sector=sector, outlook=outlook),
    )


def search_regulatory_policies(
    client,
    query: str,
    alpha: float = 0.5,
    limit: int = 5,
    policy_area: str | None = None,
) -> list[dict]:
    """Search the RegulatoryPolicies collection with optional metadata filters."""
    return hybrid_search(
        client,
        collection_name="RegulatoryPolicies",
        query=query,
        alpha=alpha,
        limit=limit,
        filters=_build_filters(policy_area=policy_area),
    )


def search_historical_decisions(
    client,
    query: str,
    alpha: float = 0.5,
    limit: int = 5,
    sector: str | None = None,
    performance_outcome: str | None = None,
) -> list[dict]:
    """Search the HistoricalDecisions collection with optional metadata filters."""
    return hybrid_search(
        client,
        collection_name="HistoricalDecisions",
        query=query,
        alpha=alpha,
        limit=limit,
        filters=_build_filters(
            sector=sector,
            performance_outcome=performance_outcome,
        ),
    )
