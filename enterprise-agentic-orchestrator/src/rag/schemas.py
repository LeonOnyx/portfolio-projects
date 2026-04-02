"""Weaviate collection schema definitions for the RAG pipeline.

Defines four collections matching the synthetic data generators:
FinancialDocuments, SectorAnalysis, RegulatoryPolicies, and
HistoricalDecisions.  Each collection uses self-provided vectors
(embeddings generated externally via OpenAI).
"""

from __future__ import annotations

import weaviate.classes.config as wvc

COLLECTION_NAMES: list[str] = [
    "FinancialDocuments",
    "SectorAnalysis",
    "RegulatoryPolicies",
    "HistoricalDecisions",
]


def create_financial_documents_collection(client) -> None:
    """Create the FinancialDocuments collection."""
    client.collections.create(
        name="FinancialDocuments",
        vector_config=wvc.Configure.Vectors.self_provided(),
        properties=[
            wvc.Property(name="document_id", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.FIELD),
            wvc.Property(name="document_type", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="company_name", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="sector", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="financial_year", data_type=wvc.DataType.INT),
            wvc.Property(name="sensitivity_level", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.FIELD),
            wvc.Property(name="content", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="chunk_index", data_type=wvc.DataType.INT),
            wvc.Property(name="total_chunks", data_type=wvc.DataType.INT),
        ],
    )


def create_sector_analysis_collection(client) -> None:
    """Create the SectorAnalysis collection."""
    client.collections.create(
        name="SectorAnalysis",
        vector_config=wvc.Configure.Vectors.self_provided(),
        properties=[
            wvc.Property(name="document_id", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.FIELD),
            wvc.Property(name="document_type", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="sector", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="title", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="outlook", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="risk_level", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="content", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="chunk_index", data_type=wvc.DataType.INT),
            wvc.Property(name="total_chunks", data_type=wvc.DataType.INT),
        ],
    )


def create_regulatory_policies_collection(client) -> None:
    """Create the RegulatoryPolicies collection."""
    client.collections.create(
        name="RegulatoryPolicies",
        vector_config=wvc.Configure.Vectors.self_provided(),
        properties=[
            wvc.Property(name="document_id", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.FIELD),
            wvc.Property(name="document_type", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="policy_area", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="title", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="regulation_reference", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.FIELD),
            wvc.Property(name="effective_date", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.FIELD),
            wvc.Property(name="content", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="chunk_index", data_type=wvc.DataType.INT),
            wvc.Property(name="total_chunks", data_type=wvc.DataType.INT),
        ],
    )


def create_historical_decisions_collection(client) -> None:
    """Create the HistoricalDecisions collection."""
    client.collections.create(
        name="HistoricalDecisions",
        vector_config=wvc.Configure.Vectors.self_provided(),
        properties=[
            wvc.Property(name="decision_id", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.FIELD),
            wvc.Property(name="document_type", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="application_id", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.FIELD),
            wvc.Property(name="company_name", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="sector", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="loan_amount", data_type=wvc.DataType.NUMBER),
            wvc.Property(name="performance_outcome", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="content", data_type=wvc.DataType.TEXT, tokenization=wvc.Tokenization.LOWERCASE),
            wvc.Property(name="chunk_index", data_type=wvc.DataType.INT),
            wvc.Property(name="total_chunks", data_type=wvc.DataType.INT),
        ],
    )


def create_all_collections(client) -> None:
    """Create all four RAG collections in Weaviate."""
    create_financial_documents_collection(client)
    create_sector_analysis_collection(client)
    create_regulatory_policies_collection(client)
    create_historical_decisions_collection(client)
