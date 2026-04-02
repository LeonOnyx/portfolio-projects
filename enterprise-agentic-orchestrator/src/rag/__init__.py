"""RAG pipeline foundation for the Enterprise Agentic Orchestrator.

Re-exports core functions for convenient access via
``from src.rag import embed_texts, chunk_document``.
"""

from src.rag.chunking import chunk_document
from src.rag.embeddings import embed_texts
from src.rag.schemas import (
    COLLECTION_NAMES,
    create_all_collections,
    create_financial_documents_collection,
    create_historical_decisions_collection,
    create_regulatory_policies_collection,
    create_sector_analysis_collection,
)

__all__ = [
    "chunk_document",
    "embed_texts",
    "COLLECTION_NAMES",
    "create_all_collections",
    "create_financial_documents_collection",
    "create_historical_decisions_collection",
    "create_regulatory_policies_collection",
    "create_sector_analysis_collection",
]
