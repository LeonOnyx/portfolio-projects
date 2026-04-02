"""Document ingestion pipeline for the RAG system.

Transforms generator output (loan applications, sector analyses, regulatory
policies, historical decisions) into chunked, embedded Weaviate objects.
Each per-collection function composes content text, chunks via
SentenceSplitter, embeds via OpenAI, and batch-inserts into the
corresponding Weaviate collection.
"""

from __future__ import annotations

import logging
from typing import Any

from src.rag.chunking import chunk_document
from src.rag.embeddings import embed_texts

logger = logging.getLogger(__name__)

# Maximum texts to send in a single embed_texts() call to stay within
# API limits and avoid excessively large payloads.
_EMBED_BATCH_SIZE = 2000


# ---------------------------------------------------------------------------
# Content composition helpers (private)
# ---------------------------------------------------------------------------


def _financial_summary_to_text(app: dict) -> str:
    """Convert a loan application dict into a prose narrative for embedding.

    Parameters
    ----------
    app:
        A loan application dictionary (from ``LoanApplication.model_dump(mode="json")``).

    Returns
    -------
    str
        Human-readable prose summary of the application.
    """
    applicant = app.get("applicant", {})
    company_name = applicant.get("company_name", "Unknown Company")
    sector = applicant.get("sector", "unknown")
    years_trading = applicant.get("years_trading", "N/A")

    loan = app.get("loan", {})
    loan_amount = loan.get("amount_requested", "N/A")
    loan_purpose = loan.get("purpose", "unspecified")

    # Latest financial year (last entry in the list)
    financials_list = app.get("financials", [])
    if financials_list:
        latest = financials_list[-1]
        financial_year = latest.get("year", "N/A")
        revenue = latest.get("revenue", "N/A")
        net_profit = latest.get("net_profit", "N/A")
        total_assets = latest.get("total_assets", "N/A")
        total_liabilities = latest.get("total_liabilities", "N/A")
    else:
        financial_year = "N/A"
        revenue = "N/A"
        net_profit = "N/A"
        total_assets = "N/A"
        total_liabilities = "N/A"

    credit_score = app.get("credit_score", "N/A")

    return (
        f"{company_name} is a {sector} company trading for {years_trading} years. "
        f"Revenue: GBP {revenue}, Net Profit: GBP {net_profit}, "
        f"Total Assets: GBP {total_assets}, Total Liabilities: GBP {total_liabilities}. "
        f"Requested loan: GBP {loan_amount} for {loan_purpose}. "
        f"Credit score: {credit_score}/100."
    )


def _historical_decision_to_text(decision: dict) -> str:
    """Compose a prose narrative from a historical decision dict.

    Parameters
    ----------
    decision:
        A historical decision dictionary as produced by
        ``generate_historical_decisions()``.

    Returns
    -------
    str
        Human-readable prose summary of the decision and its outcome.
    """
    company_name = decision.get("company_name", "Unknown Company")
    sector = decision.get("sector", "unknown")
    original_decision = decision.get("original_decision", "unknown")
    loan_amount = decision.get("loan_amount", "N/A")
    outcome = decision.get("performance_outcome", "unknown")

    risk_factors = decision.get("risk_factors_at_decision", [])
    risk_factors_text = "; ".join(risk_factors) if risk_factors else "none recorded"

    lessons = decision.get("lessons_learned", "No lessons recorded.")

    return (
        f"Decision for {company_name} ({sector}): {original_decision}. "
        f"Loan amount: GBP {loan_amount}. Outcome: {outcome}. "
        f"Risk factors: {risk_factors_text}. Lessons: {lessons}"
    )


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------


def _batch_embed(texts: list[str]) -> list[list[float]]:
    """Embed texts in batches of ``_EMBED_BATCH_SIZE`` to limit payload size.

    Returns a flat list of embeddings corresponding 1-to-1 with *texts*.
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    for start in range(0, len(texts), _EMBED_BATCH_SIZE):
        batch = texts[start : start + _EMBED_BATCH_SIZE]
        all_embeddings.extend(embed_texts(batch))
    return all_embeddings


# ---------------------------------------------------------------------------
# Batch insertion helper
# ---------------------------------------------------------------------------


def _batch_insert(
    client: Any,
    collection_name: str,
    chunks: list[dict],
    embeddings: list[list[float]],
    text_key: str = "content",
) -> int:
    """Insert chunks with their vectors into a Weaviate collection.

    Uses ``collection.batch.fixed_size`` for efficient bulk insertion.

    Returns the number of objects successfully inserted.
    """
    collection = client.collections.get(collection_name)

    inserted = 0
    failed_objects: list[Any] = []

    with collection.batch.fixed_size(batch_size=100) as batch:
        for chunk, vector in zip(chunks, embeddings):
            # Build properties from the chunk dict, excluding keys that
            # are not part of the Weaviate schema (non-string metadata).
            properties = {k: v for k, v in chunk.items()}
            batch.add_object(properties=properties, vector=vector)
            inserted += 1

    # Check for failures after batch context closes
    if hasattr(collection.batch, "failed_objects"):
        try:
            fo = collection.batch.failed_objects
            if callable(fo):
                fo = fo()
            if fo:
                failed_objects = list(fo)
        except Exception:
            pass

    if failed_objects:
        logger.warning(
            "%s: %d objects failed during batch insert",
            collection_name,
            len(failed_objects),
        )
        inserted -= len(failed_objects)

    return max(inserted, 0)


# ---------------------------------------------------------------------------
# Per-collection ingestion functions
# ---------------------------------------------------------------------------


def ingest_financial_documents(client: Any, applications: list[dict]) -> int:
    """Ingest loan applications into the FinancialDocuments collection.

    Each application is converted to a prose narrative, then chunked,
    embedded, and batch-inserted.

    Parameters
    ----------
    client:
        Connected Weaviate client instance.
    applications:
        List of loan application dicts (from ``model_dump(mode="json")``).

    Returns
    -------
    int
        Number of objects inserted.
    """
    all_chunks: list[dict] = []

    for app in applications:
        content = _financial_summary_to_text(app)

        # Latest financial year for the schema field
        financials_list = app.get("financials", [])
        financial_year = financials_list[-1].get("year", 2024) if financials_list else 2024

        doc_dict = {
            "document_id": app.get("application_id", ""),
            "document_type": "loan_application",
            "company_name": app.get("applicant", {}).get("company_name", ""),
            "sector": app.get("applicant", {}).get("sector", ""),
            "financial_year": financial_year,
            "sensitivity_level": "confidential",
            "content": content,
        }

        chunks = chunk_document(doc_dict)
        all_chunks.extend(chunks)

    # Embed all chunk contents at once (batched internally)
    all_texts = [c["content"] for c in all_chunks]
    all_embeddings = _batch_embed(all_texts)

    inserted = _batch_insert(client, "FinancialDocuments", all_chunks, all_embeddings)
    return inserted


def ingest_sector_analyses(client: Any, reports: list[dict]) -> int:
    """Ingest sector analysis reports into the SectorAnalysis collection.

    Reports already have a ``content`` field from the generator.

    Parameters
    ----------
    client:
        Connected Weaviate client instance.
    reports:
        List of sector report dicts from ``generate_sector_reports()``.

    Returns
    -------
    int
        Number of objects inserted.
    """
    all_chunks: list[dict] = []

    for report in reports:
        doc_dict = {
            "document_id": report.get("document_id", ""),
            "document_type": report.get("document_type", "sector_analysis"),
            "sector": report.get("sector", ""),
            "title": report.get("title", ""),
            "outlook": report.get("outlook", ""),
            "risk_level": report.get("risk_level", ""),
            "content": report.get("content", ""),
        }

        chunks = chunk_document(doc_dict)
        all_chunks.extend(chunks)

    all_texts = [c["content"] for c in all_chunks]
    all_embeddings = _batch_embed(all_texts)

    inserted = _batch_insert(client, "SectorAnalysis", all_chunks, all_embeddings)
    return inserted


def ingest_regulatory_policies(client: Any, policies: list[dict]) -> int:
    """Ingest regulatory policy documents into the RegulatoryPolicies collection.

    Policies already have a ``content`` field from the generator.

    Parameters
    ----------
    client:
        Connected Weaviate client instance.
    policies:
        List of regulatory policy dicts from ``generate_regulatory_docs()``.

    Returns
    -------
    int
        Number of objects inserted.
    """
    all_chunks: list[dict] = []

    for policy in policies:
        doc_dict = {
            "document_id": policy.get("document_id", ""),
            "document_type": policy.get("document_type", "regulatory_policy"),
            "policy_area": policy.get("policy_area", ""),
            "title": policy.get("title", ""),
            "regulation_reference": policy.get("regulation_reference", ""),
            "effective_date": policy.get("effective_date", ""),
            "content": policy.get("content", ""),
        }

        chunks = chunk_document(doc_dict)
        all_chunks.extend(chunks)

    all_texts = [c["content"] for c in all_chunks]
    all_embeddings = _batch_embed(all_texts)

    inserted = _batch_insert(client, "RegulatoryPolicies", all_chunks, all_embeddings)
    return inserted


def ingest_historical_decisions(client: Any, decisions: list[dict]) -> int:
    """Ingest historical lending decisions into the HistoricalDecisions collection.

    Each decision has its content composed from risk factors and lessons
    learned via ``_historical_decision_to_text()``.

    Parameters
    ----------
    client:
        Connected Weaviate client instance.
    decisions:
        List of historical decision dicts from ``generate_historical_decisions()``.

    Returns
    -------
    int
        Number of objects inserted.
    """
    all_chunks: list[dict] = []

    for decision in decisions:
        content = _historical_decision_to_text(decision)

        doc_dict = {
            "decision_id": decision.get("decision_id", ""),
            "document_type": decision.get("document_type", "historical_decision"),
            "application_id": decision.get("application_id", ""),
            "company_name": decision.get("company_name", ""),
            "sector": decision.get("sector", ""),
            "loan_amount": float(decision.get("loan_amount", 0)),
            "performance_outcome": decision.get("performance_outcome", ""),
            "content": content,
        }

        chunks = chunk_document(doc_dict)
        all_chunks.extend(chunks)

    all_texts = [c["content"] for c in all_chunks]
    all_embeddings = _batch_embed(all_texts)

    inserted = _batch_insert(client, "HistoricalDecisions", all_chunks, all_embeddings)
    return inserted


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def ingest_all(
    client: Any,
    applications: list[dict],
    sector_reports: list[dict],
    regulatory_docs: list[dict],
    historical_decisions: list[dict],
) -> dict[str, int]:
    """Run the full ingestion pipeline across all four document collections.

    Parameters
    ----------
    client:
        Connected Weaviate client instance.
    applications:
        Loan application dicts (model_dump output).
    sector_reports:
        Sector analysis dicts.
    regulatory_docs:
        Regulatory policy dicts.
    historical_decisions:
        Historical decision dicts.

    Returns
    -------
    dict[str, int]
        Counts of objects inserted per collection.
    """
    print("Ingesting FinancialDocuments ...")
    fin_count = ingest_financial_documents(client, applications)
    print(f"  FinancialDocuments: {fin_count} objects")

    print("Ingesting SectorAnalysis ...")
    sec_count = ingest_sector_analyses(client, sector_reports)
    print(f"  SectorAnalysis: {sec_count} objects")

    print("Ingesting RegulatoryPolicies ...")
    reg_count = ingest_regulatory_policies(client, regulatory_docs)
    print(f"  RegulatoryPolicies: {reg_count} objects")

    print("Ingesting HistoricalDecisions ...")
    hist_count = ingest_historical_decisions(client, historical_decisions)
    print(f"  HistoricalDecisions: {hist_count} objects")

    return {
        "FinancialDocuments": fin_count,
        "SectorAnalysis": sec_count,
        "RegulatoryPolicies": reg_count,
        "HistoricalDecisions": hist_count,
    }
