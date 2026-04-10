"""RAG retrieval accuracy smoke tests (TEST-10).

These tests require a live Weaviate instance with ingested data.
Run after ``docker-compose up`` and data ingestion::

    pytest -m rag_smoke -v

Excluded from normal test runs (no Weaviate available).
"""

import pytest

pytestmark = pytest.mark.rag_smoke


def test_financial_lookup_returns_results():
    """Known company query returns relevant financial documents."""
    from src.tools.rag_tools import rag_financial_lookup

    result = rag_financial_lookup(query="technology company financial performance")
    assert isinstance(result, dict)
    assert result["result_count"] > 0
    assert result["source_collection"] == "FinancialDocuments"
    assert len(result["results"]) > 0
    for r in result["results"]:
        assert "content" in r or "text" in r


def test_sector_analysis_returns_results():
    """Known sector query returns relevant sector analysis."""
    from src.tools.rag_tools import rag_sector_analysis

    result = rag_sector_analysis(query="construction sector outlook and risks")
    assert isinstance(result, dict)
    assert result["result_count"] > 0
    assert result["source_collection"] == "SectorAnalysis"
    assert len(result["results"]) > 0


def test_policy_lookup_returns_results():
    """Known regulatory query returns relevant policies."""
    from src.tools.rag_tools import rag_policy_lookup

    result = rag_policy_lookup(query="consumer duty fair lending regulation")
    assert isinstance(result, dict)
    assert result["result_count"] > 0
    assert result["source_collection"] == "RegulatoryPolicies"
    assert len(result["results"]) > 0


def test_historical_comparator_returns_results():
    """Known historical query returns relevant decisions."""
    from src.tools.rag_tools import historical_comparator

    result = historical_comparator(query="similar lending decision SME loan")
    assert isinstance(result, dict)
    assert result["result_count"] > 0
    assert result["source_collection"] == "HistoricalDecisions"
    assert len(result["results"]) > 0


def test_financial_lookup_returns_citations():
    """Financial lookup results include source citations."""
    from src.tools.rag_tools import rag_financial_lookup

    result = rag_financial_lookup(query="manufacturing company revenue")
    assert result["result_count"] > 0
    assert "citations" in result or all(
        "citation" in r or "source" in r for r in result["results"]
    )


def test_sector_analysis_specific_sector():
    """Query for a specific sector returns documents about that sector."""
    from src.tools.rag_tools import rag_sector_analysis

    result = rag_sector_analysis(query="hospitality sector")
    assert result["result_count"] > 0
    all_text = " ".join(
        str(r.get("content", r.get("text", ""))) for r in result["results"]
    ).lower()
    assert "hospitality" in all_text
