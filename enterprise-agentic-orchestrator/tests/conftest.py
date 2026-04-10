"""Shared fixtures for all test tiers (unit, integration, acceptance).

Provides sample data builders, config loader, circuit breaker reset,
Langfuse suppression, and mock agent responses used across the full
test suite.
"""

from __future__ import annotations

import os
from decimal import Decimal
from pathlib import Path

import pytest

from src.agents.base import AgentResponse


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_applicant() -> dict:
    """Minimal valid applicant dict."""
    return {
        "company_name": "Acme Ltd",
        "company_number": "12345678",
        "sector": "technology",
        "years_trading": 10,
        "employee_count": 50,
        "contact_name": "Jane Smith",
        "contact_role": "CFO",
    }


@pytest.fixture()
def sample_financials() -> list[dict]:
    """Two years of financial data with Decimal monetary values."""
    return [
        {
            "year": 2023,
            "revenue": Decimal("900000.00"),
            "gross_profit": Decimal("360000.00"),
            "net_profit": Decimal("180000.00"),
            "total_assets": Decimal("1800000.00"),
            "total_liabilities": Decimal("720000.00"),
            "cash_balance": Decimal("130000.00"),
            "profit_margin": 0.20,
            "debt_to_asset_ratio": 0.40,
        },
        {
            "year": 2024,
            "revenue": Decimal("1000000.00"),
            "gross_profit": Decimal("400000.00"),
            "net_profit": Decimal("200000.00"),
            "total_assets": Decimal("2000000.00"),
            "total_liabilities": Decimal("800000.00"),
            "cash_balance": Decimal("150000.00"),
            "profit_margin": 0.20,
            "debt_to_asset_ratio": 0.40,
        },
    ]


@pytest.fixture()
def sample_loan() -> dict:
    """Minimal valid loan request dict."""
    return {
        "amount_requested": Decimal("250000.00"),
        "term_months": 60,
        "purpose": "Working capital expansion",
        "security_type": "unsecured",
    }


@pytest.fixture()
def valid_application(sample_applicant, sample_financials, sample_loan) -> dict:
    """Full application dict combining applicant, financials, and loan."""
    return {
        "application_id": "APP-TEST-001",
        "applicant": sample_applicant,
        "financials": sample_financials,
        "loan": sample_loan,
    }


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


@pytest.fixture()
def config_loader():
    """ConfigLoader pointed at the real config directory."""
    from src.config.settings import ConfigLoader

    return ConfigLoader(config_dir=Path(__file__).resolve().parent.parent / "config")


# ---------------------------------------------------------------------------
# Autouse: circuit breaker reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """Clear module-level circuit breaker state between tests."""
    try:
        from src.orchestrator_nodes import _circuit_breaker_failures

        _circuit_breaker_failures.clear()
    except ImportError:
        pass
    yield
    try:
        from src.orchestrator_nodes import _circuit_breaker_failures

        _circuit_breaker_failures.clear()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Autouse: disable Langfuse
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def disable_langfuse(monkeypatch):
    """Remove Langfuse env vars so no telemetry is sent during tests."""
    for key in ("LANGFUSE_SECRET_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_HOST"):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Mock agent responses
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_analyst_response() -> AgentResponse:
    """Successful AnalystAgent response."""
    return AgentResponse(
        agent_name="financial_analyst",
        agent_framework="crewai",
        output={
            "report_id": "RPT-ANALYST-001",
            "application_id": "APP-TEST-001",
            "credit_score": 72,
            "risk_metrics": {
                "probability_of_default": 0.08,
                "loss_given_default": 0.45,
                "exposure_at_default": 250000.00,
                "expected_loss": 9000.00,
            },
            "sector_outlook": "stable",
            "recommendation": "APPROVE",
            "reasoning": "Strong financials with positive cash flow trajectory.",
            "source_citations": [
                {"source": "Companies House", "reference": "12345678"},
            ],
        },
        reasoning_trace="Analysed 2 years of financials. Score: 72.",
        confidence=0.8,
        sources_used=[{"tool": "credit_scorer"}],
        tokens_used=1500,
        latency_ms=3200.0,
    )


@pytest.fixture()
def mock_reviewer_response() -> AgentResponse:
    """Successful ReviewerAgent response."""
    return AgentResponse(
        agent_name="independent_reviewer",
        agent_framework="crewai",
        output={
            "quality_score": 0.85,
            "confidence_level": "HIGH",
            "agrees_with_analyst": True,
            "stress_test_results": {},
            "findings": [],
        },
        reasoning_trace="Reviewed analyst report. Concur with recommendation.",
        confidence=0.85,
        sources_used=[{"tool": "stress_tester"}],
        tokens_used=1200,
        latency_ms=2800.0,
    )


@pytest.fixture()
def mock_compliance_response() -> AgentResponse:
    """Successful ComplianceAgent response."""
    return AgentResponse(
        agent_name="compliance_officer",
        agent_framework="autogen",
        output={
            "overall_passed": True,
            "check_results": [],
            "plain_language_summary": "All checks passed",
        },
        reasoning_trace="All regulatory checks completed successfully.",
        confidence=0.9,
        sources_used=[{"tool": "regulation_lookup"}],
        tokens_used=1000,
        latency_ms=2500.0,
    )
