"""Smoke tests validating the entire Phase 1 foundation.

Covers: model imports, Pydantic validation, serialisation round-trips,
LangGraph StateGraph with reducers, config loading, and framework imports.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Test 1 -- All models importable
# ---------------------------------------------------------------------------

def test_all_models_importable():
    """Every public model class can be imported from src.models."""
    from src.models import (
        ApplicantDetails,
        AnalysisReport,
        AuditEntry,
        BiasCheckResult,
        ComplianceCheckResult,
        ComplianceReport,
        ConfidenceLevel,
        Decision,
        DecisionOutcome,
        FinancialSummary,
        GroundingResult,
        LoanApplication,
        LoanDetails,
        OrchestratorState,
        PIIScanResult,
        Recommendation,
        ReviewReport,
        RiskMetrics,
        SectorType,
        SourceCitation,
        WorkflowStage,
    )

    # Verify they are actual classes / types (not None or stubs)
    expected = [
        ApplicantDetails, AnalysisReport, AuditEntry, BiasCheckResult,
        ComplianceCheckResult, ComplianceReport, ConfidenceLevel, Decision,
        DecisionOutcome, FinancialSummary, GroundingResult, LoanApplication,
        LoanDetails, OrchestratorState, PIIScanResult, Recommendation,
        ReviewReport, RiskMetrics, SectorType, SourceCitation, WorkflowStage,
    ]
    for cls in expected:
        assert cls is not None


# ---------------------------------------------------------------------------
# Helpers -- sample data builders
# ---------------------------------------------------------------------------

def _sample_applicant(**overrides) -> dict:
    defaults = dict(
        company_name="Acme Ltd",
        company_number="12345678",
        sector="technology",
        years_trading=10,
        employee_count=50,
        contact_name="Jane Smith",
        contact_role="CFO",
    )
    defaults.update(overrides)
    return defaults


def _sample_financials(year: int = 2024) -> dict:
    return dict(
        year=year,
        revenue=Decimal("1000000.00"),
        gross_profit=Decimal("400000.00"),
        net_profit=Decimal("200000.00"),
        total_assets=Decimal("2000000.00"),
        total_liabilities=Decimal("800000.00"),
        cash_balance=Decimal("150000.00"),
    )


def _sample_loan(**overrides) -> dict:
    defaults = dict(
        amount_requested=Decimal("250000.00"),
        term_months=60,
        purpose="Working capital expansion",
        security_type="unsecured",
    )
    defaults.update(overrides)
    return defaults


def _valid_application(**overrides) -> dict:
    defaults = dict(
        applicant=_sample_applicant(),
        loan=_sample_loan(),
        financials=[_sample_financials(2023), _sample_financials(2024)],
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Test 2 -- LoanApplication validation
# ---------------------------------------------------------------------------

class TestLoanApplicationValidation:
    """Pydantic validation rules on LoanApplication and sub-models."""

    def test_valid_application_creates_successfully(self):
        from src.models import LoanApplication

        app = LoanApplication(**_valid_application())
        assert app.applicant.company_name == "Acme Ltd"
        assert len(app.financials) == 2

    def test_negative_loan_amount_rejected(self):
        from src.models import LoanApplication

        with pytest.raises(ValidationError, match="greater than 0"):
            LoanApplication(**_valid_application(
                loan=_sample_loan(amount_requested=Decimal("-100.00")),
            ))

    def test_duplicate_financial_years_rejected(self):
        from src.models import LoanApplication

        with pytest.raises(ValidationError, match="Duplicate financial years"):
            LoanApplication(**_valid_application(
                financials=[_sample_financials(2024), _sample_financials(2024)],
            ))

    def test_secured_loan_without_security_value_rejected(self):
        from src.models import LoanApplication

        with pytest.raises(ValidationError, match="security_value"):
            LoanApplication(**_valid_application(
                loan=_sample_loan(security_type="property", security_value=None),
            ))

    def test_company_number_not_8_digits_rejected(self):
        from src.models import LoanApplication

        with pytest.raises(ValidationError, match="company_number"):
            LoanApplication(**_valid_application(
                applicant=_sample_applicant(company_number="1234"),
            ))


# ---------------------------------------------------------------------------
# Test 3 -- Serialisation round-trip
# ---------------------------------------------------------------------------

def test_loan_application_serialization_roundtrip():
    """model_dump(mode='json') -> reconstruct -> fields match."""
    from src.models import LoanApplication

    original = LoanApplication(**_valid_application())
    data = original.model_dump(mode="json")

    # Reconstruct from plain dict (simulates JSON deserialisation)
    restored = LoanApplication(**data)

    assert restored.application_id == original.application_id
    assert restored.applicant.company_name == original.applicant.company_name
    assert restored.loan.amount_requested == original.loan.amount_requested
    assert len(restored.financials) == len(original.financials)
    assert restored.financials[0].year == original.financials[0].year
    assert restored.financials[0].revenue == original.financials[0].revenue


# ---------------------------------------------------------------------------
# Test 4 -- OrchestratorState reducer accumulation via LangGraph
# ---------------------------------------------------------------------------

def test_orchestrator_state_reducer_accumulation():
    """LangGraph StateGraph compiles with OrchestratorState and reducers accumulate."""
    from langgraph.graph import StateGraph, START, END
    from src.state import OrchestratorState

    def node_a(state):
        return {
            "current_stage": "analysis",
            "audit_trail": [{"stage": "analysis", "action": "started"}],
            "errors": [],
        }

    def node_b(state):
        return {
            "current_stage": "review",
            "audit_trail": [{"stage": "review", "action": "started"}],
            "errors": [],
        }

    graph = StateGraph(OrchestratorState)
    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    compiled = graph.compile()

    result = compiled.invoke({
        "request_id": "TEST-001",
        "audit_trail": [],
        "grounding_scores": [],
        "errors": [],
        "retrieved_documents": [],
    })

    # Reducer (operator.add) should have accumulated both entries
    assert len(result["audit_trail"]) == 2
    assert result["audit_trail"][0]["stage"] == "analysis"
    assert result["audit_trail"][1]["stage"] == "review"

    # Last-write-wins for current_stage
    assert result["current_stage"] == "review"


# ---------------------------------------------------------------------------
# Test 5 -- ConfigLoader loads all configs
# ---------------------------------------------------------------------------

def test_config_loader_all_configs():
    """ConfigLoader reads and validates all four YAML files."""
    from src.config.settings import ConfigLoader

    config_dir = Path(__file__).resolve().parent.parent / "config"
    loader = ConfigLoader(config_dir=config_dir)

    # Agents
    agents = loader.agents()
    assert agents.analyst.role
    assert agents.reviewer.role
    assert agents.compliance.role

    # Guardrails
    guardrails = loader.guardrails()
    assert guardrails.grounding.threshold == 0.7
    assert guardrails.pii.enabled is True

    # Scoring
    scoring = loader.scoring()
    assert len(scoring.stress_test.scenarios) == 5
    assert scoring.credit_scoring.weights.profit_margin == 0.20

    # App
    app_cfg = loader.app()
    assert app_cfg.app.name
    assert app_cfg.providers.weaviate_url


# ---------------------------------------------------------------------------
# Test 6 -- Config validation rejects invalid
# ---------------------------------------------------------------------------

def test_config_validation_rejects_invalid():
    """AgentConfig rejects temperature outside [0.0, 2.0]."""
    from src.config.settings import AgentConfig

    with pytest.raises(ValidationError, match="temperature"):
        AgentConfig(
            role="test",
            goal="test",
            backstory="test",
            temperature=5.0,
        )


# ---------------------------------------------------------------------------
# Test 7 -- All framework imports
# ---------------------------------------------------------------------------

def test_all_framework_imports():
    """All core framework packages are importable."""
    import langgraph
    import crewai
    import autogen_agentchat
    import llama_index.core
    import weaviate
    import langfuse
    import fastapi

    assert langgraph is not None
    assert crewai is not None
    assert autogen_agentchat is not None
    assert llama_index.core is not None
    assert weaviate is not None
    assert langfuse is not None
    assert fastapi is not None
