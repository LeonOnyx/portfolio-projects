"""CrewAI BaseTool adapters for all domain and RAG tools.

Wraps each pure-function tool as a CrewAI BaseTool subclass with a typed
args_schema, enabling seamless integration with CrewAI agents.  Uses lazy
imports inside ``_run()`` to avoid circular dependency chains (per Phase 4
decision on lazy imports).
"""

from __future__ import annotations

import json
from typing import Any, Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------


def _serialize_result(result: Any) -> str:
    """Convert a tool result to a JSON string for agent consumption.

    - Pydantic models: ``model_dump_json()``
    - Dicts/lists: ``json.dumps`` with ``default=str`` for Decimal/datetime
    - Everything else: ``str()``
    """
    if hasattr(result, "model_dump_json"):
        return result.model_dump_json()
    if isinstance(result, (dict, list)):
        return json.dumps(result, default=str)
    return str(result)


# ===========================================================================
# Domain tool adapters
# ===========================================================================


# ---------------------------------------------------------------------------
# CreditScorerTool
# ---------------------------------------------------------------------------


class CreditScorerInput(BaseModel):
    """Input schema for the credit scorer tool."""

    financials: list[dict[str, Any]] = Field(
        description="List of financial year dicts with keys: year, revenue, "
        "profit_margin, debt_to_asset_ratio, cash_balance, total_liabilities"
    )
    years_trading: int = Field(description="Number of years the business has been trading")
    sector_outlook: str = Field(
        description="Sector outlook: positive, stable, cautious, or negative"
    )
    ccj_count: int = Field(default=0, description="Number of County Court Judgments")
    security_value: float = Field(default=0.0, description="Value of collateral offered")
    loan_amount: float = Field(default=0.0, description="Requested loan amount")


class CreditScorerTool(BaseTool):
    """Calculate a credit score from financial metrics and contextual data."""

    name: str = "credit_scorer"
    description: str = (
        "Calculate a 0-100 credit score from financial statements, trading history, "
        "sector outlook, CCJ records, and security coverage. Returns factor breakdown."
    )
    args_schema: type[BaseModel] = CreditScorerInput

    def _run(self, **kwargs: Any) -> str:
        from src.tools.credit_scorer import calculate_credit_score

        result = calculate_credit_score(**kwargs)
        return _serialize_result(result)


# ---------------------------------------------------------------------------
# RiskCalculatorTool
# ---------------------------------------------------------------------------


class RiskCalculatorInput(BaseModel):
    """Input schema for the risk calculator tool."""

    credit_score: int = Field(description="Credit score 0-100 from the credit scorer")
    loan_amount: float = Field(description="Requested loan amount")
    security_value: Optional[float] = Field(
        default=None, description="Value of collateral, or null if unsecured"
    )
    security_type: str = Field(
        default="unsecured", description="Type of security (e.g. property, unsecured)"
    )


class RiskCalculatorTool(BaseTool):
    """Calculate PD, LGD, EAD, and Expected Loss risk metrics."""

    name: str = "risk_calculator"
    description: str = (
        "Calculate probability of default, loss given default, exposure at default, "
        "expected loss, and risk rating from a credit score and loan details."
    )
    args_schema: type[BaseModel] = RiskCalculatorInput

    def _run(self, **kwargs: Any) -> str:
        from decimal import Decimal

        from src.tools.risk_calculator import calculate_risk_metrics

        kwargs["loan_amount"] = Decimal(str(kwargs["loan_amount"]))
        if kwargs.get("security_value") is not None:
            kwargs["security_value"] = Decimal(str(kwargs["security_value"]))
        result = calculate_risk_metrics(**kwargs)
        return _serialize_result(result)


# ---------------------------------------------------------------------------
# StressTesterTool
# ---------------------------------------------------------------------------


class StressTesterInput(BaseModel):
    """Input schema for the stress tester tool."""

    revenue: float = Field(description="Baseline annual revenue")
    total_costs: float = Field(description="Baseline total costs")
    net_profit: float = Field(description="Baseline net profit")
    credit_score: int = Field(description="Credit score 0-100 for PD calculation")


class StressTesterTool(BaseTool):
    """Run stress test scenarios against a borrower's financial position."""

    name: str = "stress_tester"
    description: str = (
        "Run multiple stress scenarios (revenue shocks, cost increases) against "
        "a borrower's financials to assess resilience and survival probability."
    )
    args_schema: type[BaseModel] = StressTesterInput

    def _run(self, **kwargs: Any) -> str:
        from decimal import Decimal

        from src.tools.stress_tester import run_stress_tests

        kwargs["revenue"] = Decimal(str(kwargs["revenue"]))
        kwargs["total_costs"] = Decimal(str(kwargs["total_costs"]))
        kwargs["net_profit"] = Decimal(str(kwargs["net_profit"]))
        result = run_stress_tests(**kwargs)
        return _serialize_result(result)


# ---------------------------------------------------------------------------
# ConcentrationCheckerTool
# ---------------------------------------------------------------------------


class ConcentrationCheckerInput(BaseModel):
    """Input schema for the concentration checker tool."""

    loan_amount: float = Field(description="Proposed loan amount")
    borrower_name: str = Field(description="Name of the borrower")
    sector: str = Field(description="Business sector of the borrower")
    portfolio_total: float = Field(
        description="Current total portfolio value before this loan"
    )
    existing_exposures_by_name: dict[str, float] = Field(
        description="Current exposure per borrower name"
    )
    existing_exposures_by_sector: dict[str, float] = Field(
        description="Current exposure per sector"
    )


class ConcentrationCheckerTool(BaseTool):
    """Check whether a proposed loan breaches portfolio concentration limits."""

    name: str = "concentration_checker"
    description: str = (
        "Validate single-name and sector concentration risk against configured "
        "thresholds. Returns exposure percentages and breach flags."
    )
    args_schema: type[BaseModel] = ConcentrationCheckerInput

    def _run(self, **kwargs: Any) -> str:
        from decimal import Decimal

        from src.tools.concentration_checker import check_concentration

        kwargs["loan_amount"] = Decimal(str(kwargs["loan_amount"]))
        kwargs["portfolio_total"] = Decimal(str(kwargs["portfolio_total"]))
        kwargs["existing_exposures_by_name"] = {
            k: Decimal(str(v)) for k, v in kwargs["existing_exposures_by_name"].items()
        }
        kwargs["existing_exposures_by_sector"] = {
            k: Decimal(str(v)) for k, v in kwargs["existing_exposures_by_sector"].items()
        }
        result = check_concentration(**kwargs)
        return _serialize_result(result)


# ---------------------------------------------------------------------------
# SectorLookupTool
# ---------------------------------------------------------------------------


class SectorLookupInput(BaseModel):
    """Input schema for the sector lookup tool."""

    sector: str = Field(description="Sector to look up (e.g. Technology, Construction)")


class SectorLookupTool(BaseTool):
    """Look up sector outlook and risk assessment via RAG pipeline."""

    name: str = "sector_lookup"
    description: str = (
        "Look up a business sector's current outlook, risk level, and key findings "
        "using the RAG knowledge base. Returns outlook assessment and citations."
    )
    args_schema: type[BaseModel] = SectorLookupInput

    def _run(self, **kwargs: Any) -> str:
        from src.tools.sector_lookup import lookup_sector

        result = lookup_sector(**kwargs)
        return _serialize_result(result)


# ===========================================================================
# RAG tool adapters
# ===========================================================================


# ---------------------------------------------------------------------------
# RAGFinancialLookupTool
# ---------------------------------------------------------------------------


class RAGFinancialLookupInput(BaseModel):
    """Input schema for the RAG financial lookup tool."""

    query: str = Field(description="Natural-language search query")
    sector: Optional[str] = Field(default=None, description="Optional sector filter")
    financial_year: Optional[int] = Field(
        default=None, description="Optional financial year filter"
    )
    company_name: Optional[str] = Field(
        default=None, description="Optional company name filter"
    )
    limit: int = Field(default=5, description="Maximum number of results")
    alpha: float = Field(
        default=0.5,
        description="Hybrid search blend weight (0=keyword, 1=vector, 0.5=balanced)",
    )


class RAGFinancialLookupTool(BaseTool):
    """Search financial documents in the RAG knowledge base."""

    name: str = "rag_financial_lookup"
    description: str = (
        "Search the financial documents knowledge base for company financials, "
        "revenue data, and financial metrics. Supports sector and year filtering."
    )
    args_schema: type[BaseModel] = RAGFinancialLookupInput

    def _run(self, **kwargs: Any) -> str:
        from src.tools.rag_tools import rag_financial_lookup

        result = rag_financial_lookup(**kwargs)
        return _serialize_result(result)


# ---------------------------------------------------------------------------
# RAGSectorAnalysisTool
# ---------------------------------------------------------------------------


class RAGSectorAnalysisInput(BaseModel):
    """Input schema for the RAG sector analysis tool."""

    query: str = Field(description="Natural-language search query")
    sector: Optional[str] = Field(default=None, description="Optional sector filter")
    limit: int = Field(default=5, description="Maximum number of results")
    alpha: float = Field(
        default=0.5,
        description="Hybrid search blend weight (0=keyword, 1=vector, 0.5=balanced)",
    )


class RAGSectorAnalysisTool(BaseTool):
    """Search sector analysis documents in the RAG knowledge base."""

    name: str = "rag_sector_analysis"
    description: str = (
        "Search the sector analysis knowledge base for industry outlooks, "
        "risk assessments, and market trends. Supports sector filtering."
    )
    args_schema: type[BaseModel] = RAGSectorAnalysisInput

    def _run(self, **kwargs: Any) -> str:
        from src.tools.rag_tools import rag_sector_analysis

        result = rag_sector_analysis(**kwargs)
        return _serialize_result(result)


# ---------------------------------------------------------------------------
# HistoricalComparatorTool
# ---------------------------------------------------------------------------


class HistoricalComparatorInput(BaseModel):
    """Input schema for the historical comparator tool."""

    query: str = Field(description="Natural-language search query")
    sector: Optional[str] = Field(default=None, description="Optional sector filter")
    performance_outcome: Optional[str] = Field(
        default=None,
        description="Optional outcome filter (e.g. default, performing)",
    )
    limit: int = Field(default=5, description="Maximum number of results")
    alpha: float = Field(
        default=0.5,
        description="Hybrid search blend weight (0=keyword, 1=vector, 0.5=balanced)",
    )


class HistoricalComparatorTool(BaseTool):
    """Search historical lending decisions in the RAG knowledge base."""

    name: str = "historical_comparator"
    description: str = (
        "Search historical lending decisions for comparable cases, outcomes, "
        "and precedents. Supports sector and performance outcome filtering."
    )
    args_schema: type[BaseModel] = HistoricalComparatorInput

    def _run(self, **kwargs: Any) -> str:
        from src.tools.rag_tools import historical_comparator

        result = historical_comparator(**kwargs)
        return _serialize_result(result)


# ---------------------------------------------------------------------------
# RAGPolicyLookupTool
# ---------------------------------------------------------------------------


class RAGPolicyLookupInput(BaseModel):
    """Input schema for the RAG policy lookup tool."""

    query: str = Field(description="Natural-language search query")
    policy_area: Optional[str] = Field(
        default=None,
        description="Optional policy area filter (e.g. Capital Requirements)",
    )
    limit: int = Field(default=5, description="Maximum number of results")
    alpha: float = Field(
        default=0.5,
        description="Hybrid search blend weight (0=keyword, 1=vector, 0.5=balanced)",
    )


class RAGPolicyLookupTool(BaseTool):
    """Search regulatory policy documents in the RAG knowledge base."""

    name: str = "rag_policy_lookup"
    description: str = (
        "Search regulatory policies, FCA/PRA guidance, and CRR requirements "
        "in the knowledge base. Supports policy area filtering."
    )
    args_schema: type[BaseModel] = RAGPolicyLookupInput

    def _run(self, **kwargs: Any) -> str:
        from src.tools.rag_tools import rag_policy_lookup

        result = rag_policy_lookup(**kwargs)
        return _serialize_result(result)


# ===========================================================================
# Tool-set factory functions
# ===========================================================================


def get_analyst_tools() -> list[BaseTool]:
    """Return the 5 tools required by the credit analyst agent.

    Tools: credit_scorer, risk_calculator, rag_financial_lookup,
    rag_sector_analysis, historical_comparator.
    """
    return [
        CreditScorerTool(),
        RiskCalculatorTool(),
        RAGFinancialLookupTool(),
        RAGSectorAnalysisTool(),
        HistoricalComparatorTool(),
    ]


def get_reviewer_tools() -> list[BaseTool]:
    """Return the 5 tools required by the credit reviewer agent.

    Tools: rag_financial_lookup, rag_sector_analysis, credit_scorer,
    risk_calculator, stress_tester.
    """
    return [
        RAGFinancialLookupTool(),
        RAGSectorAnalysisTool(),
        CreditScorerTool(),
        RiskCalculatorTool(),
        StressTesterTool(),
    ]


# ===========================================================================
# AutoGen FunctionTool adapters (for ComplianceAgent)
# ===========================================================================


def _rag_policy_lookup_for_autogen(
    query: str,
    policy_area: str = "",
) -> str:
    """Search regulatory policy documents for compliance verification.

    Queries the RAG knowledge base for FCA, PRA, and CRR regulatory
    guidance relevant to UK lending compliance checks.

    Args:
        query: Natural-language search query about regulations.
        policy_area: Optional policy area filter (e.g. 'Consumer Duty',
            'Capital Requirements'). Pass empty string for no filter.

    Returns:
        JSON string with matching regulatory policy documents and citations.
    """
    from src.tools.rag_tools import rag_policy_lookup

    result = rag_policy_lookup(
        query=query,
        policy_area=policy_area if policy_area else None,
    )
    return json.dumps(result, default=str)


def _concentration_checker_for_autogen(
    loan_amount: float,
    borrower_name: str,
    sector: str,
    portfolio_total: float,
    existing_exposures_by_name_json: str,
    existing_exposures_by_sector_json: str,
) -> str:
    """Check portfolio concentration limits against regulatory thresholds.

    Validates single-name and sector exposure percentages including the
    proposed loan against CRR Article 395 / Large Exposures limits.

    Args:
        loan_amount: Proposed loan amount as a number.
        borrower_name: Name of the borrower.
        sector: Business sector of the borrower.
        portfolio_total: Current total portfolio value before this loan.
        existing_exposures_by_name_json: JSON string mapping borrower names
            to their current exposure amounts, e.g. '{"Acme Ltd": 500000}'.
        existing_exposures_by_sector_json: JSON string mapping sectors to
            their current exposure amounts, e.g. '{"Technology": 2000000}'.

    Returns:
        JSON string with exposure percentages and breach flags.
    """
    from decimal import Decimal

    from src.tools.concentration_checker import check_concentration

    exposures_by_name = {
        k: Decimal(str(v))
        for k, v in json.loads(existing_exposures_by_name_json).items()
    }
    exposures_by_sector = {
        k: Decimal(str(v))
        for k, v in json.loads(existing_exposures_by_sector_json).items()
    }

    result = check_concentration(
        loan_amount=Decimal(str(loan_amount)),
        borrower_name=borrower_name,
        sector=sector,
        portfolio_total=Decimal(str(portfolio_total)),
        existing_exposures_by_name=exposures_by_name,
        existing_exposures_by_sector=exposures_by_sector,
    )
    return json.dumps(result.model_dump(mode="json"), default=str)


def get_compliance_tools_autogen() -> list:
    """Return AutoGen-compatible FunctionTool instances for the compliance agent.

    Tools: rag_policy_lookup (regulatory policy search),
    concentration_checker (portfolio limit validation).

    All AutoGen imports are lazy (inside this function body) to avoid
    requiring autogen packages at module-level import time.
    """
    from autogen_core.tools import FunctionTool

    return [
        FunctionTool(
            _rag_policy_lookup_for_autogen,
            description=(
                "Search regulatory policies, FCA/PRA guidance, and CRR "
                "requirements in the knowledge base"
            ),
        ),
        FunctionTool(
            _concentration_checker_for_autogen,
            description=(
                "Check portfolio concentration limits against regulatory "
                "thresholds (CRR Article 395 / Large Exposures)"
            ),
        ),
    ]
