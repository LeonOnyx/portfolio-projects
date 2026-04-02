"""Agent-callable tool functions and registry for the Enterprise Agentic Orchestrator.

Re-exports all public tool functions, result models, and the tool registry
for convenient access via ``from src.tools import <name>``.
"""

from src.tools.concentration_checker import ConcentrationResult, check_concentration
from src.tools.credit_scorer import CreditScoreResult, calculate_credit_score
from src.tools.rag_tools import (
    historical_comparator,
    rag_financial_lookup,
    rag_policy_lookup,
    rag_sector_analysis,
)
from src.tools.registry import ToolMetadata, ToolRegistry, register_all_tools, registry
from src.tools.risk_calculator import RiskCalculationResult, calculate_risk_metrics
from src.tools.sector_lookup import SectorLookupResult, lookup_sector
from src.tools.stress_tester import ScenarioResult, StressTestResult, run_stress_tests

__all__ = [
    # Result models
    "ConcentrationResult",
    "CreditScoreResult",
    "RiskCalculationResult",
    "ScenarioResult",
    "SectorLookupResult",
    "StressTestResult",
    # Domain tool functions
    "calculate_credit_score",
    "calculate_risk_metrics",
    "check_concentration",
    "lookup_sector",
    "run_stress_tests",
    # RAG tool functions
    "historical_comparator",
    "rag_financial_lookup",
    "rag_policy_lookup",
    "rag_sector_analysis",
    # Registry
    "ToolMetadata",
    "ToolRegistry",
    "register_all_tools",
    "registry",
]
