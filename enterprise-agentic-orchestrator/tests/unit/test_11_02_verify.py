"""Verification tests for plan 11-02 Task 2 changes."""

import inspect

import pytest


class TestCreditScorerEmptyFinancials:
    """P1-22: credit scorer rejects empty financials."""

    def test_empty_financials_raises_value_error(self):
        from src.tools.credit_scorer import calculate_credit_score

        with pytest.raises(ValueError, match="at least one year"):
            calculate_credit_score(
                financials=[], years_trading=5, sector_outlook="stable"
            )


class TestSectorLookupImportable:
    """P0-6: sector lookup importable with defensive guard."""

    def test_sector_lookup_importable(self):
        from src.tools.sector_lookup import lookup_sector

        assert callable(lookup_sector)


class TestRagToolsAlphaDefault:
    """P1-21: alpha default is 0.7."""

    def test_rag_financial_lookup_alpha(self):
        from src.tools import rag_tools as rt

        sig = inspect.signature(rt.rag_financial_lookup)
        assert sig.parameters["alpha"].default == 0.7

    def test_rag_sector_analysis_alpha(self):
        from src.tools import rag_tools as rt

        sig = inspect.signature(rt.rag_sector_analysis)
        assert sig.parameters["alpha"].default == 0.7

    def test_rag_policy_lookup_alpha(self):
        from src.tools import rag_tools as rt

        sig = inspect.signature(rt.rag_policy_lookup)
        assert sig.parameters["alpha"].default == 0.7

    def test_historical_comparator_alpha(self):
        from src.tools import rag_tools as rt

        sig = inspect.signature(rt.historical_comparator)
        assert sig.parameters["alpha"].default == 0.7


class TestEscalationThresholdsFromConfig:
    """P0-7: escalation thresholds loaded from config."""

    def test_thresholds_present_in_config(self):
        from src.config.settings import ConfigLoader

        triggers = ConfigLoader().guardrails().escalation.triggers
        threshold_map = {t.name: t.threshold for t in triggers if t.threshold is not None}
        assert "high_value_loan" in threshold_map
        assert threshold_map["high_value_loan"] == 500000
        assert "low_reviewer_confidence" in threshold_map
        assert threshold_map["low_reviewer_confidence"] == 0.5
        assert "low_average_grounding" in threshold_map
        assert threshold_map["low_average_grounding"] == 0.75

    def test_triggers_without_threshold_have_none(self):
        from src.config.settings import ConfigLoader

        triggers = ConfigLoader().guardrails().escalation.triggers
        no_threshold = [t for t in triggers if t.threshold is None]
        names = [t.name for t in no_threshold]
        assert "deteriorating_sector" in names
        assert "compliance_failure" in names
        assert "grounding_failure" in names
