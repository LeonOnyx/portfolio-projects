"""Unit tests for BiasChecker (TEST-07).

Tests bias detection using word boundary matching for single-word terms,
substring matching for multi-word terms, proxy variable detection,
multiple findings, and disabled state.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.guardrails.bias import BiasChecker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def checker(config_loader) -> BiasChecker:
    """BiasChecker using real config patterns."""
    return BiasChecker(config=config_loader)


def _disabled_checker() -> BiasChecker:
    """BiasChecker with enabled=False."""
    bias_config = MagicMock()
    bias_config.enabled = False
    bias_config.protected_characteristics = []
    bias_config.proxy_variables = []

    guardrails = MagicMock()
    guardrails.bias = bias_config

    config = MagicMock()
    config.guardrails.return_value = guardrails
    return BiasChecker(config=config)


# ---------------------------------------------------------------------------
# Clean text
# ---------------------------------------------------------------------------

class TestBiasCleanText:
    """Tests that clean financial text passes without false positives."""

    def test_clean_financial_text(self, checker):
        result = checker.check("Revenue grew by 20% and profit margins improved")
        assert result.bias_detected is False
        assert result.protected_characteristics_found == []
        assert result.proxy_variables_found == []

    def test_clean_analysis_text(self, checker):
        result = checker.check(
            "The company shows strong cash coverage and low debt ratios. "
            "Sector outlook is positive with stable growth trajectory."
        )
        assert result.bias_detected is False


# ---------------------------------------------------------------------------
# Word boundary matching
# ---------------------------------------------------------------------------

class TestBiasWordBoundary:
    """Tests word boundary regex for single-word bias terms."""

    def test_age_detected(self, checker):
        result = checker.check("The applicant's age is 45")
        assert result.bias_detected is True
        assert "age" in result.protected_characteristics_found

    def test_postage_not_detected(self, checker):
        """'age' inside 'postage' must NOT trigger."""
        result = checker.check("The postage was high for this delivery")
        assert result.bias_detected is False

    def test_age_in_percentage_not_detected(self, checker):
        """'age' inside 'percentage' must NOT trigger."""
        result = checker.check("The percentage of returns was low")
        assert result.bias_detected is False

    def test_gender_detected(self, checker):
        result = checker.check("The applicant's gender should not influence the decision")
        assert result.bias_detected is True
        assert "gender" in result.protected_characteristics_found


# ---------------------------------------------------------------------------
# Proxy variable detection
# ---------------------------------------------------------------------------

class TestBiasProxy:
    """Tests for proxy variable detection."""

    def test_postcode_detected(self, checker):
        result = checker.check("The applicant's postcode is SW1A 1AA")
        assert result.bias_detected is True
        assert "postcode" in result.proxy_variables_found

    def test_first_name_detected(self, checker):
        """Multi-word proxy 'first name' uses substring match."""
        result = checker.check("The first name should not affect the lending decision")
        assert result.bias_detected is True
        assert "first name" in result.proxy_variables_found

    def test_nationality_detected(self, checker):
        result = checker.check("The nationality of the borrower was considered")
        assert result.bias_detected is True
        assert "nationality" in result.proxy_variables_found


# ---------------------------------------------------------------------------
# Multiple findings
# ---------------------------------------------------------------------------

class TestBiasMultipleFindings:
    """Tests for text containing multiple bias indicators."""

    def test_protected_and_proxy(self, checker):
        text = "The applicant's age is 45 and postcode is SW1A 1AA"
        result = checker.check(text)
        assert result.bias_detected is True
        assert "age" in result.protected_characteristics_found
        assert "postcode" in result.proxy_variables_found
        assert "Protected characteristics" in result.details
        assert "Proxy variables" in result.details

    def test_multiple_protected(self, checker):
        text = "Consider the age and gender of the applicant"
        result = checker.check(text)
        assert result.bias_detected is True
        assert "age" in result.protected_characteristics_found
        assert "gender" in result.protected_characteristics_found


# ---------------------------------------------------------------------------
# _term_matches() static method
# ---------------------------------------------------------------------------

class TestTermMatches:
    """Direct tests for BiasChecker._term_matches()."""

    def test_single_word_matches(self):
        assert BiasChecker._term_matches("age", "age is 45") is True

    def test_single_word_no_match_in_larger_word(self):
        assert BiasChecker._term_matches("age", "postage stamp") is False

    def test_multi_word_matches(self):
        assert BiasChecker._term_matches(
            "sexual orientation", "sexual orientation disclosure"
        ) is True

    def test_multi_word_substring(self):
        assert BiasChecker._term_matches(
            "first name", "first name should not affect"
        ) is True

    def test_multi_word_no_match(self):
        assert BiasChecker._term_matches(
            "first name", "the first thing we checked was the name"
        ) is False

    def test_case_insensitive(self):
        """_term_matches expects pre-lowered text (check() does the lowering).
        Test that the term itself is lowered before matching."""
        assert BiasChecker._term_matches("Age", "the applicant's age is 30") is True


# ---------------------------------------------------------------------------
# Disabled state
# ---------------------------------------------------------------------------

class TestBiasDisabled:
    """Tests for disabled BiasChecker."""

    def test_disabled_returns_no_bias(self):
        checker = _disabled_checker()
        result = checker.check("The applicant's age is 45")
        assert result.bias_detected is False

    def test_disabled_checked_text_length(self):
        checker = _disabled_checker()
        text = "Some text"
        result = checker.check(text)
        assert result.checked_text_length == len(text)
