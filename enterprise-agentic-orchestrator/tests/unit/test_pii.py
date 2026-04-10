"""Unit tests for PIIDetector (TEST-06).

Tests PII detection and redaction using real config patterns from
guardrails.yaml. Covers NI number, sort code, phone, company number
behaviour, clean text, redaction, pattern ordering, and disabled state.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.guardrails.pii import PIIDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def detector(config_loader) -> PIIDetector:
    """PIIDetector using real config patterns."""
    return PIIDetector(config=config_loader)


def _disabled_detector() -> PIIDetector:
    """PIIDetector with enabled=False."""
    pii_config = MagicMock()
    pii_config.enabled = False
    pii_config.redaction_char = "*"
    pii_config.patterns = []

    guardrails = MagicMock()
    guardrails.pii = pii_config

    config = MagicMock()
    config.guardrails.return_value = guardrails
    return PIIDetector(config=config)


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestPIIScan:
    """Tests for PIIDetector.scan()."""

    def test_detects_ni_number(self, detector):
        result = detector.scan("NI number is AB123456C")
        assert result.pii_found is True
        assert "NI Number" in result.pii_types_detected

    def test_detects_sort_code(self, detector):
        result = detector.scan("Sort code 12-34-56")
        assert result.pii_found is True
        assert "Sort Code" in result.pii_types_detected

    def test_detects_phone_number(self, detector):
        result = detector.scan("Call 07700900123")
        assert result.pii_found is True
        assert "Phone UK" in result.pii_types_detected

    def test_company_number_boundary(self, detector):
        """Company number (8 digits) may match Bank Account pattern.

        The Bank Account pattern uses \\b\\d{8}\\b word boundaries. A
        standalone 8-digit company number like "12345678" will match
        because it IS an 8-digit number bounded by word boundaries.
        This is a known limitation of regex-based PII detection.
        """
        result = detector.scan("Company number 12345678 registered at Companies House")
        # The Bank Account pattern WILL match standalone 8-digit strings
        assert result.pii_found is True
        assert "Bank Account" in result.pii_types_detected

    def test_clean_text(self, detector):
        result = detector.scan("Acme Ltd revenue grew 20%")
        assert result.pii_found is False
        assert result.pii_types_detected == []

    def test_detects_email(self, detector):
        result = detector.scan("Contact us at test@example.com")
        assert result.pii_found is True
        assert "Email" in result.pii_types_detected

    def test_scanned_text_length(self, detector):
        text = "Short text"
        result = detector.scan(text)
        assert result.scanned_text_length == len(text)


# ---------------------------------------------------------------------------
# Redaction tests
# ---------------------------------------------------------------------------

class TestPIIRedaction:
    """Tests for PIIDetector.scan_and_redact()."""

    def test_ni_number_redacted(self, detector):
        text = "NI number is AB123456C in the record"
        result = detector.scan_and_redact(text)
        assert result.pii_found is True
        assert "NI Number" in result.pii_types_detected
        assert result.redacted_text is not None
        # NI number (9 chars) should be replaced with 9 asterisks
        assert "AB123456C" not in result.redacted_text
        assert "*********" in result.redacted_text

    def test_multiple_patterns_redacted(self, detector):
        """Both NI number and sort code are redacted without interference."""
        text = "NI AB123456C and sort 12-34-56 found"
        result = detector.scan_and_redact(text)
        assert result.pii_found is True
        assert "NI Number" in result.pii_types_detected
        assert "Sort Code" in result.pii_types_detected
        assert result.redacted_text is not None
        assert "AB123456C" not in result.redacted_text
        assert "12-34-56" not in result.redacted_text

    def test_clean_text_no_redaction(self, detector):
        result = detector.scan_and_redact("Acme Ltd revenue grew 20%")
        assert result.pii_found is False
        assert result.redacted_text is None


# ---------------------------------------------------------------------------
# Disabled state
# ---------------------------------------------------------------------------

class TestPIIDisabled:
    """Tests for disabled PIIDetector."""

    def test_disabled_returns_no_pii(self):
        detector = _disabled_detector()
        result = detector.scan("NI number is AB123456C")
        assert result.pii_found is False

    def test_disabled_scan_and_redact(self):
        detector = _disabled_detector()
        result = detector.scan_and_redact("NI number is AB123456C")
        assert result.pii_found is False
        assert result.redacted_text is None
