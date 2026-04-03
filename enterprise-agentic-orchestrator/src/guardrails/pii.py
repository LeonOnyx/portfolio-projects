"""PII Detection and Redaction.

Config-driven personally identifiable information scanner that loads
regex patterns from ``guardrails.yaml`` and returns structured
:class:`PIIScanResult` models.  Supports both detect-only (``scan``)
and detect-and-redact (``scan_and_redact``) workflows so the same
detector can guard both pipeline inputs and outputs.

Regulatory driver:
    GOV-04 -- PII must never appear in agent outputs.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.models.governance import PIIScanResult

if TYPE_CHECKING:
    from src.config.settings import ConfigLoader


class PIIDetector:
    """Detect and redact PII using regex patterns from guardrails.yaml.

    Parameters
    ----------
    config : ConfigLoader, optional
        Configuration loader.  When *None* a fresh :class:`ConfigLoader`
        is created, which reads ``config/guardrails.yaml`` by default.

    Notes
    -----
    The "Bank Account" pattern (``\\d{8}``) receives explicit word
    boundaries (``\\b\\d{8}\\b``) at compile time to prevent false
    positives on 8-digit substrings of larger numbers (research
    pitfall 4).  All other patterns are used as-is from config since
    they are already well-scoped.

    Pattern processing order in ``scan_and_redact`` is: NI Number
    first, then Sort Code, then Bank Account.  This avoids the sort
    code's 6 digits being caught by the bank account boundary pattern
    after partial redaction.
    """

    _BANK_ACCOUNT_PATTERN_NAME = "Bank Account"

    def __init__(self, config: "ConfigLoader | None" = None) -> None:
        if config is None:
            from src.config.settings import ConfigLoader

            config = ConfigLoader()

        pii_config = config.guardrails().pii
        self.enabled: bool = pii_config.enabled
        self.redaction_char: str = pii_config.redaction_char

        # Compile regex patterns -- add word boundaries to Bank Account
        self.patterns: list[tuple[str, re.Pattern[str]]] = []
        for p in pii_config.patterns:
            regex = p.regex
            if p.name == self._BANK_ACCOUNT_PATTERN_NAME:
                regex = r"\b" + regex + r"\b"
            self.patterns.append((p.name, re.compile(regex)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, text: str) -> PIIScanResult:
        """Detect PII in *text* without redaction.

        Useful for input scanning where you want to detect PII but
        preserve the original text.
        """
        if not self.enabled:
            return PIIScanResult(
                scanned_text_length=len(text),
                pii_found=False,
            )

        detected_types: list[str] = []
        for name, pattern in self.patterns:
            if pattern.search(text):
                detected_types.append(name)

        return PIIScanResult(
            scanned_text_length=len(text),
            pii_found=bool(detected_types),
            pii_types_detected=detected_types,
        )

    def scan_and_redact(self, text: str) -> PIIScanResult:
        """Detect PII in *text* and replace matches with redaction chars.

        Processing order is controlled to prevent cross-pattern
        interference: NI Number -> Sort Code -> Bank Account -> rest.
        """
        if not self.enabled:
            return PIIScanResult(
                scanned_text_length=len(text),
                pii_found=False,
            )

        detected_types: list[str] = []
        redacted = text

        # Build an ordered list: NI Number first, Sort Code second,
        # Bank Account third, everything else in config order.
        priority_order = ["NI Number", "Sort Code", "Bank Account"]
        ordered_patterns = sorted(
            self.patterns,
            key=lambda p: (
                priority_order.index(p[0]) if p[0] in priority_order else len(priority_order)
            ),
        )

        for name, pattern in ordered_patterns:
            if pattern.search(redacted):
                detected_types.append(name)
                redacted = pattern.sub(
                    lambda m: self.redaction_char * len(m.group()),
                    redacted,
                )

        return PIIScanResult(
            scanned_text_length=len(text),
            pii_found=bool(detected_types),
            pii_types_detected=detected_types,
            redacted_text=redacted if detected_types else None,
        )
