"""Bias Detection for Lending Decisions.

Config-driven bias checker that scans agent-generated text for
references to protected characteristics (Equality Act 2010) and
known proxy variables.  Lending decisions must not reference these
terms -- their presence signals potential discriminatory reasoning.

Regulatory driver:
    FCA PRIN 2.1 / Equality Act 2010 -- fair lending compliance.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from src.models.governance import BiasCheckResult

if TYPE_CHECKING:
    from src.config.settings import ConfigLoader


class BiasChecker:
    """Detect bias indicators in text using terms from guardrails.yaml.

    Parameters
    ----------
    config : ConfigLoader, optional
        Configuration loader.  When *None* a fresh :class:`ConfigLoader`
        is created, which reads ``config/guardrails.yaml`` by default.

    Notes
    -----
    Matching strategy:

    * **Single-word terms** (e.g. "age", "gender") use ``\\b``
      word-boundary regex to avoid false positives -- "postage"
      should not trigger on "age".
    * **Multi-word terms** (e.g. "sexual orientation", "first name")
      use simple ``in`` substring matching on lowercased text, since
      word boundaries at the phrase level are unnecessary.
    """

    def __init__(self, config: "ConfigLoader | None" = None) -> None:
        if config is None:
            from src.config.settings import ConfigLoader

            config = ConfigLoader()

        bias_config = config.guardrails().bias
        self.enabled: bool = bias_config.enabled
        self.protected_characteristics: list[str] = bias_config.protected_characteristics
        self.proxy_variables: list[str] = bias_config.proxy_variables

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, text: str) -> BiasCheckResult:
        """Scan *text* for protected characteristics and proxy variables.

        Returns a :class:`BiasCheckResult` indicating whether any bias
        terms were detected and which ones.
        """
        if not self.enabled:
            return BiasCheckResult(
                checked_text_length=len(text),
                bias_detected=False,
            )

        text_lower = text.lower()

        protected_found = [
            term
            for term in self.protected_characteristics
            if self._term_matches(term, text_lower)
        ]
        proxy_found = [
            term
            for term in self.proxy_variables
            if self._term_matches(term, text_lower)
        ]

        bias_detected = bool(protected_found or proxy_found)
        details = ""
        if bias_detected:
            parts: list[str] = []
            if protected_found:
                parts.append(
                    f"Protected characteristics: {', '.join(protected_found)}"
                )
            if proxy_found:
                parts.append(f"Proxy variables: {', '.join(proxy_found)}")
            details = ". ".join(parts)

        return BiasCheckResult(
            checked_text_length=len(text),
            bias_detected=bias_detected,
            protected_characteristics_found=protected_found,
            proxy_variables_found=proxy_found,
            details=details,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _term_matches(term: str, text_lower: str) -> bool:
        """Check whether *term* appears in *text_lower*.

        Single-word terms use word-boundary regex to prevent false
        positives (e.g. "age" must not match "postage").  Multi-word
        terms use plain substring matching.
        """
        if " " in term:
            # Multi-word: simple substring on lowercased text
            return term.lower() in text_lower
        # Single-word: word-boundary regex
        return bool(re.search(r"\b" + re.escape(term.lower()) + r"\b", text_lower))
