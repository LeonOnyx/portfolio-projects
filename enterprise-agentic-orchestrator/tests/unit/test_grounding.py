"""Unit tests for GroundingChecker (TEST-05).

Tests claim extraction, source text extraction, and grounding verification
with mocked embeddings. Verifies grounded/ungrounded/mixed/edge cases
and config threshold sensitivity.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(threshold: float = 0.7, max_retries: int = 2, limit: float = 0.2):
    """Build a mock ConfigLoader that returns grounding config."""
    grounding = MagicMock()
    grounding.threshold = threshold
    grounding.max_retries = max_retries
    grounding.ungrounded_claim_limit = limit

    guardrails = MagicMock()
    guardrails.grounding = grounding

    config = MagicMock()
    config.guardrails.return_value = guardrails
    return config


def _build_checker(threshold: float = 0.7, limit: float = 0.2):
    """Build a GroundingChecker with mocked config."""
    from src.guardrails.grounding import GroundingChecker

    return GroundingChecker(config=_make_config(threshold=threshold, limit=limit))


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

class TestExtractClaims:
    """Tests for GroundingChecker._extract_claims()."""

    def test_three_sentences(self):
        checker = _build_checker()
        text = "Revenue grew by 20%. Profit margins improved significantly. Cash position is strong."
        claims = checker._extract_claims(text)
        assert len(claims) == 3

    def test_short_fragments_filtered(self):
        checker = _build_checker()
        text = "OK. This is a proper claim with enough length. No."
        claims = checker._extract_claims(text)
        # "OK" (2 chars) and "No" (2 chars) should be filtered
        assert len(claims) == 1
        assert "proper claim" in claims[0]

    def test_empty_text(self):
        checker = _build_checker()
        claims = checker._extract_claims("")
        assert claims == []

    def test_single_long_sentence(self):
        checker = _build_checker()
        text = "The company reported strong financial performance throughout the year"
        claims = checker._extract_claims(text)
        assert len(claims) == 1

    def test_newline_separated(self):
        checker = _build_checker()
        text = "Revenue grew by 20%.\nProfit margins improved significantly."
        claims = checker._extract_claims(text)
        assert len(claims) == 2


# ---------------------------------------------------------------------------
# Source text extraction
# ---------------------------------------------------------------------------

class TestExtractSourceTexts:
    """Tests for GroundingChecker._extract_source_texts()."""

    def test_dict_with_content_key(self):
        checker = _build_checker()
        docs = [{"content": "Some document text"}]
        texts = checker._extract_source_texts(docs)
        assert texts == ["Some document text"]

    def test_dict_with_text_key(self):
        checker = _build_checker()
        docs = [{"text": "Fallback text key"}]
        texts = checker._extract_source_texts(docs)
        assert texts == ["Fallback text key"]

    def test_plain_string(self):
        checker = _build_checker()
        docs = ["plain string document"]
        texts = checker._extract_source_texts(docs)
        assert texts == ["plain string document"]

    def test_mixed_sources(self):
        checker = _build_checker()
        docs = [
            {"content": "From content key"},
            {"text": "From text key"},
            "Plain string",
        ]
        texts = checker._extract_source_texts(docs)
        assert len(texts) == 3
        assert texts[0] == "From content key"
        assert texts[1] == "From text key"
        assert texts[2] == "Plain string"


# ---------------------------------------------------------------------------
# Verification with mocked embeddings
# ---------------------------------------------------------------------------

class TestVerifyGrounded:
    """Tests for GroundingChecker.verify() -- grounded outputs."""

    @patch("src.rag.embeddings.embed_texts")
    def test_fully_grounded(self, mock_embed):
        """All claims match sources with high similarity."""
        checker = _build_checker(threshold=0.7)

        # 2 claims, 2 sources: claim[i] matches source[i]
        claim_vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        source_vecs = np.array([
            [0.95, 0.05, 0.0],  # high sim with claim 0
            [0.05, 0.95, 0.0],  # high sim with claim 1
        ])
        # Normalise to unit vectors
        claim_vecs = claim_vecs / np.linalg.norm(claim_vecs, axis=1, keepdims=True)
        source_vecs = source_vecs / np.linalg.norm(source_vecs, axis=1, keepdims=True)

        mock_embed.side_effect = [claim_vecs.tolist(), source_vecs.tolist()]

        result = checker.verify(
            "Claim one is grounded. Claim two is also grounded.",
            [{"content": "Source one text"}, {"content": "Source two text"}],
            claims=["Claim one is grounded", "Claim two is also grounded"],
        )

        assert result.is_grounded is True
        assert result.grounding_score == 1.0
        assert len(result.grounded_claims) == 2
        assert len(result.ungrounded_claims) == 0

    @patch("src.rag.embeddings.embed_texts")
    def test_fully_ungrounded(self, mock_embed):
        """All claims have low similarity -- orthogonal vectors."""
        checker = _build_checker(threshold=0.7)

        claim_vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        source_vecs = np.array([
            [0.0, 0.0, 1.0],  # orthogonal to both claims
            [0.0, 0.0, 1.0],
        ])

        mock_embed.side_effect = [claim_vecs.tolist(), source_vecs.tolist()]

        result = checker.verify(
            "Ungrounded claim one. Ungrounded claim two.",
            [{"content": "Irrelevant source"}],
            claims=["Ungrounded claim one here", "Ungrounded claim two here"],
        )

        assert result.is_grounded is False
        assert result.grounding_score == 0.0
        assert len(result.ungrounded_claims) == 2

    @patch("src.rag.embeddings.embed_texts")
    def test_mixed_over_limit(self, mock_embed):
        """3 claims, 1 ungrounded = 33% > 20% limit -> not grounded."""
        checker = _build_checker(threshold=0.7, limit=0.2)

        claim_vecs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        source_vecs = np.array([
            [0.99, 0.01, 0.0],   # high sim with claim 0
            [0.01, 0.99, 0.0],   # high sim with claim 1
            [0.5, 0.5, 0.0],     # low sim with claim 2 (no z component)
        ])
        claim_vecs = claim_vecs / np.linalg.norm(claim_vecs, axis=1, keepdims=True)
        source_vecs = source_vecs / np.linalg.norm(source_vecs, axis=1, keepdims=True)

        mock_embed.side_effect = [claim_vecs.tolist(), source_vecs.tolist()]

        result = checker.verify(
            "dummy text",
            [{"content": "s1"}, {"content": "s2"}, {"content": "s3"}],
            claims=[
                "First claim is grounded here",
                "Second claim is grounded here",
                "Third claim is ungrounded here",
            ],
        )

        assert result.is_grounded is False
        assert len(result.ungrounded_claims) == 1
        assert len(result.grounded_claims) == 2

    @patch("src.rag.embeddings.embed_texts")
    def test_mixed_at_limit(self, mock_embed):
        """5 claims, 1 ungrounded = 20% = limit -> is_grounded=True (<=)."""
        checker = _build_checker(threshold=0.7, limit=0.2)

        # Build 5 claim vectors and 5 source vectors
        # Claims 0-3 grounded, claim 4 ungrounded
        claim_vecs = np.eye(5)  # 5 orthogonal unit vectors
        source_vecs = np.zeros((5, 5))
        for i in range(4):
            source_vecs[i] = claim_vecs[i] * 0.99  # high sim for first 4
        source_vecs[4] = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # low sim with claim 4

        claim_vecs = claim_vecs / np.linalg.norm(claim_vecs, axis=1, keepdims=True)
        source_vecs = source_vecs / np.linalg.norm(source_vecs, axis=1, keepdims=True)

        mock_embed.side_effect = [claim_vecs.tolist(), source_vecs.tolist()]

        claims = [
            f"Claim number {i} has enough content" for i in range(5)
        ]
        sources = [{"content": f"Source {i}"} for i in range(5)]

        result = checker.verify("dummy", sources, claims=claims)

        # 1/5 = 20% ungrounded == limit of 0.2, so should pass
        assert result.is_grounded is True
        assert len(result.ungrounded_claims) == 1
        assert len(result.grounded_claims) == 4


class TestVerifyEdgeCases:
    """Edge cases for verify()."""

    def test_no_claims(self):
        checker = _build_checker()
        result = checker.verify("Hi", [{"content": "source"}], claims=[])
        assert result.is_grounded is False
        assert result.grounding_score == 0.0

    def test_no_sources(self):
        checker = _build_checker()
        result = checker.verify(
            "This is a sufficiently long claim text.",
            [],
            claims=["This is a sufficiently long claim text"],
        )
        assert result.is_grounded is False
        assert result.grounding_score == 0.0


class TestThresholdFromConfig:
    """Verify that threshold from config is respected."""

    @patch("src.rag.embeddings.embed_texts")
    def test_lower_threshold_passes(self, mock_embed):
        """With threshold=0.3, low-similarity vectors pass."""
        checker = _build_checker(threshold=0.3, limit=0.2)

        # Vectors with ~0.5 cosine similarity
        claim_vecs = np.array([[1.0, 0.0, 0.0]])
        source_vecs = np.array([[0.5, 0.866, 0.0]])  # ~60 degrees, cos ~0.5
        claim_vecs = claim_vecs / np.linalg.norm(claim_vecs, axis=1, keepdims=True)
        source_vecs = source_vecs / np.linalg.norm(source_vecs, axis=1, keepdims=True)

        mock_embed.side_effect = [claim_vecs.tolist(), source_vecs.tolist()]

        result = checker.verify(
            "dummy",
            [{"content": "source text"}],
            claims=["This claim would fail at 0.7 threshold"],
        )

        # cos sim ~0.5 > threshold 0.3 -> grounded
        assert result.is_grounded is True
        assert len(result.grounded_claims) == 1

    @patch("src.rag.embeddings.embed_texts")
    def test_high_threshold_fails(self, mock_embed):
        """Same vectors fail at threshold=0.9."""
        checker = _build_checker(threshold=0.9, limit=0.2)

        claim_vecs = np.array([[1.0, 0.0, 0.0]])
        source_vecs = np.array([[0.5, 0.866, 0.0]])
        claim_vecs = claim_vecs / np.linalg.norm(claim_vecs, axis=1, keepdims=True)
        source_vecs = source_vecs / np.linalg.norm(source_vecs, axis=1, keepdims=True)

        mock_embed.side_effect = [claim_vecs.tolist(), source_vecs.tolist()]

        result = checker.verify(
            "dummy",
            [{"content": "source text"}],
            claims=["This claim would fail at 0.9 threshold"],
        )

        # cos sim ~0.5 < threshold 0.9 -> ungrounded
        assert result.is_grounded is False
        assert len(result.ungrounded_claims) == 1
