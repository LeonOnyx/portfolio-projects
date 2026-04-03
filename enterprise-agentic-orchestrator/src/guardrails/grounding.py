"""
Grounding Verification
=======================
Ensures that agent outputs are faithful to retrieved source documents.
In regulated industries, ungrounded AI outputs are not just wrong -- they're
a compliance risk. This module verifies that every claim in an agent's
response can be traced back to a specific source document.

Verification approach:
    1. Extract claims from agent output via sentence splitting.
    2. Batch-embed all claims and all source texts (exactly 2 API calls).
    3. Compute pairwise cosine similarity (dot product on unit-normalized
       OpenAI embeddings).
    4. Flag claims below the per-claim threshold (default 0.7).
    5. If >20 % of claims are ungrounded, trigger re-prompting (up to 2 retries).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class GroundingChecker:
    """Verifies that agent outputs are grounded in retrieved source documents.

    This is the critical governance control for RAG-based systems in
    regulated environments.  Every claim in an agent's output must be
    traceable to a specific source document.  Ungrounded claims are
    flagged and can trigger escalation to human review.

    Configuration is loaded from ``config/guardrails.yaml`` via
    :class:`~src.config.settings.ConfigLoader`.  Key settings:

    * ``threshold`` -- per-claim embedding similarity cutoff (default 0.7)
    * ``max_retries`` -- re-prompt attempts when too many claims fail (default 2)
    * ``ungrounded_claim_limit`` -- maximum ratio of ungrounded claims (default 0.2)
    """

    def __init__(self, config=None) -> None:
        from src.config.settings import ConfigLoader

        if config is None:
            config = ConfigLoader()

        grounding_cfg = config.guardrails().grounding
        self.threshold: float = grounding_cfg.threshold
        self.max_retries: int = grounding_cfg.max_retries
        self.ungrounded_claim_limit: float = grounding_cfg.ungrounded_claim_limit

        logger.info(
            "GroundingChecker initialised (threshold: %.2f, max_retries: %d, "
            "ungrounded_claim_limit: %.2f)",
            self.threshold,
            self.max_retries,
            self.ungrounded_claim_limit,
        )

    # ------------------------------------------------------------------
    # Claim extraction
    # ------------------------------------------------------------------

    def _extract_claims(self, text: str) -> list[str]:
        """Extract individual verifiable claims from output text.

        Splits on ". " (period-space) and ".\\n" (period-newline), then
        filters out fragments shorter than 10 characters.
        """
        # Split on period followed by space or newline
        raw_sentences = re.split(r"\.\s", text)
        claims: list[str] = []
        for sentence in raw_sentences:
            cleaned = sentence.strip().rstrip(".")
            if len(cleaned) >= 10:
                claims.append(cleaned)
        return claims

    # ------------------------------------------------------------------
    # Batched embedding similarity
    # ------------------------------------------------------------------

    def _compute_similarity_batch(
        self,
        claims: list[str],
        source_texts: list[str],
    ) -> "np.ndarray":
        """Compute pairwise cosine similarity between claims and sources.

        Uses exactly **2** ``embed_texts()`` API calls:
        one for all claims, one for all source texts.

        OpenAI embeddings are unit-normalised, so the dot product equals
        cosine similarity.

        Returns
        -------
        np.ndarray
            Similarity matrix of shape ``(len(claims), len(source_texts))``.
        """
        import numpy as np
        from src.rag.embeddings import embed_texts

        claim_embeddings = np.array(embed_texts(claims))
        source_embeddings = np.array(embed_texts(source_texts))

        # Dot product on unit-normalised vectors == cosine similarity
        similarity_matrix: np.ndarray = claim_embeddings @ source_embeddings.T
        return similarity_matrix

    # ------------------------------------------------------------------
    # Core verification
    # ------------------------------------------------------------------

    def verify(
        self,
        output_text: str,
        source_documents: list,
        claims: list[str] | None = None,
    ):
        """Verify that the output is grounded in the source documents.

        Parameters
        ----------
        output_text:
            The agent's output text to verify.
        source_documents:
            Retrieved documents used as context.  Each element may be a
            ``dict`` (with a ``"content"`` or ``"text"`` key) or a plain
            ``str``.
        claims:
            Optional pre-extracted claims.  When *None*, claims are
            extracted from *output_text* via sentence splitting.

        Returns
        -------
        GroundingResult
            Pydantic model with grounded/ungrounded claims and source
            citations.
        """
        from src.models.governance import GroundingResult, SourceCitation

        # Extract claims if not provided
        if claims is None:
            claims = self._extract_claims(output_text)

        # Extract plain-text content from source documents
        source_texts = self._extract_source_texts(source_documents)

        # Edge case: nothing to verify
        if not claims or not source_texts:
            logger.warning(
                "No %s provided -- output is ungrounded by default",
                "claims" if not claims else "source documents",
            )
            return GroundingResult(
                is_grounded=False,
                grounding_score=0.0,
                grounded_claims=[],
                ungrounded_claims=[
                    {"claim": c, "best_score": 0.0} for c in (claims or [output_text])
                ],
                source_citations=[],
                verification_method="embedding_similarity",
            )

        # Compute similarity matrix (2 API calls)
        similarity_matrix = self._compute_similarity_batch(claims, source_texts)

        import numpy as np

        grounded: list[dict] = []
        ungrounded: list[dict] = []
        citations: list[SourceCitation] = []

        for i, claim in enumerate(claims):
            best_idx = int(np.argmax(similarity_matrix[i]))
            best_score = float(similarity_matrix[i, best_idx])

            if best_score >= self.threshold:
                grounded.append({"claim": claim, "score": best_score})
                # Build SourceCitation from the matching source document
                source_doc = source_documents[best_idx]
                citations.append(
                    SourceCitation(
                        document_id=self._get_doc_field(source_doc, "document_id", "unknown"),
                        document_type=self._get_doc_field(source_doc, "source_collection", "unknown"),
                        chunk_text=source_texts[best_idx][:200],
                        relevance_score=best_score,
                    )
                )
            else:
                ungrounded.append({"claim": claim, "best_score": best_score})

        total = len(claims)
        grounding_score = len(grounded) / total
        ungrounded_ratio = len(ungrounded) / total
        is_grounded = ungrounded_ratio <= self.ungrounded_claim_limit

        result = GroundingResult(
            is_grounded=is_grounded,
            grounding_score=grounding_score,
            grounded_claims=grounded,
            ungrounded_claims=ungrounded,
            source_citations=citations,
            verification_method="embedding_similarity",
        )

        if not result.is_grounded:
            logger.warning(
                "Output failed grounding check (score: %.2f, ungrounded: %.0f%%). "
                "%d/%d claims ungrounded.",
                grounding_score,
                ungrounded_ratio * 100,
                len(ungrounded),
                total,
            )
        else:
            logger.info(
                "Output passed grounding check (score: %.2f, ungrounded: %.0f%%)",
                grounding_score,
                ungrounded_ratio * 100,
            )

        return result

    # ------------------------------------------------------------------
    # Retry logic
    # ------------------------------------------------------------------

    def verify_with_retry(
        self,
        output_text: str,
        source_documents: list,
        reprompt_fn: callable | None = None,
    ) -> tuple:
        """Verify grounding and retry with re-prompting if necessary.

        Parameters
        ----------
        output_text:
            The agent's output text.
        source_documents:
            Source documents for verification.
        reprompt_fn:
            Optional callback ``(ungrounded_claims: list[dict]) -> str``
            that returns a revised output.  Called when >20 % of claims
            are ungrounded.

        Returns
        -------
        tuple[GroundingResult, int]
            The final grounding result and the number of retries used.
        """
        result = self.verify(output_text, source_documents)

        if result.is_grounded or reprompt_fn is None:
            return result, 0

        retries = 0
        current_text = output_text

        for attempt in range(1, self.max_retries + 1):
            logger.info(
                "Grounding retry %d/%d -- %d ungrounded claims",
                attempt,
                self.max_retries,
                len(result.ungrounded_claims),
            )
            current_text = reprompt_fn(result.ungrounded_claims)
            result = self.verify(current_text, source_documents)
            retries = attempt

            if result.is_grounded:
                logger.info("Grounding passed after %d retries", retries)
                break
        else:
            logger.warning(
                "Grounding retries exhausted (%d/%d) -- output still has "
                "%d ungrounded claims",
                retries,
                self.max_retries,
                len(result.ungrounded_claims),
            )

        return result, retries

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_source_texts(source_documents: list) -> list[str]:
        """Extract plain text from source documents.

        Handles dicts with ``"content"`` or ``"text"`` keys, and plain
        strings.
        """
        texts: list[str] = []
        for doc in source_documents:
            if isinstance(doc, dict):
                text = doc.get("content") or doc.get("text") or ""
                texts.append(str(text))
            else:
                texts.append(str(doc))
        return texts

    @staticmethod
    def _get_doc_field(doc, field: str, default: str = "") -> str:
        """Safely extract a field from a source document (dict or str)."""
        if isinstance(doc, dict):
            return str(doc.get(field, default))
        return default
