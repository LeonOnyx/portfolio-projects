"""
Grounding Verification
=======================
Ensures that agent outputs are faithful to retrieved source documents.
In regulated industries, ungrounded AI outputs are not just wrong — they're
a compliance risk. This module verifies that every claim in an agent's
response can be traced back to a specific source document.
"""

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GroundingResult:
    """Result of a grounding verification check."""
    is_grounded: bool
    grounding_score: float  # 0.0 to 1.0
    grounded_claims: list
    ungrounded_claims: list
    source_citations: list
    verification_method: str


class GroundingChecker:
    """Verifies that agent outputs are grounded in retrieved source documents.

    This is the critical governance control for RAG-based systems in
    regulated environments. Every claim in an agent's output must be
    traceable to a specific source document. Ungrounded claims are
    flagged and can trigger escalation to human review.

    Verification methods:
    1. Semantic similarity — compare output claims against source chunks
    2. Entailment checking — use NLI model to verify logical entailment
    3. Citation matching — verify that cited sources actually contain the claimed info
    """

    def __init__(self, config: dict):
        self.threshold = config.get("grounding_threshold", 0.7)
        self.method = config.get("verification_method", "semantic_similarity")
        logger.info("GroundingChecker initialised (threshold: %.2f, method: %s)",
                     self.threshold, self.method)

    def verify(self, output_text: str, source_documents: list,
               claims: list = None) -> GroundingResult:
        """Verify that the output is grounded in the source documents.

        Args:
            output_text: The agent's output text to verify
            source_documents: The retrieved documents used as context
            claims: Optional pre-extracted claims to verify individually

        Returns:
            GroundingResult with detailed verification outcome
        """
        if not source_documents:
            logger.warning("No source documents provided — output is ungrounded by default")
            return GroundingResult(
                is_grounded=False,
                grounding_score=0.0,
                grounded_claims=[],
                ungrounded_claims=[output_text],
                source_citations=[],
                verification_method=self.method,
            )

        # Extract claims if not provided
        if claims is None:
            claims = self._extract_claims(output_text)

        grounded = []
        ungrounded = []
        citations = []

        for claim in claims:
            score, source = self._check_claim(claim, source_documents)
            if score >= self.threshold:
                grounded.append({"claim": claim, "score": score, "source": source})
                citations.append(source)
            else:
                ungrounded.append({"claim": claim, "best_score": score})

        total = len(claims) if claims else 1
        grounding_score = len(grounded) / total if total > 0 else 0.0

        result = GroundingResult(
            is_grounded=grounding_score >= self.threshold,
            grounding_score=grounding_score,
            grounded_claims=grounded,
            ungrounded_claims=ungrounded,
            source_citations=citations,
            verification_method=self.method,
        )

        if not result.is_grounded:
            logger.warning("Output failed grounding check (score: %.2f, threshold: %.2f). "
                           "%d/%d claims ungrounded.",
                           grounding_score, self.threshold,
                           len(ungrounded), total)
        else:
            logger.info("Output passed grounding check (score: %.2f)", grounding_score)

        return result

    def _extract_claims(self, text: str) -> list:
        """Extract individual verifiable claims from output text.

        TODO: Implement claim extraction using:
        - Sentence segmentation
        - Claim classification (factual vs opinion vs procedural)
        - Compound claim decomposition
        """
        # Simple sentence-based extraction for initial implementation
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences

    def _check_claim(self, claim: str, sources: list) -> tuple:
        """Check a single claim against source documents.

        TODO: Implement using:
        - Semantic similarity (embedding cosine distance)
        - NLI entailment model
        - Keyword overlap with TF-IDF weighting

        Returns:
            Tuple of (score, best_matching_source)
        """
        # Placeholder — will be replaced with embedding-based similarity
        best_score = 0.0
        best_source = None

        for source in sources:
            score = self._compute_similarity(claim, source)
            if score > best_score:
                best_score = score
                best_source = source

        return best_score, best_source

    def _compute_similarity(self, claim: str, source: dict) -> float:
        """Compute semantic similarity between a claim and a source document.

        TODO: Replace with actual embedding-based similarity:
        - Generate embeddings for claim and source
        - Compute cosine similarity
        - Return normalised score
        """
        # Placeholder implementation
        source_text = source.get("text", "") if isinstance(source, dict) else str(source)
        claim_words = set(claim.lower().split())
        source_words = set(source_text.lower().split())

        if not claim_words:
            return 0.0

        overlap = claim_words.intersection(source_words)
        return len(overlap) / len(claim_words)
