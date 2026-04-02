"""Governance models for the Enterprise Agentic Orchestrator.

Defines Pydantic v2 models for audit trails, grounding verification,
PII scanning, bias checking, and source citation tracking used by
the governance layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Reference to a source document chunk used for grounding."""

    document_id: str
    document_type: str
    chunk_text: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class GroundingResult(BaseModel):
    """Result of grounding verification against source documents."""

    is_grounded: bool
    grounding_score: float = Field(ge=0.0, le=1.0)
    grounded_claims: list[dict] = Field(default_factory=list)
    ungrounded_claims: list[dict] = Field(default_factory=list)
    source_citations: list[SourceCitation] = Field(default_factory=list)
    verification_method: str


class AuditEntry(BaseModel):
    """Immutable audit trail entry for pipeline observability."""

    entry_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    stage: str
    agent: Optional[str] = Field(default=None)
    action: str
    details: dict = Field(default_factory=dict)
    duration_ms: Optional[float] = Field(default=None)
    token_count: Optional[int] = Field(default=None)
    hash: str = Field(default="")


class PIIScanResult(BaseModel):
    """Result of personally identifiable information scanning."""

    scanned_text_length: int
    pii_found: bool
    pii_types_detected: list[str] = Field(default_factory=list)
    redacted_text: Optional[str] = Field(default=None)
    scan_timestamp: datetime = Field(default_factory=datetime.utcnow)


class BiasCheckResult(BaseModel):
    """Result of bias detection in agent-generated text."""

    checked_text_length: int
    bias_detected: bool
    protected_characteristics_found: list[str] = Field(default_factory=list)
    proxy_variables_found: list[str] = Field(default_factory=list)
    details: str = Field(default="")
    check_timestamp: datetime = Field(default_factory=datetime.utcnow)
