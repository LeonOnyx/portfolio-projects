"""Typed configuration loader with Pydantic validation for all YAML config files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Agent config models
# ---------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    role: str
    goal: str
    backstory: str
    llm_model: str = "gpt-4o"
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_iterations: int = Field(default=5, ge=1, le=20)
    memory: bool = False
    permissions: list[str] = Field(default_factory=list)


class AgentsConfig(BaseModel):
    """Top-level agents configuration."""

    analyst: AgentConfig
    reviewer: AgentConfig
    compliance: AgentConfig


# ---------------------------------------------------------------------------
# Guardrails config models
# ---------------------------------------------------------------------------

class GroundingConfig(BaseModel):
    """Grounding verification settings."""

    threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    ungrounded_claim_limit: float = Field(default=0.2, ge=0.0, le=1.0)


class PIIPatternConfig(BaseModel):
    """A single PII detection pattern."""

    name: str
    regex: str


class PIIConfig(BaseModel):
    """PII detection and redaction settings."""

    enabled: bool = True
    patterns: list[PIIPatternConfig]
    redaction_char: str = "*"


class BiasConfig(BaseModel):
    """Bias detection settings."""

    enabled: bool = True
    protected_characteristics: list[str]
    proxy_variables: list[str]


class EscalationTrigger(BaseModel):
    """A single escalation trigger."""

    name: str
    condition: str


class EscalationConfig(BaseModel):
    """Escalation trigger settings."""

    triggers: list[EscalationTrigger]


class GuardrailsConfig(BaseModel):
    """Top-level guardrails configuration."""

    grounding: GroundingConfig
    pii: PIIConfig
    bias: BiasConfig
    escalation: EscalationConfig


# ---------------------------------------------------------------------------
# Scoring config models
# ---------------------------------------------------------------------------

class CreditScoringWeights(BaseModel):
    """Credit scoring factor weights -- must sum to 1.0."""

    profit_margin: float
    debt_to_asset_ratio: float
    revenue_trend: float
    cash_coverage: float
    years_trading: float
    sector_outlook: float
    ccj_history: float
    security_coverage: float

    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> "CreditScoringWeights":
        total = (
            self.profit_margin
            + self.debt_to_asset_ratio
            + self.revenue_trend
            + self.cash_coverage
            + self.years_trading
            + self.sector_outlook
            + self.ccj_history
            + self.security_coverage
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Credit scoring weights must sum to 1.0 (got {total:.4f})"
            )
        return self


class ScoreRange(BaseModel):
    """Valid score range."""

    min: int = 0
    max: int = 100


class CreditScoringConfig(BaseModel):
    """Credit scoring configuration."""

    weights: CreditScoringWeights
    score_range: ScoreRange
    ccj_penalty_per_count: int = 5


class StressScenario(BaseModel):
    """A single stress test scenario."""

    name: str
    description: str
    revenue_shock: float
    cost_increase: float
    default_rate_multiplier: float


class StressTestConfig(BaseModel):
    """Stress test configuration."""

    scenarios: list[StressScenario]


class ConcentrationConfig(BaseModel):
    """Portfolio concentration limits."""

    single_name_limit: float = 0.05
    sector_limit: float = 0.25


class ScoringConfig(BaseModel):
    """Top-level scoring configuration."""

    credit_scoring: CreditScoringConfig
    stress_test: StressTestConfig
    concentration: ConcentrationConfig


# ---------------------------------------------------------------------------
# App config models
# ---------------------------------------------------------------------------

class ProviderConfig(BaseModel):
    """External provider settings."""

    weaviate_url: str = "http://localhost:8080"
    weaviate_grpc_port: int = 50051
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536


class ObservabilityConfig(BaseModel):
    """Observability and tracing settings."""

    langfuse_enabled: bool = True
    trace_all_requests: bool = True
    cost_attribution: bool = True


class ProcessingConfig(BaseModel):
    """Request processing settings."""

    max_concurrent_requests: int = 5
    request_timeout_seconds: int = 300
    llm_retry_attempts: int = 3
    llm_retry_backoff_seconds: int = 2


class AppTopLevel(BaseModel):
    """Core application metadata."""

    name: str
    version: str
    log_level: str


class AppConfig(BaseModel):
    """Top-level application configuration."""

    app: AppTopLevel
    providers: ProviderConfig
    observability: ObservabilityConfig
    processing: ProcessingConfig


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------

class ConfigLoader:
    """Loads and validates YAML configuration files through Pydantic models.

    Each accessor is cached so the YAML file is read and parsed only once.
    """

    def __init__(self, config_dir: Path = Path("config")) -> None:
        self._config_dir = config_dir

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        """Read a YAML file and return the parsed dict."""
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a YAML mapping in {path}, got {type(data)}")
        return data

    @lru_cache(maxsize=1)
    def agents(self) -> AgentsConfig:
        """Load and validate agents.yaml."""
        data = self._load_yaml(self._config_dir / "agents.yaml")
        return AgentsConfig(**data)

    @lru_cache(maxsize=1)
    def guardrails(self) -> GuardrailsConfig:
        """Load and validate guardrails.yaml."""
        data = self._load_yaml(self._config_dir / "guardrails.yaml")
        return GuardrailsConfig(**data)

    @lru_cache(maxsize=1)
    def scoring(self) -> ScoringConfig:
        """Load and validate scoring.yaml."""
        data = self._load_yaml(self._config_dir / "scoring.yaml")
        return ScoringConfig(**data)

    @lru_cache(maxsize=1)
    def app(self) -> AppConfig:
        """Load and validate config.yaml."""
        data = self._load_yaml(self._config_dir / "config.yaml")
        return AppConfig(**data)
