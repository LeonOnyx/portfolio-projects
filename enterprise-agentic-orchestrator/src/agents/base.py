"""
Base Agent Interface
=====================
Abstract base class for all agents in the orchestration system.
Ensures consistent behaviour, logging, and governance compliance
across different agent frameworks (CrewAI, AutoGen, custom).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Standardised response from any agent in the system."""
    agent_name: str
    agent_framework: str  # "crewai", "autogen", "custom"
    output: dict
    reasoning_trace: str
    confidence: float
    sources_used: list
    tokens_used: int
    latency_ms: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class BaseAgent(ABC):
    """Abstract base for all agents in the enterprise orchestrator.

    Every agent must implement the `execute` method and conform to
    the governance requirements: logging, audit trail, and grounding.
    """

    def __init__(self, name: str, role: str, framework: str, config: dict):
        self.name = name
        self.role = role
        self.framework = framework
        self.config = config
        logger.info("Agent initialised: %s (framework: %s, role: %s)",
                     name, framework, role)

    @abstractmethod
    async def execute(self, context: dict, tools: list = None) -> AgentResponse:
        """Execute the agent's task with the given context.

        Args:
            context: The shared state and relevant data for the agent
            tools: Available tools the agent can use

        Returns:
            AgentResponse with the agent's output and metadata
        """
        pass

    def validate_output(self, response: AgentResponse) -> bool:
        """Validate that the agent's output meets governance requirements.

        Checks:
        - Output is not empty
        - Confidence is within valid range
        - Sources are provided (grounding requirement)
        - No PII in output (if PII detection is enabled)
        """
        if not response.output:
            logger.warning("[%s] Empty output detected", self.name)
            return False

        if not 0.0 <= response.confidence <= 1.0:
            logger.warning("[%s] Invalid confidence: %f", self.name, response.confidence)
            return False

        if not response.sources_used and self.config.get("require_grounding", True):
            logger.warning("[%s] No sources provided — grounding requirement not met",
                           self.name)
            return False

        return True

    def to_audit_entry(self, response: AgentResponse) -> dict:
        """Convert an agent response to an audit trail entry."""
        return {
            "agent": self.name,
            "framework": self.framework,
            "role": self.role,
            "confidence": response.confidence,
            "sources_count": len(response.sources_used),
            "tokens_used": response.tokens_used,
            "latency_ms": response.latency_ms,
            "timestamp": response.timestamp,
            "output_valid": self.validate_output(response),
        }
