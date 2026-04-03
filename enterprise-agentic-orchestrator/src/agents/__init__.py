"""Agents package -- base classes, tool adapters, and factory functions.

Re-exports core agent abstractions and CrewAI tool-set factories for
convenient single-import access throughout the orchestrator.
"""

from src.agents.base import AgentResponse, BaseAgent
from src.agents.tools_adapter import get_analyst_tools, get_reviewer_tools

__all__ = [
    "BaseAgent",
    "AgentResponse",
    "get_analyst_tools",
    "get_reviewer_tools",
]
