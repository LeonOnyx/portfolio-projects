# Architecture

## System Overview

This repository contains two independent portfolio projects demonstrating enterprise AI patterns:

1. **AI-Ready Data Platform** — Governed RAG data platform with medallion architecture (documentation only)
2. **Enterprise Agentic Orchestrator** — Multi-agent workflow orchestrator using LangGraph (partially implemented)

---

## Project 1: AI-Ready Data Platform

### Pattern: Medallion Architecture + Governed RAG

```
Raw Data → Bronze (raw) → Silver (cleansed) → Gold (curated) → RAG Pipeline → API
                  ↓              ↓               ↓
            Governance Layer (lineage, quality, PII, access control)
```

### Layers (Planned)
1. **Ingestion** (`src/ingestion/`) — Batch and streaming data intake, schema detection
2. **Transforms** (`src/transforms/`) — Bronze→Silver→Gold pipeline, feature store
3. **Governance** (`src/governance/`) — Classification, lineage, quality, PII detection, access control, audit
4. **RAG** (`src/rag/`) — Document processing, embeddings, vector store, retrieval, grounding
5. **API** (`src/api/`) — Query endpoints, governance API, auth

### Implementation Status
- Architecture fully documented in `ai-ready-data-platform/README.md`
- All source modules are empty `__init__.py` stubs
- No runnable code

---

## Project 2: Enterprise Agentic Orchestrator

### Pattern: State Machine Multi-Agent Orchestration

```
Request → INTAKE → ANALYSIS → REVIEW → COMPLIANCE → DECISION → Response
                                ↓            ↓
                           (fail → INTAKE)  (fail → ESCALATE)
```

### Core Architecture

**State Machine (LangGraph)**
- Defined in `enterprise-agentic-orchestrator/src/orchestrator.py`
- `AgenticOrchestrator` class builds a directed graph with typed state
- Each node represents a workflow stage handled by a specialized agent
- Conditional edges route based on agent outputs (confidence scores, compliance status)

**Shared State**
```python
@dataclass
class OrchestratorState:
    # Input
    request_id, use_case, input_text, user_role
    # Agent outputs
    analysis_result, review_result, compliance_result
    # RAG context
    retrieved_documents, grounding_score
    # Governance
    guardrail_checks, audit_trail, pii_detected, requires_human_review
    # Decision
    final_decision, confidence_score, reasoning_trace
    # Control
    current_stage, error
```

### Layers

1. **Orchestration** (`src/orchestrator.py`) — LangGraph workflow graph, state management, routing
2. **Agents** (`src/agents/`) — Base agent class with validation and audit; concrete agents not yet implemented
3. **Guardrails** (`src/guardrails/`) — Grounding verification (naive word-overlap similarity)
4. **Memory** (`src/memory/`) — Empty stub, planned for conversation/context persistence
5. **Observability** (`src/observability/`) — Empty stub, planned for Langfuse integration
6. **Tools** (`src/tools/`) — Empty stub, planned for agent tool registration

### Data Flow

1. Request enters via `async run(request_id, use_case, input_text, user_role)`
2. **Intake** validates input (TODO: PII detection, sanitization)
3. **Analysis** invokes analyst agent (TODO: CrewAI integration)
4. **Review** validates analysis quality (TODO: reviewer agent)
5. **Compliance** checks regulatory requirements (TODO: AutoGen agent, RLS)
6. **Decision** synthesizes final output with confidence score
7. **Escalate** triggered when compliance fails or confidence is low

### Entry Points
- `AgenticOrchestrator.run()` — sole entry point (no REST API, no CLI)
- No `__main__.py` or application bootstrap

### Abstractions

**BaseAgent** (`src/agents/base.py`)
- Abstract base for all agent implementations
- Provides `AgentResponse` dataclass with standardized fields
- Built-in validation: confidence range, source grounding, PII checks
- `to_audit_entry()` for governance trail

**GroundingChecker** (`src/guardrails/grounding.py`)
- Verifies agent outputs against source documents
- Returns `GroundingResult` with scored claims
- Current implementation: naive word-overlap (placeholder for embeddings)

### Cross-Cutting Concerns
- **Audit Trail** — Mutable list in shared state (not persisted)
- **Error Handling** — Minimal; no try/catch in node handlers
- **Configuration** — Environment variables via `.env` (not loaded in code)

---

## Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Agent orchestration | LangGraph state machine | Typed state, conditional routing, visualizable workflows |
| Multi-framework agents | CrewAI + AutoGen | Demonstrate interoperability of agent frameworks |
| RAG framework | LlamaIndex | Mature ecosystem, Weaviate integration |
| Vector store | Weaviate | Cloud-native, hybrid search |
| Data architecture | Medallion (Delta Lake) | Industry standard for governed analytics |
| LLM providers | Azure OpenAI + Claude | Enterprise-grade, multi-provider resilience |

---

## Summary

The repository showcases two complementary AI platform patterns. The orchestrator has a working LangGraph skeleton demonstrating state machine agent coordination, but agent implementations, tool integrations, and production concerns (auth, error handling, persistence) remain as TODOs. The data platform is purely architectural documentation with no implementation.
