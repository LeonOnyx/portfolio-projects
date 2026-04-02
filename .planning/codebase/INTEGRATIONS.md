# External Integrations

## Overview

Both projects define significant external integrations, but most are architectural plans with minimal implementation.

---

## LLM Providers

### Azure OpenAI
- **Status:** Configured in `.env.example`, not yet connected in code
- **Config:** `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`
- **Default model:** gpt-4o
- **Used by:** Both projects (orchestrator agents, RAG response generation)
- **Package:** `openai>=1.12.0`

### Anthropic Claude
- **Status:** API key configured, not yet connected
- **Config:** `ANTHROPIC_API_KEY`
- **Package:** `anthropic>=0.25.0`
- **Used by:** Enterprise orchestrator (alternative LLM provider)

---

## Vector Database

### Weaviate
- **Status:** Client configured, not initialized or connected
- **Config:** `WEAVIATE_URL`, `WEAVIATE_API_KEY`
- **Packages:** `weaviate-client>=4.0.0`, `llama-index-vector-stores-weaviate>=0.2.0`
- **Used by:** Both projects (RAG document retrieval, embedding storage)
- **Integration points:**
  - `enterprise-agentic-orchestrator/src/orchestrator.py` — references `retrieved_documents` in state
  - `ai-ready-data-platform/src/rag/` — planned vector store module

---

## Observability

### Langfuse
- **Status:** Configured in `.env.example`, module stub exists but not connected
- **Config:** `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`, `LANGFUSE_ENABLED`
- **Package:** `langfuse>=2.0.0`
- **Purpose:** LLM call tracing, token usage tracking, latency monitoring
- **Integration point:** `enterprise-agentic-orchestrator/src/observability/__init__.py` (empty)

---

## Data & Storage (AI-Ready Data Platform — Planned)

### Delta Lake
- **Status:** Architectural plan only
- **Purpose:** Medallion architecture (bronze/silver/gold) data storage
- **Integration:** PySpark read/write operations

### Azure Data Factory / Airflow
- **Status:** Architectural plan only
- **Purpose:** Pipeline orchestration and scheduling

### Great Expectations
- **Status:** Architectural plan only
- **Purpose:** Data quality validation rules

### Power BI
- **Status:** Architectural plan only
- **Purpose:** Governance dashboards and data quality reporting

---

## Authentication & Authorization

### Planned but Not Implemented
- **User roles** defined in orchestrator state (`user_role` field) but not enforced
- **Row-Level Security (RLS)** mentioned in data platform architecture
- **PII detection** referenced in orchestrator but not integrated
- No OAuth, SSO, or API key authentication implemented
- No middleware or request validation

---

## Agent Frameworks (Enterprise Orchestrator)

### LangGraph
- **Status:** IMPLEMENTED — core workflow graph in `orchestrator.py`
- **Package:** `langgraph>=0.2.0`
- **Pattern:** State machine with typed state transitions
- **Nodes:** INTAKE → ANALYSIS → REVIEW → COMPLIANCE → DECISION (with ESCALATE branch)

### CrewAI
- **Status:** In requirements, not integrated
- **Package:** `crewai>=0.28.0`
- **Planned use:** Analysis agent (mentioned in orchestrator TODO)

### AutoGen (PyAutoGen)
- **Status:** In requirements, not integrated
- **Package:** `pyautogen>=0.3.0`
- **Planned use:** Compliance checking agent

---

## API Endpoints

No REST API endpoints are defined in either project. The orchestrator exposes only an `async run()` method for programmatic invocation.

---

## Missing Integrations (Referenced but Not Present)

| Integration | Where Referenced | Status |
|-------------|-----------------|--------|
| NeMo Guardrails | README | Not in requirements or code |
| Azure Blob Storage | Data platform README | Not implemented |
| Databricks Connect | Data platform README | Not implemented |
| Redis/caching | Neither | Not present but needed for embeddings |
| Docker/containers | Neither | No Dockerfile or compose files |
| CI/CD | Neither | No pipeline configuration |

---

## Summary

The codebase has a clear integration architecture with Azure OpenAI, Weaviate, and Langfuse as the primary external services. However, only the LangGraph orchestration graph is actually wired up. All other integrations exist as configuration placeholders (`.env.example`) or empty module stubs. No database connections, API endpoints, or authentication flows are implemented.
