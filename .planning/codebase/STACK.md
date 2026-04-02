# Technology Stack

## Languages & Runtimes

| Language | Version | Usage |
|----------|---------|-------|
| Python | 3.11+ | Both projects — core implementation language |

## Project 1: AI-Ready Data Platform

**Location:** `ai-ready-data-platform/`

### Core Frameworks
- **PySpark** (Databricks / Microsoft Fabric) — distributed data processing
- **Delta Lake** — lakehouse storage format (bronze/silver/gold medallion)
- **LlamaIndex** — RAG pipeline framework
- **Great Expectations** — data quality validation

### Cloud & Infrastructure
- **Azure** (primary cloud)
  - Azure Data Factory / Airflow — pipeline orchestration
  - Azure OpenAI — LLM and embeddings
- **Databricks** — alternative compute platform

### Data & AI
- **Weaviate** — vector database for RAG
- **Sentence Transformers** — embedding generation (alternative to Azure OpenAI)
- **Power BI** — visualization and dashboards

### Implementation Status
- **Documentation only** — comprehensive README with architecture design
- All `src/` modules contain empty `__init__.py` stubs
- No dependencies installed (no requirements.txt or pyproject.toml)

---

## Project 2: Enterprise Agentic Orchestrator

**Location:** `enterprise-agentic-orchestrator/`

### Core Frameworks
- **LangGraph** >=0.2.0 — state machine orchestration for multi-agent workflows
- **CrewAI** >=0.28.0 — agent framework (referenced, not yet integrated)
- **PyAutoGen** >=0.3.0 — Microsoft AutoGen agents (referenced, not yet integrated)
- **LlamaIndex** >=0.10.0 — RAG pipeline (not yet integrated)

### Dependencies (`requirements.txt`)

```
langgraph>=0.2.0
crewai>=0.28.0
pyautogen>=0.3.0
llama-index>=0.10.0
llama-index-vector-stores-weaviate>=0.2.0
langfuse>=2.0.0
weaviate-client>=4.0.0
openai>=1.12.0
anthropic>=0.25.0
pydantic>=2.0.0
python-dotenv>=1.0.0
pyyaml>=6.0.0
pytest>=8.0.0
```

### AI / LLM
- **Azure OpenAI** (GPT-4o default deployment)
- **Anthropic Claude** >=0.25.0

### Observability
- **Langfuse** >=2.0.0 — LLM observability platform (configured, not connected)

### Data Validation
- **Pydantic** >=2.0.0 — data modeling and validation

### Testing
- **pytest** >=8.0.0 — test framework (no tests written yet)

### Implementation Status
- 3 files implemented: `orchestrator.py`, `agents/base.py`, `guardrails/grounding.py`
- Modules with empty stubs: `memory/`, `observability/`, `tools/`
- ~78 TODO items in implemented code

---

## Configuration

### Environment Variables (`.env.example`)
```
# LLM Provider
AZURE_OPENAI_API_KEY
AZURE_OPENAI_ENDPOINT
AZURE_OPENAI_DEPLOYMENT=gpt-4o
ANTHROPIC_API_KEY

# Weaviate Vector Database
WEAVIATE_URL
WEAVIATE_API_KEY

# Langfuse Observability
LANGFUSE_PUBLIC_KEY
LANGFUSE_SECRET_KEY
LANGFUSE_HOST=https://cloud.langfuse.com
LANGFUSE_ENABLED

# Application Settings
LOG_LEVEL=INFO
GROUNDING_THRESHOLD=0.7
MAX_AGENT_RETRIES=3
HUMAN_REVIEW_CONFIDENCE_THRESHOLD=0.6
```

### Build & Tooling
- No build system configured (no Makefile, setup.py, pyproject.toml)
- No package manager lock files
- No Docker configuration
- No CI/CD pipeline

---

## Summary

Both projects are Python-based AI/data platforms at early stages of development. The orchestrator project has a working LangGraph skeleton with agent base classes and grounding verification. The data platform project is documentation-only with no implementation. Key technology choices (LangGraph, CrewAI, LlamaIndex, Weaviate, Azure OpenAI) are defined but most integrations remain unimplemented.
