# System Architecture

**Enterprise Agentic Orchestrator -- Governed Multi-Agent Credit Risk Assessment**

## Overview

The Enterprise Agentic Orchestrator is a governed multi-agent system for SME (Small and Medium Enterprise) credit risk assessment in UK commercial lending. Loan applications flow through a pipeline of specialised AI agents -- Analyst, Reviewer, and Compliance -- each with defined roles, tool access, and governance constraints, producing traceable, explainable, auditable lending recommendations.

**Core value proposition:** Every lending recommendation is explainable to a regulator, grounded against source data, and traceable through an immutable audit trail. Governance is architecture, not a bolt-on.

**Author:** Leon Gordon -- Principal Data & AI Architect, 5x Microsoft MVP, Oxford Said AI Programme. Enterprise AI for regulated industries.

### Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | LangGraph 1.1.4 | State machine with conditional edges, retry policies |
| Agent Framework (Line 1, 2) | CrewAI | Analyst and Reviewer agents with tool integration |
| Agent Framework (Line 3) | AutoGen 0.4 | Compliance agent with structured Pydantic output |
| RAG Framework | LlamaIndex | Document chunking (SentenceSplitter) |
| Vector Database | Weaviate 1.28 | Hybrid search (vector + BM25) with metadata filtering |
| Embeddings | OpenAI text-embedding-3-small | 1536-dimension embeddings for retrieval and grounding |
| LLM | Azure OpenAI GPT-4o | Agent reasoning (configurable provider) |
| API | FastAPI | REST endpoints with async request handling |
| Tracing | Langfuse v4 | End-to-end observability with nested spans |
| Metrics | prometheus_client | Domain and HTTP-level monitoring |
| Language | Python 3.11+ | Type hints, async/await, Pydantic v2 |

---

## Pipeline Flow

The credit risk assessment pipeline is implemented as a LangGraph `StateGraph` over an `OrchestratorState` TypedDict. The pipeline processes a loan application through 9 nodes with conditional routing at 5 decision points.

### Complete Pipeline

```
START
  |
  v
INTAKE ----[validation error]----> ESCALATE ---> END
  |
  v (valid)
ANALYSIS (Analyst Agent)
  |
  v
GROUNDING_ANALYSIS ---[retry budget]--> ANALYSIS (retry loop)
  |                   |
  v (grounded)        +--[exhausted/circuit-broken]--> ESCALATE ---> END
  |
  v
REVIEW (Reviewer Agent)
  |
  v
GROUNDING_REVIEW ---[retry budget]--> REVIEW (retry loop)
  |                  |
  v (grounded)       +--[exhausted/circuit-broken]--> ESCALATE ---> END
  |
  v
COMPLIANCE (Compliance Agent)
  |
  v
GROUNDING_COMPLIANCE ---[retry budget]--> COMPLIANCE (retry loop)
  |                      |
  v (grounded)           +--[exhausted/circuit-broken]--> ESCALATE ---> END
  |
  v
DECISION ---[escalation triggers]--> ESCALATE ---> END
  |
  v (no triggers)
END
```

### Stage Descriptions

| Stage | Node Function | Purpose | Output |
|-------|--------------|---------|--------|
| INTAKE | `intake_node` | Validate `LoanApplication`, scan for PII, initialise audit trail | `pii_detected`, `audit_trail` entries |
| ANALYSIS | `analysis_node` | Run AnalystAgent (CrewAI) for financial analysis and risk assessment | `analysis_result` dict (AnalysisReport) |
| GROUNDING_ANALYSIS | `grounding_analysis_node` | Verify analyst claims against retrieved source documents | `grounding_scores` entry for `post_analyst` |
| REVIEW | `review_node` | Run ReviewerAgent (CrewAI) for independent validation and stress testing | `review_result` dict (ReviewReport) |
| GROUNDING_REVIEW | `grounding_review_node` | Verify reviewer claims against retrieved source documents | `grounding_scores` entry for `post_reviewer` |
| COMPLIANCE | `compliance_node` | Run ComplianceAgent (AutoGen) for 5 regulatory checks | `compliance_result` dict (ComplianceReport) |
| GROUNDING_COMPLIANCE | `grounding_compliance_node` | Verify compliance citations against retrieved source documents | `grounding_scores` entry for `post_compliance` |
| DECISION | `decision_node` | Apply deterministic decision matrix, evaluate escalation triggers | `final_decision`, `confidence_score`, `reasoning_trace` |
| ESCALATE | `escalation_node` | Terminal node routing to human underwriter review | `requires_escalation=True`, `final_decision="ESCALATED"` |

### Routing Logic

Conditional edges implement 5 routing decisions:

1. **After intake:** Proceed to analysis or escalate (on validation error)
2. **After grounding_analysis:** Proceed to review, retry analysis (budget remaining), or escalate
3. **After grounding_review:** Proceed to compliance, retry review (budget remaining), or escalate
4. **After grounding_compliance:** Proceed to decision, retry compliance (budget remaining), or escalate
5. **After decision:** End pipeline or escalate (if triggers fired)

Grounding retry budget is loaded from `guardrails.yaml::grounding.max_retries` (default: 2). Each grounding checkpoint has an independent retry counter. Circuit-broken checkpoints route directly to escalate without retry.

### Retry Policy

LangGraph `RetryPolicy` is applied to all LLM-calling and embedding-calling nodes (analysis, review, compliance, grounding_analysis, grounding_review, grounding_compliance). Configuration from `config.yaml`:

- `llm_retry_attempts`: 3
- `llm_retry_backoff_seconds`: 2.0
- Backoff factor: 2.0 (exponential)

The RetryPolicy import is version-tolerant -- probes `langgraph.types`, `langgraph.pregel`, and `langgraph.pregel.retry` in order.

---

## Multi-Agent Design

### Three Lines of Defence

The agent architecture maps directly to the FCA's three lines of defence model for financial services governance:

| Line | Agent | Framework | Role | Tools |
|------|-------|-----------|------|-------|
| **Line 1: Business** | Analyst | CrewAI | Financial analysis, credit scoring, risk assessment, recommendation | `rag_financial_lookup`, `rag_sector_analysis`, `credit_scorer`, `risk_calculator`, `historical_comparator` |
| **Line 2: Risk Management** | Reviewer | CrewAI | Independent validation, stress testing, quality scoring, agree/disagree | `rag_financial_lookup`, `rag_sector_analysis`, `rag_policy_lookup`, `stress_tester`, `historical_comparator` |
| **Line 3: Compliance/Audit** | Compliance | AutoGen | 5 regulatory checks (Consumer Duty, Fair Lending, Risk Appetite, Concentration, Documentation) | `rag_policy_lookup`, `concentration_checker` |

### Why Mixed Frameworks

The system uses both CrewAI and AutoGen to demonstrate a **framework-agnostic architecture**:

- **CrewAI** suits the Analyst and Reviewer roles -- these agents perform iterative tool-calling workflows with natural-language reasoning. CrewAI's `Crew` abstraction wraps a sequence of tasks with tool access and structured output parsing.
- **AutoGen 0.4** suits the Compliance role -- this agent produces structured `ComplianceReport` output using `output_content_type` for Pydantic model validation. AutoGen's `AssistantAgent` with `FunctionTool` adapters provides clean structured output with JSON fallback parsing.

### Agent Communication

Agents do **not** communicate directly with each other. All inter-agent communication flows through the LangGraph state:

1. Analyst writes `analysis_result` to state
2. Reviewer reads `analysis_result` from state, writes `review_result`
3. Compliance reads both `analysis_result` and `review_result` from state, writes `compliance_result`
4. Decision node reads all three results from state

This design ensures every inter-agent data flow is observable, auditable, and serialisable.

### Agent Configuration

Agents are configured via `config/agents.yaml`:

- **LLM model:** GPT-4o for all agents
- **Temperature:** 0.1 for Analyst and Reviewer (low creativity), 0.0 for Compliance (deterministic)
- **Max iterations:** 5 for Analyst and Reviewer, 3 for Compliance
- **Memory:** Disabled (stateless per-request)
- **Permissions:** Each agent is restricted to its allowed tool set

### Lazy Imports

All agent, guardrail, governance, model, and config imports in node functions are **lazy** -- performed inside function bodies, not at module scope. This avoids import-time circular chains between the orchestrator and agents (CrewAI/AutoGen pull in heavy dependencies). Only stdlib (`json`, `logging`) and the lightweight `WorkflowStage` enum are imported at module scope.

### Graceful Degradation

Agent wrapper nodes wrap their `execute()` calls in try/except. On exception:
- An error delta is returned (not re-raised)
- The orchestrator's decision matrix handles the missing agent output
- The pipeline continues to the decision node rather than crashing

---

## RAG Pipeline

### Weaviate Collections

Four collections store the knowledge base for agent retrieval:

| Collection | Content | Key Properties |
|-----------|---------|----------------|
| `FinancialDocuments` | Company financial statements, balance sheets, P&L | `company_name`, `sector`, `financial_year`, `sensitivity_level` |
| `SectorAnalysis` | Industry analysis reports for 10 sectors | `sector`, `outlook`, `risk_level`, `title` |
| `RegulatoryPolicies` | FCA, PRA, and internal policy documents | `policy_area`, `regulation_reference`, `effective_date` |
| `HistoricalDecisions` | 200 historical lending decisions with outcomes | `application_id`, `company_name`, `sector`, `loan_amount`, `performance_outcome` |

All collections use `Configure.Vectors.self_provided()` -- embeddings are generated externally via OpenAI and provided at ingestion time.

### Hybrid Retrieval

The retrieval module (`src/rag/retrieval.py`) combines:

- **Vector similarity search** using cosine similarity on OpenAI embeddings
- **BM25 keyword search** for lexical matching
- **Configurable alpha blending:** `0.0` = pure BM25, `1.0` = pure vector, `0.5` = balanced hybrid (default)
- **Metadata filtering** via Weaviate's `Filter.all_of()` for multi-condition AND queries (sector, financial_year, document_type, etc.)

### Chunking Strategy

Documents are chunked using LlamaIndex's `SentenceSplitter`:

- **Chunk size:** 512 tokens
- **Overlap:** 50 tokens
- **Strategy:** Token-aware sentence splitting (not semantic splitting)
- Metadata is preserved from the source document dictionary across all chunks

### Embeddings

- **Model:** OpenAI `text-embedding-3-small`
- **Dimensions:** 1536
- **Consistency:** The same model is used at both ingestion time and query time
- **Batch API:** `embed_texts()` accepts a list of strings and returns a list of embedding vectors

### Agent-Callable RAG Tools

Four tool functions are available to agents, each managing its own Weaviate connection via context manager:

| Tool | Used By | Purpose |
|------|---------|---------|
| `rag_financial_lookup` | Analyst, Reviewer | Retrieve company financial documents |
| `rag_sector_analysis` | Analyst, Reviewer | Retrieve sector analysis reports |
| `rag_policy_lookup` | Reviewer, Compliance | Retrieve regulatory policy documents |
| `historical_comparator` | Analyst, Reviewer | Retrieve similar historical lending decisions |

---

## Domain Tools

### Tool Inventory

| Tool | Module | Purpose | Key Characteristics |
|------|--------|---------|-------------------|
| Credit Scorer | `tools/credit_scorer.py` | Rule-based credit scoring (0-100) | 8 weighted factors, CCJ penalty, Decimal arithmetic |
| Risk Calculator | `tools/risk_calculator.py` | PD, LGD, EAD, Expected Loss | Decimal multiplication for regulatory precision |
| Sector Lookup | `tools/sector_lookup.py` | Current sector outlook from Weaviate | Connects to RAG pipeline |
| Concentration Checker | `tools/concentration_checker.py` | Single-name (<5%) and sector (<25%) limits | Pure function, explicit portfolio parameters |
| Stress Tester | `tools/stress_tester.py` | 5 adverse scenario impact assessment | Revenue shocks, cost increases, default rate multipliers |
| Tool Registry | `tools/registry.py` | Registration and discovery of all tools | Metadata: name, description, callable reference |

### Credit Scoring Weights

From `config/scoring.yaml`:

| Factor | Weight |
|--------|--------|
| Profit margin | 0.20 |
| Debt-to-asset ratio | 0.15 |
| Revenue trend | 0.15 |
| Cash coverage | 0.10 |
| Years trading | 0.10 |
| Sector outlook | 0.10 |
| CCJ history | 0.10 |
| Security coverage | 0.10 |

CCJ penalty: 5 points per CCJ count. Score range: 0-100.

### Stress Test Scenarios

From `config/scoring.yaml`:

| Scenario | Revenue Shock | Cost Increase | Default Rate Multiplier |
|----------|--------------|---------------|------------------------|
| Mild recession | -10% | +5% | 1.5x |
| Severe recession | -25% | +15% | 3.0x |
| Sector shock | -35% | +10% | 2.5x |
| Rate shock | -5% | +25% | 1.8x |
| Combined stress | -40% | +30% | 4.0x |

### Design Principles

- **Pure functions** taking dicts (not Pydantic models) for decoupling
- **Decimal arithmetic** only where regulatory precision matters (Expected Loss multiplication)
- **Float args_schema** with Decimal conversion at the boundary (LLM outputs float, precision applied in `_run()`)
- **Lazy imports** inside `register_all_tools()` and `_run()` to avoid circular import chains

---

## API Layer

### FastAPI Application

The API is served via FastAPI (`src/api/app.py`) with:

- CORS middleware (permissive for demo)
- Prometheus HTTP instrumentation via `prometheus-fastapi-instrumentator`
- Lifespan management for dependency initialisation and Langfuse cleanup

### Endpoints

| Method | Path | Purpose | Status Codes |
|--------|------|---------|-------------|
| `POST` | `/api/v1/assess` | Submit loan application for assessment | 200 (decided), 202 (escalated), 400 (validation error), 500 (internal error) |
| `GET` | `/api/v1/decisions/{request_id}` | Retrieve previous decision | 200, 404 |
| `GET` | `/api/v1/decisions/{request_id}/explain` | Explainability breakdown (per-stage summaries + grounding) | 200, 404 |
| `GET` | `/api/v1/decisions/{request_id}/audit` | Full audit trail export | 200, 404 |
| `GET` | `/api/v1/health` | System health with service availability | 200 |
| `GET` | `/api/v1/metrics` | Prometheus-compatible metrics | 200 |

### Request/Response Models

- **`LoanApplicationRequest`** reuses `ApplicantDetails`, `LoanDetails`, `FinancialSummary` from the domain layer -- no field constraint duplication
- **`build_assessment_response()`** centralises status-code routing so endpoint handlers stay thin
- **`AssessmentStorage`** uses sync file I/O in async wrappers for JSON persistence

### Dependencies

FastAPI dependency injection provides:
- `get_orchestrator()` -- singleton `CreditRiskOrchestrator` instance
- `get_storage()` -- `AssessmentStorage` for JSON file persistence
- `get_langfuse_handler()` -- Langfuse `CallbackHandler` (returns `None` when env vars missing)

---

## Observability

### Langfuse Tracing

End-to-end request tracing with Langfuse v4 SDK:

- **Trace per request:** Each `POST /api/v1/assess` call creates a trace
- **Nested spans:** Agent execution, tool calls, and governance checks are nested under the request trace
- **Agent spans:** `as_type='agent'` for all three agent executions
- **Grounding spans:** `as_type='evaluator'` for grounding checkpoint verification
- **Metrics per span:** Token usage (`usage_details: Dict[str,int]`), cost (`cost_details: Dict[str,float]`), latency
- **Optional integration:** Returns `None` when `LANGFUSE_SECRET_KEY` / `LANGFUSE_PUBLIC_KEY` are not set; tracing never blocks startup or request processing
- **Explicit span wrapping** for all 3 agents (CrewAI and AutoGen have no native Langfuse parameter)

### Prometheus Metrics

Domain-level metrics via `prometheus_client`:

| Metric | Type | Description |
|--------|------|-------------|
| `credit_decision_total` | Counter | Decisions by outcome label (APPROVED, REJECTED, REFERRED_TO_UNDERWRITER, ESCALATED) |
| `grounding_score` | Histogram | Distribution of grounding scores (buckets: 0.0 to 1.0 in 0.1 steps) |
| `credit_escalation_total` | Counter | Total assessments escalated to human review |
| `assessment_duration_seconds` | Histogram | End-to-end latency (buckets: 1s to 300s) |

HTTP-level auto-instrumentation is provided by `prometheus-fastapi-instrumentator`.

---

## Data Layer

### Synthetic Dataset

All data is synthetic -- no real borrower, company, or financial data at any point:

- **50 loan applications** with applicant details, 3 years of financial statements, credit scores, and CCJ history
- **10 sector analysis reports** covering construction, hospitality, retail, technology, manufacturing, healthcare, logistics, professional services, agriculture, and energy
- **Regulatory policy documents** covering FCA Consumer Duty, fair lending guidelines, risk appetite framework, and concentration limits
- **200 historical lending decisions** with realistic outcome distribution (performing, arrears, default, written off)

Data generation uses seeded Faker + Random for reproducibility (UUIDs are non-deterministic by design). Three risk profiles are represented: healthy, stressed, and distressed.

### Assessment Persistence

Completed assessments are persisted as JSON files in `data/assessments/` via `AssessmentStorage`. This enables the GET endpoints to retrieve previous decisions without maintaining an in-process cache.

---

## Deployment

### Docker Compose Architecture

The system runs with `docker-compose up`:

```
+------------------+          +------------------+
|                  |          |                  |
|   app (FastAPI)  |--------->|    weaviate      |
|   Port 8000      |          |    Port 8080     |
|                  |          |    gRPC 50051    |
+------------------+          +------------------+
```

**Weaviate service:**
- Image: `cr.weaviate.io/semitechnologies/weaviate:1.28.0`
- Anonymous access enabled (demo mode)
- Health check: `GET /v1/.well-known/ready` every 5s
- Persistent volume: `weaviate_data`

**Application service:**
- Multi-stage Dockerfile: builder stage installs dependencies, runtime stage runs as non-root user (`appuser`, UID 1000)
- Depends on Weaviate healthy status
- Environment variables for Azure OpenAI, OpenAI API keys, and Langfuse credentials
- Entrypoint script separates Weaviate collection setup from data ingestion

### Entrypoint Flow

1. Wait for Weaviate readiness
2. Create RAG collections (if not exist)
3. Ingest synthetic data (if collections empty)
4. Start FastAPI server via uvicorn

---

## State Management

### OrchestratorState TypedDict

The central state is a `TypedDict` with `total=False` (all fields optional). Nodes return partial dicts (deltas) containing only the fields they update.

**Reducer semantics:**
- `Annotated[list, operator.add]` -- accumulating fields (`audit_trail`, `grounding_scores`, `errors`, `retrieved_documents`). Returned lists are appended to existing state.
- Bare types -- last-write-wins (`final_decision`, `confidence_score`, `current_stage`, etc.)

**Agent output storage:** Plain `dict` instances via `model_dump(mode="json")`. This keeps state fully JSON-serialisable without importing Pydantic models into the state module.

### Workflow Stages

```python
class WorkflowStage(str, Enum):
    INTAKE = "intake"
    ANALYSIS = "analysis"
    GROUNDING_ANALYSIS = "grounding_analysis"
    REVIEW = "review"
    GROUNDING_REVIEW = "grounding_review"
    COMPLIANCE = "compliance"
    GROUNDING_COMPLIANCE = "grounding_compliance"
    DECISION = "decision"
    COMPLETE = "complete"
    ESCALATE = "escalate"
```
