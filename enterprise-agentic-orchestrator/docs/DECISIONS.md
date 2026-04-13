# Architecture Decision Records

**Enterprise Agentic Orchestrator -- Key Technical Decisions Across All Phases**

This document compiles the architecture decision records (ADRs) made during the design and implementation of the Enterprise Agentic Orchestrator. Each decision is documented with the context that prompted it, the chosen approach, the rationale, and the consequences.

---

## ADR-01: LangGraph for Orchestration

**Phase:** 7 (Orchestration)
**Date:** 2026-04-07
**Status:** Accepted

**Context:** The credit risk pipeline requires a state machine that maps to lending workflow stages (INTAKE -> ANALYSIS -> REVIEW -> COMPLIANCE -> DECISION) with conditional edges enabling governance checkpoints, retry loops, and escalation paths. The orchestration layer must support async execution, state accumulation via reducers, and deterministic routing.

**Decision:** Use LangGraph `StateGraph` with a `TypedDict` state schema and conditional edges for pipeline routing.

**Rationale:** LangGraph's state machine model maps directly to lending workflow stages. Conditional edges enable governance checkpoints (grounding verification after each agent) and retry loops. The `Annotated[list, operator.add]` reducer pattern supports accumulating audit trails and grounding scores across nodes. LangGraph's `RetryPolicy` provides transient-error resilience at the framework level. The `ainvoke()` async interface integrates cleanly with FastAPI.

**Consequences:**
- (+) Pipeline flow is declarative and inspectable via graph structure
- (+) Conditional routing enables grounding retry loops without custom loop logic
- (+) State reducers handle accumulation cleanly
- (-) RetryPolicy import location varies between LangGraph versions, requiring a version-tolerant probe chain
- (-) Routing functions must be module-level (not class methods) for zero-fixture unit testability

---

## ADR-02: CrewAI + AutoGen Mixed Framework

**Phase:** 5-6 (Analyst/Reviewer Agents + Compliance Agent)
**Date:** 2026-04-03
**Status:** Accepted

**Context:** The system requires three specialised agents. Two (Analyst and Reviewer) perform iterative tool-calling workflows with natural-language reasoning. One (Compliance) produces structured output against a fixed checklist. Using a single agent framework would be simpler but would not demonstrate framework-agnostic architecture.

**Decision:** Use CrewAI for Analyst and Reviewer agents, AutoGen 0.4 for the Compliance agent.

**Rationale:** CrewAI suits iterative tool-calling workflows -- the Analyst and Reviewer agents need to invoke multiple tools, reason over results, and produce natural-language reports. AutoGen 0.4's `output_content_type` parameter enables structured Pydantic output for the Compliance agent's pass/fail checklist. The mixed approach demonstrates that the orchestration layer (LangGraph) is framework-agnostic -- it treats agents as async callables regardless of their internal implementation. This is architecturally significant for enterprise environments where different teams may prefer different agent frameworks.

**Consequences:**
- (+) Demonstrates framework-agnostic orchestration pattern
- (+) Each framework used where its strengths apply
- (+) Compliance agent gets native Pydantic structured output
- (-) Two agent framework dependencies to maintain
- (-) Different mocking strategies needed in tests (Crew-level for CrewAI, AssistantAgent-level for AutoGen)
- (-) `max_tool_iterations` (not `max_turns`) is the correct AutoGen 0.4 turn-limiting mechanism -- required hands-on validation

---

## ADR-03: Weaviate for Vector Database

**Phase:** 3 (RAG Pipeline)
**Date:** 2026-04-02
**Status:** Accepted

**Context:** The RAG pipeline requires a vector database that supports hybrid search (vector similarity + keyword matching), metadata filtering, and self-provided embeddings. The database must handle 4 distinct collections with different schemas.

**Decision:** Use Weaviate 1.28 with hybrid search and `Configure.Vectors.self_provided()` for externally generated embeddings.

**Rationale:** Weaviate provides hybrid search (vector + BM25) out of the box with configurable alpha blending. Metadata filtering via `Filter.all_of()` is critical for financial data queries (e.g., filtering by sector, financial year, document type). Self-provided vectors allow using OpenAI embeddings at ingestion time without a Weaviate vectorizer module. The Docker image provides a zero-infrastructure local setup.

**Consequences:**
- (+) Hybrid search combines semantic and lexical matching in a single query
- (+) Metadata filtering narrows results by structured properties
- (+) Docker image with healthcheck enables reliable `docker-compose` setup
- (-) `Configure.Vectors.self_provided()` replaces the deprecated `Vectorizer.none()` -- required API exploration
- (-) RAG tool functions must manage their own Weaviate connections via context managers
- (-) RAG modules excluded from test coverage (require running Weaviate instance)

---

## ADR-04: Decimal Arithmetic for Financial Precision

**Phase:** 1 (Foundation) and 4 (Domain Tools)
**Date:** 2026-04-02
**Status:** Accepted

**Context:** Financial calculations (Expected Loss = PD x LGD x EAD) require precision beyond floating-point arithmetic. Regulatory environments demand that monetary computations are reproducible and not subject to floating-point rounding drift.

**Decision:** Use Python `Decimal` with `max_digits`/`decimal_places` constraints on Pydantic model fields. Apply Decimal arithmetic only for Expected Loss multiplication. Use float everywhere else.

**Rationale:** Full Decimal arithmetic throughout the pipeline would be impractical -- LLMs output floats, embedding similarity scores are floats, and most intermediate computations do not require regulatory precision. The compromise is to use Decimal only where it matters: the final Expected Loss calculation and monetary field storage in Pydantic models. `ROUND_HALF_UP` rounding ensures consistency with banking conventions. Agent tool schemas use float parameters with Decimal conversion at the boundary inside `_run()`.

**Consequences:**
- (+) Regulatory-grade precision where it matters
- (+) `model_dump(mode="json")` handles Decimal serialisation
- (+) LLM compatibility maintained (agents see float parameters)
- (-) Boundary conversion (float -> Decimal) adds complexity in tool adapters
- (-) `to_decimal()` utility function needed for consistent conversion

---

## ADR-05: TypedDict with Annotated Reducers for LangGraph State

**Phase:** 1 (Foundation)
**Date:** 2026-04-02
**Status:** Accepted

**Context:** LangGraph state must support both last-write-wins semantics (for agent outputs, decision fields) and accumulation semantics (for audit trail entries, grounding scores, errors). The existing `@dataclass`-based state needed migration.

**Decision:** Use `TypedDict` with `total=False` and `Annotated[list, operator.add]` reducers for accumulating fields.

**Rationale:** `total=False` means every field is optional -- nodes return partial dicts (deltas) containing only the fields they update. The `operator.add` reducer appends returned lists to existing state, enabling multiple nodes to contribute audit entries without overwriting. Plain `dict` storage for Pydantic models (via `model_dump(mode="json")`) keeps the state fully JSON-serialisable without importing Pydantic into the state module.

**Consequences:**
- (+) Nodes return minimal deltas, not full state copies
- (+) Audit trail accumulates across all pipeline stages automatically
- (+) State is JSON-serialisable for persistence and debugging
- (-) Nodes must return flat lists for reducer fields (not nested lists)
- (-) Type checking is less strict than Pydantic model validation

---

## ADR-06: Lazy Imports to Avoid Circular Dependencies

**Phase:** 4 (Domain Tools), 5 (Agents), 7 (Orchestration)
**Date:** 2026-04-03
**Status:** Accepted

**Context:** The agent framework (CrewAI, AutoGen) and domain tools have deep dependency trees that create circular import chains when imported at module scope. For example, `orchestrator_nodes.py` imports agents, which import tools, which import config, which imports models -- circular back to the orchestrator.

**Decision:** All imports of agents, guardrails, governance, models, config, and tool adapters are lazy -- performed inside function bodies, not at module scope. Only stdlib (`json`, `logging`) and lightweight enums are imported at top level.

**Rationale:** Lazy imports keep module import-time cost near zero and avoid circular chains during package initialisation. Each function loads its dependencies on first call. This was validated via AST walk to confirm `orchestrator_nodes.py` has no heavy top-level imports. The `register_all_tools()` function in the tool registry uses the same pattern. Agent constructors use `_get_llm()` lazy factory for LLM instantiation.

**Consequences:**
- (+) No circular import errors during package initialisation
- (+) Modules can be imported independently for testing
- (+) Import cost amortised to first function call
- (-) Import failures surface at runtime, not import time
- (-) Mocking in tests must patch source modules (e.g., `src.config.settings.ConfigLoader`) not consumer modules

---

## ADR-07: Embedding Similarity for Grounding Verification

**Phase:** 6 (Compliance Agent and Governance)
**Date:** 2026-04-03
**Status:** Accepted

**Context:** Agent outputs must be verified against retrieved source documents. Three approaches were considered: keyword overlap (fast but fragile), NLI/natural language inference (accurate but 10x slower), and embedding similarity (balanced).

**Decision:** Use batched embedding similarity with cosine distance for grounding verification.

**Rationale:** Embedding similarity provides a reasonable balance between accuracy and speed. Two API calls per verification (one for claims, one for sources) keeps costs predictable. OpenAI embeddings are unit-normalised, so dot product equals cosine similarity -- no additional normalisation needed. The 0.7 threshold and 20% ungrounded claim limit are configurable via `guardrails.yaml`.

**Consequences:**
- (+) Two API calls per checkpoint (predictable cost)
- (+) Semantic matching captures paraphrasing and synonym use
- (+) Configurable thresholds allow tuning without code changes
- (-) Cannot verify numerical precision (sentence-level semantics only)
- (-) Threshold not empirically calibrated against labelled ground truth
- (-) Requires embedding API availability (circuit breaker mitigates)

---

## ADR-08: SHA-256 Hash Chain for Audit Trail Integrity

**Phase:** 6 (Compliance Agent and Governance)
**Date:** 2026-04-03
**Status:** Accepted

**Context:** The audit trail must be tamper-evident for regulatory submission. FCA/PRA require that lending decisions are traceable and that audit records cannot be modified without detection.

**Decision:** Implement a SHA-256 hash chain where each audit entry's hash is computed from the previous entry's hash and the current entry's immutable fields.

**Rationale:** A hash chain creates a tamper-evident record -- modifying any entry breaks the chain for all subsequent entries. `GENESIS` as the well-known starting hash provides a deterministic root. Only immutable fields (`entry_id`, `timestamp`, `stage`, `action`) are hashed; the `details` dict is excluded because its serialisation order is non-deterministic. `sort_keys=True` in `json.dumps` ensures consistent hash computation. `verify_chain()` enables integrity validation at any time.

**Consequences:**
- (+) Tamper-evident: any modification is detectable
- (+) `export_json()` includes `chain_valid` metadata for regulatory submission
- (+) `GENESIS` starting point is deterministic and well-known
- (-) Hash chain is per-request only (no cross-request linking)
- (-) In-memory storage; production would need persistent storage with 7-year retention
- (-) `details` dict exclusion means supplementary data is not integrity-protected

---

## ADR-09: Fail-Closed Decision Matrix

**Phase:** 7 (Orchestration)
**Date:** 2026-04-07
**Status:** Accepted

**Context:** The decision matrix maps three inputs (analyst recommendation, reviewer agreement, compliance pass) to a lending outcome. The system must handle unknown or unexpected input values safely.

**Decision:** Unknown analyst recommendations route to `REFERRED_TO_UNDERWRITER` as a fail-closed default. The matrix never silently approves or rejects on unexpected input.

**Rationale:** In regulated lending, the safest response to uncertainty is human review, not automated approval or rejection. `REFERRED_TO_UNDERWRITER` routes the case to a qualified underwriter who can evaluate the situation with full context. This is a deliberate "fail safe" (not "fail open") design -- the system prefers false escalation over false confidence.

**Consequences:**
- (+) No risk of automated approval on malformed data
- (+) Unknown states always reach a human reviewer
- (+) Auditable: the decision log records the unknown input and the safe default
- (-) May increase human workload if agents produce unexpected recommendation formats
- (-) Requires human underwriter capacity for escalated volume

---

## ADR-10: Circuit Breaker for Vector DB Availability

**Phase:** 7 (Orchestration)
**Date:** 2026-04-07
**Status:** Accepted

**Context:** Grounding verification requires Weaviate for embedding retrieval. If Weaviate becomes unavailable, the grounding nodes would fail repeatedly, blocking the pipeline. The system needs to degrade gracefully rather than retry indefinitely.

**Decision:** Implement a per-checkpoint circuit breaker with a threshold of 3 consecutive failures. When tripped, return zero-score grounding results without calling the vector DB.

**Rationale:** After 3 consecutive failures, retrying is unlikely to succeed (the issue is systemic, not transient). Circuit-broken results carry `circuit_broken=True` and route to escalation (not retry). Successful calls reset the failure counter. Module-level dict state is appropriate for single-worker deployment; multi-worker would need Redis-backed state (out of scope).

**Consequences:**
- (+) Pipeline completes even when Weaviate is down
- (+) Escalation ensures human review of unverified outputs
- (+) Langfuse spans are created even for circuit-broken calls (dashboard visibility)
- (-) Single-worker scope only (module-level state)
- (-) Threshold (3) is hardcoded; could be made configurable

---

## ADR-11: Config-Driven Governance (YAML-Driven Escalation Triggers)

**Phase:** 7 (Orchestration)
**Date:** 2026-04-07
**Status:** Accepted

**Context:** Escalation triggers need to be extensible without code changes. Business rules around what constitutes a high-risk case evolve over time, and the system should support configuration-driven extension.

**Decision:** Escalation trigger names come from `guardrails.yaml::escalation.triggers`. Each name maps to an evaluator closure in `check_escalation_triggers()`. Adding a new trigger requires only a YAML entry and a matching evaluator function.

**Rationale:** This approach keeps the trigger list in configuration (visible to non-developers, version-controlled alongside other governance settings) while keeping the evaluation logic in Python (testable, type-safe). Per-trigger `try/except` isolation ensures one malformed state field skips that trigger without crashing the pipeline. The `ConfigLoader().guardrails()` typed accessor is lru-cached for single-read efficiency.

**Consequences:**
- (+) New triggers can be added via YAML without touching orchestration code
- (+) Per-trigger isolation prevents cascade failures
- (+) Trigger configuration is co-located with other governance settings
- (-) Each new trigger still requires a matching Python evaluator
- (-) Trigger conditions in YAML are human-readable descriptions, not executable

---

## ADR-12: FastAPI with Langfuse Tracing

**Phase:** 8 (API and Observability)
**Date:** 2026-04-09
**Status:** Accepted

**Context:** The system needs a REST API layer with async request handling, structured request/response models, automatic OpenAPI documentation, and observability. Tracing must capture end-to-end request flow with nested spans for agent execution, tool calls, and governance checks.

**Decision:** Use FastAPI with Langfuse v4 SDK for tracing and prometheus_client for metrics.

**Rationale:** FastAPI provides native async support, Pydantic model validation for request/response schemas, automatic OpenAPI/Swagger generation, and dependency injection. Langfuse v4 provides end-to-end tracing with nested spans, token usage tracking, and cost attribution. The tracing integration is optional -- `create_langfuse_handler()` returns `None` when environment variables are not set, and `traced_orchestrator_run()` falls back to untraced execution. Prometheus metrics are added via `prometheus_client` directly (domain metrics) and `prometheus-fastapi-instrumentator` (HTTP auto-instrumentation).

**Consequences:**
- (+) Async pipeline execution integrates naturally with FastAPI
- (+) Pydantic validation reuses domain models (no field constraint duplication)
- (+) Langfuse provides cost attribution per agent and per request
- (+) Optional tracing -- app starts and serves requests without Langfuse credentials
- (-) Langfuse v4 API differs significantly from v2/v3 (uses `start_observation()` not `client.span()`)
- (-) Neither CrewAI nor AutoGen has native Langfuse integration; explicit span wrapping required

---

## ADR-13: Multi-Stage Dockerfile with Non-Root User

**Phase:** 10 (Deployment and Demo)
**Date:** 2026-04-10
**Status:** Accepted

**Context:** The application container must install Python dependencies, copy application code, and run as a non-root user for security. The image should be as small as practical.

**Decision:** Use a multi-stage Dockerfile: builder stage installs dependencies with `--prefix=/install`, runtime stage copies the installed packages and runs as `appuser` (UID 1000).

**Rationale:** Multi-stage build separates dependency installation (requires pip, compiler headers) from the runtime image (only needs Python and installed packages). Non-root user (`appuser`) prevents privilege escalation in container breakout scenarios. The runtime stage includes `curl` for the entrypoint health check script. Data directories are created and owned by `appuser` before the `USER` directive.

**Consequences:**
- (+) Smaller runtime image (no pip cache, no compiler headers)
- (+) Non-root execution for security
- (+) Entrypoint script separates infrastructure setup from application startup
- (-) Two-stage build adds Dockerfile complexity
- (-) `curl` dependency in runtime stage (for Weaviate readiness check)

---

## ADR-14: Entrypoint Separates Collection Setup from Data Ingestion

**Phase:** 10 (Deployment and Demo)
**Date:** 2026-04-10
**Status:** Accepted

**Context:** When the application container starts, Weaviate must be ready, RAG collections must exist, and synthetic data must be ingested before the API can serve requests. These steps have different failure modes and should be independently retryable.

**Decision:** The Docker entrypoint script runs three sequential phases: (1) wait for Weaviate readiness, (2) create collections if they do not exist, (3) ingest data if collections are empty. Then it starts the FastAPI server.

**Rationale:** Separating infrastructure setup from data ingestion allows the container to start serving health checks even before data ingestion completes. The Weaviate readiness check uses a retry loop with configurable timeout. Collection creation is idempotent (checks existence first). Data ingestion only runs if collections are empty (preventing duplicate ingestion on container restart).

**Consequences:**
- (+) Idempotent startup -- safe to restart the container
- (+) Health endpoint available before full data ingestion
- (+) Clear failure modes for each startup phase
- (-) First startup is slower (data ingestion takes time)
- (-) Container depends on Weaviate being healthy (Docker Compose `depends_on` with health check)

---

## ADR-15: Demo Scripts Call API (Not Orchestrator Directly)

**Phase:** 10 (Deployment and Demo)
**Date:** 2026-04-10
**Status:** Accepted

**Context:** Demo scripts need to process loan applications and display formatted results. They could either import and call the orchestrator directly, or call the REST API endpoints.

**Decision:** Demo scripts call the `POST /api/v1/assess` API endpoint and `GET /api/v1/decisions/{id}/explain` for results, rather than importing the orchestrator Python module directly.

**Rationale:** Calling the API demonstrates the full stack: request validation, Langfuse tracing, Prometheus metrics, orchestrator execution, JSON persistence, and response formatting. It also means the demo scripts work against the Docker Compose deployment without modification. Direct orchestrator calls would bypass the API layer's tracing, metrics, and error handling -- producing results that do not reflect the production execution path.

**Consequences:**
- (+) Demo exercises the full API stack including tracing and metrics
- (+) Works identically against local and Docker Compose deployments
- (+) Response format matches what API consumers would see
- (-) Requires the API server to be running before demo scripts execute
- (-) Network latency overhead (negligible for demo purposes)

---

## Decision Timeline

| Phase | Date | ADRs |
|-------|------|------|
| 1: Foundation and Data Models | 2026-04-02 | ADR-04, ADR-05 |
| 2: Synthetic Data Generation | 2026-04-02 | (no architectural decisions) |
| 3: RAG Pipeline | 2026-04-02 | ADR-03 |
| 4: Domain Tools | 2026-04-03 | ADR-06 |
| 5: Analyst and Reviewer Agents | 2026-04-03 | ADR-02 |
| 6: Compliance Agent and Governance | 2026-04-03 | ADR-02, ADR-07, ADR-08 |
| 7: Orchestration | 2026-04-07 | ADR-01, ADR-09, ADR-10, ADR-11 |
| 8: API and Observability | 2026-04-09 | ADR-12 |
| 9: Testing | 2026-04-10 | (no architectural decisions) |
| 10: Deployment and Demo | 2026-04-10 | ADR-13, ADR-14, ADR-15 |
