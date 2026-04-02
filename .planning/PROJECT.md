# Enterprise Agentic Orchestrator

## What This Is

A governed multi-agent orchestration system for financial services credit risk assessment. SME loan applications flow through a pipeline of specialised AI agents — Analyst, Reviewer, and Compliance — each with defined roles, tool access, and governance constraints, producing traceable, explainable, auditable lending recommendations. Built as a reference architecture and portfolio demonstration for regulated industries where compliance is not optional.

## Core Value

Every lending recommendation is explainable to a regulator, grounded against source data, and traceable through an immutable audit trail — governance is architecture, not a bolt-on.

## Requirements

### Validated

- ✓ LangGraph state machine with workflow stages (INTAKE → ANALYSIS → REVIEW → COMPLIANCE → DECISION) — existing
- ✓ Base agent interface with standardised response, validation, and audit entry — existing
- ✓ Grounding checker with claim extraction and similarity scoring — existing (naive implementation)

### Active

- [ ] Pydantic data models for all domain objects (LoanApplication, OrchestratorState, agent reports, governance models)
- [ ] Synthetic data generation — 50 loan applications, financial statements, sector reports, regulatory policies, historical decisions
- [ ] Weaviate vector database setup with 4 collections (FinancialDocuments, SectorAnalysis, RegulatoryPolicies, HistoricalDecisions)
- [ ] Document ingestion pipeline with semantic chunking and metadata attachment
- [ ] LlamaIndex RAG pipeline with hybrid retrieval (vector + BM25) and metadata filtering
- [ ] Credit scorer tool — rule-based scoring (0-100) with factor breakdown
- [ ] Risk calculator tool — PD, LGD, EAD, Expected Loss computation
- [ ] Sector lookup and concentration checker tools
- [ ] Analyst agent (CrewAI) — financial analysis, risk assessment, recommendation with source citations
- [ ] Reviewer agent (CrewAI) — independent validation, 5 stress test scenarios, confidence scoring
- [ ] Compliance agent (AutoGen) — 5 regulatory checks (Consumer Duty, Fair Lending, Risk Appetite, Concentration, Documentation)
- [ ] Embedding-based grounding verification (replace naive word-overlap with semantic similarity)
- [ ] PII detector — scan inputs/outputs, redact NI numbers, bank details, phone, email, DOB
- [ ] Bias checker — flag protected characteristics and proxy variables in agent outputs
- [ ] Immutable audit trail with SHA-256 hashing, full lifecycle logging (30-50 entries per request)
- [ ] Deterministic decision logic following the decision matrix (approve/reject/refer)
- [ ] Escalation rules — high-value loans, agent disagreement, compliance failure, low grounding
- [ ] Configuration management (agents.yaml, guardrails.yaml, scoring.yaml, config.yaml)
- [ ] Langfuse end-to-end tracing with token usage, latency, and cost attribution
- [ ] FastAPI endpoints (POST /assess, GET /decisions, GET /explain, GET /audit, GET /health, GET /metrics)
- [ ] Demo script for single and batch processing
- [ ] Docker Compose for Weaviate + application
- [ ] Unit tests (>= 85% coverage), integration tests, acceptance tests
- [ ] Architecture documentation, governance framework docs, responsible AI principles

### Out of Scope

- Real financial data or credit bureau integrations — synthetic data only
- Production FCA authorisation or regulatory submission — this is a portfolio demo
- Trained ML credit scoring models — using rule-based simulation
- Real banking API integrations — mock/synthetic only
- Mobile or frontend UI — API-only interface
- Multi-tenancy or user management — single-user demo

## Context

- **Author:** Leon Gordon — Principal Data & AI Architect, 5x Microsoft MVP, Oxford Saïd AI Programme
- **Regulatory context:** FCA Consumer Duty (PS22/9), EU AI Act (credit scoring = high-risk), Bank of England SS1/23, PRA Model Risk Management
- **Agent architecture rationale:** Multi-agent maps to FCA's "three lines of defence" — Analyst (Line 1: business), Reviewer (Line 2: risk management), Compliance (Line 3: compliance/audit)
- **Existing code:** ~540 lines across 3 files — LangGraph skeleton (`orchestrator.py`), base agent class (`agents/base.py`), naive grounding checker (`guardrails/grounding.py`). All other modules are empty stubs.
- **Specification:** Full technical specification in `SPECIFICATION.docx` covering data models, agent prompts, RAG pipeline, governance config, test strategy, decision matrix, and API spec.
- **Tech stack:** Python 3.11+, LangGraph, CrewAI, AutoGen, LlamaIndex, Weaviate, Azure OpenAI/Claude, Langfuse, FastAPI, SQLite, Docker

## Constraints

- **Tech stack**: Python 3.11+ with LangGraph orchestration, CrewAI + AutoGen agents, LlamaIndex RAG — specified in SPECIFICATION.docx
- **Data**: All synthetic — no real borrower, company, or financial data at any point
- **LLM**: Configurable provider (Azure OpenAI GPT-4o / Anthropic Claude Sonnet) — enterprise environments mandate specific providers
- **Governance**: Grounding threshold >= 0.7, max 20% ungrounded claims, 7-year audit retention, PII redaction mandatory
- **Testing**: Unit coverage >= 85%, all integration and acceptance tests must pass
- **Deployment**: Docker Compose for local development, deployment guides for Azure

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| LangGraph for orchestration | State machine model maps to lending workflow stages; conditional edges enable governance checkpoints | — Pending |
| CrewAI + AutoGen mix | Demonstrates framework-agnostic architecture; CrewAI suits analyst/reviewer roles, AutoGen suits compliance structured output | — Pending |
| Weaviate for vector DB | Hybrid search (vector + BM25) out of box; metadata filtering critical for financial data | — Pending |
| LlamaIndex for RAG | Superior document processing and semantic chunking for structured financial documents | — Pending |
| SQLite for audit storage | Zero infrastructure for portfolio project; sufficient for demo scale | — Pending |
| Embedding similarity for grounding | Balanced accuracy and speed vs NLI (10x slower) or keyword overlap (too fragile) | — Pending |
| Synthetic data via custom script | Financial data needs realistic distributions and cross-referencing; Faker insufficient | — Pending |

---
*Last updated: 2026-04-02 after initialization*
