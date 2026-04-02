# Enterprise Agentic Orchestrator

**A governed multi-agent system for regulated industries**

> Reference architecture demonstrating enterprise-grade agentic AI orchestration with built-in governance, observability, and responsible AI controls. Designed for industries where compliance isn't optional — financial services, pharma, healthcare.

## The Problem

Most enterprise AI programmes fail not because the models don't work, but because they can't be governed. Autonomous agents making decisions in regulated environments need guardrails, audit trails, and grounded outputs — not just clever prompts.

## What This Project Demonstrates

A multi-agent "Decision Engine" where specialised AI agents collaborate to process complex business workflows (procurement approvals, credit risk assessments, customer query routing) with full governance controls.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Orchestrator                       │
│              (LangGraph State Machine)               │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Analyst   │  │ Reviewer │  │ Compliance       │   │
│  │ Agent     │  │ Agent    │  │ Agent            │   │
│  │(CrewAI)   │  │(CrewAI)  │  │(AutoGen)         │   │
│  └─────┬─────┘  └─────┬────┘  └────────┬─────────┘   │
│        │              │               │              │
│  ┌─────▼──────────────▼───────────────▼──────────┐   │
│  │            Shared Tool Layer                    │   │
│  │  RAG Pipeline (LlamaIndex) │ Vector DB (Weaviate)│  │
│  │  External APIs │ Calculator │ Data Lookup       │   │
│  └───────────────────────────────────────────────┘   │
│                                                      │
│  ┌───────────────────────────────────────────────┐   │
│  │              Governance Layer                   │   │
│  │  Guardrails │ Audit Log │ RLS │ Grounding Check │  │
│  └───────────────────────────────────────────────┘   │
│                                                      │
│  ┌───────────────────────────────────────────────┐   │
│  │           Observability (Langfuse)              │   │
│  │  Traces │ Token Usage │ Latency │ Quality Score │  │
│  └───────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Governed by default** — Every agent decision passes through a compliance checkpoint before execution
2. **Grounded outputs only** — RAG pipeline ensures all responses are traceable to source data (zero hallucination tolerance)
3. **Observable end-to-end** — Langfuse tracing on every agent interaction, tool call, and decision point
4. **Framework-agnostic agents** — Demonstrates LangGraph, CrewAI, and AutoGen working in the same orchestration
5. **Production patterns** — Memory management, retry logic, graceful degradation, and circuit breakers

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | State machine for multi-agent workflow routing |
| **Agent Framework 1** | CrewAI | Role-based agents (Analyst, Reviewer) |
| **Agent Framework 2** | AutoGen | Conversational agents (Compliance checker) |
| **RAG Pipeline** | LlamaIndex | Document ingestion, chunking, retrieval |
| **Vector Database** | Weaviate | Embedding storage and semantic search |
| **LLM** | Azure OpenAI / Claude | Foundation model for agent reasoning |
| **Observability** | Langfuse | Tracing, token tracking, quality scoring |
| **Guardrails** | Custom + NeMo Guardrails | Input/output validation, PII detection, grounding checks |
| **Language** | Python 3.11+ | All components |
| **Cloud** | Azure (primary), GCP docs included | Deployment and infrastructure |

## Project Structure

```
enterprise-agentic-orchestrator/
├── src/
│   ├── orchestrator.py          # LangGraph state machine and workflow definition
│   ├── agents/
│   │   ├── analyst.py           # CrewAI analyst agent — data gathering and analysis
│   │   ├── reviewer.py          # CrewAI reviewer agent — quality and accuracy checks
│   │   ├── compliance.py        # AutoGen compliance agent — regulatory validation
│   │   └── base.py              # Base agent interface and shared behaviours
│   ├── tools/
│   │   ├── rag_tool.py          # LlamaIndex RAG pipeline with Weaviate
│   │   ├── data_lookup.py       # External data source connectors
│   │   ├── calculator.py        # Numerical computation tool
│   │   └── registry.py          # Tool registration and discovery
│   ├── memory/
│   │   ├── conversation.py      # Short-term conversation memory
│   │   ├── episodic.py          # Long-term episodic memory store
│   │   └── shared_state.py      # Cross-agent shared state management
│   ├── guardrails/
│   │   ├── grounding.py         # Output grounding verification
│   │   ├── pii_detector.py      # PII detection and redaction
│   │   ├── input_validator.py   # Input sanitisation and validation
│   │   └── audit_logger.py      # Immutable audit trail
│   └── observability/
│       ├── langfuse_tracer.py   # Langfuse integration for tracing
│       └── metrics.py           # Custom metrics collection
├── config/
│   ├── agents.yaml              # Agent role definitions and permissions
│   ├── guardrails.yaml          # Guardrail rules and thresholds
│   └── config.yaml              # Application configuration
├── tests/
│   ├── test_orchestrator.py     # Orchestration workflow tests
│   ├── test_agents.py           # Individual agent tests
│   ├── test_guardrails.py       # Guardrail validation tests
│   └── test_rag.py              # RAG pipeline accuracy tests
├── docs/
│   ├── architecture.md          # Detailed architecture documentation
│   ├── governance.md            # Governance framework and controls
│   ├── deployment.md            # Deployment guide (Azure + GCP)
│   └── responsible_ai.md        # Responsible AI principles and implementation
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── .env.example
└── README.md
```

## Use Cases

### 1. Procurement Decision Engine
An enterprise procurement request flows through three agents:
- **Analyst Agent** retrieves supplier data via RAG, analyses pricing history, and generates a recommendation
- **Reviewer Agent** validates the analysis, checks for data quality issues, and scores confidence
- **Compliance Agent** verifies the recommendation against procurement policies, spending limits, and regulatory requirements

### 2. Credit Risk Assessment
A loan application is evaluated by the agent pipeline:
- **Analyst Agent** gathers financial data, credit history, and market conditions
- **Reviewer Agent** stress-tests the analysis and flags anomalies
- **Compliance Agent** checks against regulatory lending criteria and fair lending rules

### 3. Customer Query Routing
A complex customer inquiry is triaged and resolved:
- **Analyst Agent** classifies the query and retrieves relevant knowledge base articles
- **Reviewer Agent** validates the proposed response for accuracy
- **Compliance Agent** ensures the response meets regulatory disclosure requirements

## Getting Started

### Prerequisites
- Python 3.11+
- Docker (optional, for containerised deployment)
- API keys for: Azure OpenAI or Anthropic Claude, Weaviate Cloud, Langfuse

### Installation
```bash
git clone https://github.com/yourusername/enterprise-agentic-orchestrator.git
cd enterprise-agentic-orchestrator
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env  # Add your API keys
```

### Quick Start
```bash
# Run the procurement decision engine demo
python -m src.orchestrator --use-case procurement --input "Evaluate supplier bid from Acme Corp for £500k office equipment"

# Run with Langfuse tracing enabled
LANGFUSE_ENABLED=true python -m src.orchestrator --use-case credit-risk --input "Assess loan application: £2M commercial property, LTV 65%"
```

## Governance & Responsible AI

This project implements governance as a first-class architectural concern, not a bolt-on:

- **Grounding verification**: Every agent output is checked against source documents. Ungrounded claims are flagged and rejected.
- **Audit trail**: Immutable log of every agent decision, tool call, and human override.
- **PII protection**: Automatic detection and redaction of personally identifiable information.
- **Explainability**: Every recommendation includes a reasoning trace that can be reviewed by a human.
- **Human-in-the-loop**: Configurable escalation points where human approval is required before proceeding.
- **Token and cost tracking**: Full visibility into LLM usage and cost per decision via Langfuse.

## Roadmap

- [x] Project structure and architecture documentation
- [ ] LangGraph orchestrator with state management
- [ ] CrewAI analyst and reviewer agents
- [ ] AutoGen compliance agent
- [ ] LlamaIndex RAG pipeline with Weaviate
- [ ] Guardrails layer (grounding, PII, input validation)
- [ ] Langfuse observability integration
- [ ] Memory management (conversation + episodic)
- [ ] Procurement use case end-to-end
- [ ] Credit risk use case
- [ ] Docker deployment
- [ ] Azure deployment guide
- [ ] GCP deployment guide
- [ ] Comprehensive test suite

## Author

**Leon Gordon** — Principal Data & AI Architect | 5x Microsoft Data Platform MVP | Oxford Saïd AI Programme

- LinkedIn: [linkedin.com/in/leon-gordon](https://linkedin.com/in/leon-gordon)
- Microsoft Build & Ignite featured customer success story

## License

MIT
