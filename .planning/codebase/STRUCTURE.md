# Directory Structure

## Repository Root

```
d:\Onyx Data\Porfolio Projects/
├── .planning/                          # GSD planning documents
│   └── codebase/                       # Codebase analysis (this directory)
├── .vscode/
│   └── settings.json                   # VS Code agent skills config
├── ai-ready-data-platform/             # Project 1: Governed RAG data platform
│   ├── README.md                       # Comprehensive architecture documentation (15 KB)
│   └── src/                            # Source code (empty stubs only)
│       ├── __init__.py
│       ├── api/__init__.py
│       ├── governance/__init__.py
│       ├── ingestion/__init__.py
│       ├── rag/__init__.py
│       └── transforms/__init__.py
└── enterprise-agentic-orchestrator/    # Project 2: Multi-agent orchestrator
    ├── .env.example                    # Environment variable template
    ├── .gitignore
    ├── README.md                       # Architecture and usage documentation (11 KB)
    ├── requirements.txt                # Python dependencies
    └── src/                            # Source code (partially implemented)
        ├── __init__.py
        ├── orchestrator.py             # Core LangGraph workflow (278 lines) ★
        ├── agents/
        │   ├── __init__.py
        │   └── base.py                 # Abstract agent base class (101 lines) ★
        ├── guardrails/
        │   ├── __init__.py
        │   └── grounding.py            # Grounding verification (162 lines) ★
        ├── memory/__init__.py          # Empty stub
        ├── observability/__init__.py   # Empty stub
        └── tools/__init__.py           # Empty stub
```

★ = Has implementation (not just a stub)

---

## Key Locations

### Implemented Code
| File | Lines | Purpose |
|------|-------|---------|
| `enterprise-agentic-orchestrator/src/orchestrator.py` | 278 | LangGraph state machine, workflow nodes, routing |
| `enterprise-agentic-orchestrator/src/agents/base.py` | 101 | BaseAgent abstract class, AgentResponse dataclass |
| `enterprise-agentic-orchestrator/src/guardrails/grounding.py` | 162 | GroundingChecker, claim extraction, similarity scoring |

### Configuration
| File | Purpose |
|------|---------|
| `enterprise-agentic-orchestrator/.env.example` | Environment variable reference |
| `enterprise-agentic-orchestrator/requirements.txt` | Python package dependencies |
| `enterprise-agentic-orchestrator/.gitignore` | Git ignore rules |
| `.vscode/settings.json` | VS Code agent skills paths |

### Documentation
| File | Purpose |
|------|---------|
| `ai-ready-data-platform/README.md` | Full architecture, use cases, implementation guide |
| `enterprise-agentic-orchestrator/README.md` | Architecture, agent patterns, deployment guide |

---

## Naming Conventions

### Files & Directories
- **snake_case** for Python files and directories
- Module directories match domain concepts: `agents/`, `guardrails/`, `memory/`, `tools/`
- Each module has `__init__.py` (most empty)

### Code
- **Classes:** PascalCase (`AgenticOrchestrator`, `BaseAgent`, `GroundingChecker`)
- **Dataclasses:** PascalCase (`OrchestratorState`, `AgentResponse`, `GroundingResult`)
- **Enums:** PascalCase with UPPER_CASE members (`WorkflowStage.INTAKE`)
- **Methods:** snake_case with leading underscore for private (`_build_graph`, `_intake_node`)
- **Constants:** Not explicitly defined (thresholds are constructor params)

---

## Where to Add New Code

| Adding... | Location |
|-----------|----------|
| New agent implementation | `enterprise-agentic-orchestrator/src/agents/{name}.py` |
| New guardrail | `enterprise-agentic-orchestrator/src/guardrails/{name}.py` |
| Agent tools | `enterprise-agentic-orchestrator/src/tools/{name}.py` |
| Memory/persistence | `enterprise-agentic-orchestrator/src/memory/{name}.py` |
| Observability hooks | `enterprise-agentic-orchestrator/src/observability/{name}.py` |
| Data ingestion module | `ai-ready-data-platform/src/ingestion/{name}.py` |
| Data transforms | `ai-ready-data-platform/src/transforms/{name}.py` |
| Governance rules | `ai-ready-data-platform/src/governance/{name}.py` |
| RAG components | `ai-ready-data-platform/src/rag/{name}.py` |
| API endpoints | `ai-ready-data-platform/src/api/{name}.py` |
| Tests | `{project}/tests/` (directory not yet created) |
| Config files | `{project}/config/` (directory not yet created) |

---

## Directories Not Yet Created (Referenced in READMEs)

- `enterprise-agentic-orchestrator/config/` — Agent YAML configs, guardrail rules
- `enterprise-agentic-orchestrator/tests/` — pytest test suite
- `enterprise-agentic-orchestrator/docs/` — Additional documentation
- `ai-ready-data-platform/config/` — Governance rules, access policies
- `ai-ready-data-platform/tests/` — Data quality and integration tests
- `ai-ready-data-platform/dashboards/` — Power BI templates
- `ai-ready-data-platform/notebooks/` — Interactive walkthrough notebooks
- `ai-ready-data-platform/sample_data/` — Test datasets
