# Code Conventions

## Language & Style

### Python Style
- **Version:** Python 3.11+
- **Type hints:** Used consistently on dataclass fields and method signatures
- **Docstrings:** Present on classes and key methods (descriptive, not formal format)
- **Imports:** Standard library first, then third-party, organized by domain
- **Line length:** No explicit config (follows PEP 8 defaults)

### No Linter/Formatter Configuration
- No `.flake8`, `pyproject.toml [tool.ruff]`, `.pylintrc`, or `setup.cfg`
- No `black`, `isort`, or `ruff` in dependencies
- Code appears to follow PEP 8 conventions by convention, not enforcement

---

## Patterns

### Dataclasses for State
All structured data uses Python `@dataclass`:
```python
# enterprise-agentic-orchestrator/src/orchestrator.py
@dataclass
class OrchestratorState:
    request_id: str = ""
    use_case: str = ""
    ...
```

### Abstract Base Classes
Agent framework uses ABC pattern:
```python
# enterprise-agentic-orchestrator/src/agents/base.py
class BaseAgent(ABC):
    @abstractmethod
    async def execute(self, state: dict) -> AgentResponse:
        pass
```

### Enum for Workflow States
```python
# enterprise-agentic-orchestrator/src/orchestrator.py
class WorkflowStage(Enum):
    INTAKE = "intake"
    ANALYSIS = "analysis"
    ...
```

### Async/Await
- Core orchestration is async (`async def run()`, `async def _intake_node()`)
- Agent execution expected to be async (`async def execute()`)

### TODO Comments for Incomplete Work
Extensive use of `# TODO:` comments marking planned implementation:
```python
# TODO: Implement PII detection
# TODO: Add input sanitization
# TODO: Invoke CrewAI analyst agent
```

---

## Error Handling

### Current State
- **Minimal error handling** — no try/except blocks in orchestrator nodes
- `BaseAgent.validate_output()` returns boolean validation without raising
- Orchestrator state has `error` field but no code sets it
- No custom exception classes defined

### Validation Pattern
```python
# enterprise-agentic-orchestrator/src/agents/base.py
def validate_output(self, response: AgentResponse) -> bool:
    if not response.output:
        return False
    if not (0.0 <= response.confidence <= 1.0):
        return False
    if not response.sources_used:
        return False
    return True
```

---

## Naming Patterns

| Element | Convention | Example |
|---------|-----------|---------|
| Classes | PascalCase | `AgenticOrchestrator`, `GroundingResult` |
| Methods (public) | snake_case | `verify()`, `validate_output()` |
| Methods (private) | _snake_case | `_build_graph()`, `_intake_node()` |
| Dataclass fields | snake_case | `grounding_score`, `reasoning_trace` |
| Enum members | UPPER_CASE | `WorkflowStage.COMPLIANCE` |
| Module files | snake_case | `orchestrator.py`, `grounding.py` |
| Directories | snake_case | `guardrails/`, `agents/` |

---

## Project Organization

### Module Structure
Each domain has its own directory with `__init__.py`:
```
src/
├── agents/          # Agent implementations
├── guardrails/      # Safety and validation
├── memory/          # Persistence (planned)
├── observability/   # Monitoring (planned)
└── tools/           # Agent tools (planned)
```

### Configuration
- Environment variables via `.env` (using `python-dotenv`)
- No YAML/TOML config files yet (referenced in README but not created)
- Thresholds passed as constructor parameters with defaults

### Dependencies
- Managed via `requirements.txt` (no `pyproject.toml` or `setup.py`)
- Loose version constraints (`>=` only, no upper bounds)
- No lock file

---

## Summary

The codebase follows standard Python conventions (PEP 8, dataclasses, ABC, async/await) without formal enforcement tooling. Code is clean and well-structured but lacks error handling, configuration management, and the automated tooling (linters, formatters, type checkers) expected in production Python projects.
