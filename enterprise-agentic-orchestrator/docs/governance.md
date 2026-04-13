# Governance Framework

**Enterprise Agentic Orchestrator -- Regulatory Compliance and Governance Controls**

## Regulatory Context

This system operates in the context of UK financial services regulation, where AI-assisted lending decisions must meet specific obligations:

| Regulation | Relevance | System Mapping |
|-----------|-----------|----------------|
| **FCA Consumer Duty (PS22/9)** | Firms must act to deliver good outcomes for retail customers; outcomes must be monitored and evidenced | Explainability reports, grounding verification, audit trails |
| **EU AI Act** | Credit scoring classified as high-risk AI; requires transparency, human oversight, accuracy | Three-agent validation, human-in-the-loop escalation, grounding thresholds |
| **Bank of England SS1/23** | Model risk management for AI/ML; models must be validated, monitored, and documented | Independent reviewer agent, decision matrix auditability, Langfuse tracing |
| **PRA Model Risk Management** | Models used in capital decisions require independent validation and ongoing monitoring | Reviewer agent as Line 2 validation, stress testing, Prometheus metrics |
| **Equality Act 2010 / FCA PRIN 2.1** | Lending decisions must not discriminate on protected characteristics | Bias checker scans all agent outputs for protected terms and proxy variables |

The three-agent architecture directly maps to the **FCA's three lines of defence** model:

- **Line 1 (Business):** Analyst agent -- owns the risk assessment
- **Line 2 (Risk Management):** Reviewer agent -- independently validates Line 1
- **Line 3 (Compliance/Audit):** Compliance agent -- ensures regulatory adherence

---

## Governance Guarantees

### GOV-01: Grounding Verification

**Guarantee:** Every agent output is verified against retrieved source documents before the pipeline proceeds.

**Implementation:**
- `GroundingChecker` (`src/guardrails/grounding.py`) extracts claims via sentence splitting, batch-embeds all claims and source texts (exactly 2 API calls), and computes pairwise cosine similarity
- Per-claim threshold: 0.7 (from `guardrails.yaml::grounding.threshold`)
- Ungrounded claim limit: 20% (from `guardrails.yaml::grounding.ungrounded_claim_limit`)
- Re-prompting: up to 2 retries (from `guardrails.yaml::grounding.max_retries`)
- Three grounding checkpoints: `post_analyst`, `post_reviewer`, `post_compliance`
- Grounding always runs after every agent -- it is never skipped on a "pass/fail" short-circuit

**Enforcement:** Conditional edges in the LangGraph state machine route ungrounded outputs back to the agent for retry or to escalation when retries are exhausted.

### GOV-02: PII Protection

**Guarantee:** Personally identifiable information is detected in inputs and never appears in agent outputs.

**Implementation:**
- `PIIDetector` (`src/guardrails/pii.py`) uses config-driven regex patterns from `guardrails.yaml`
- Detection patterns (6 types):
  - NI Number: `[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]`
  - Sort Code: `\d{2}-\d{2}-\d{2}`
  - Bank Account: `\d{8}` (with word boundaries to prevent false positives)
  - Date of Birth: `\d{2}/\d{2}/\d{4}`
  - Phone UK: `(?:0|\+44)\d{10,11}`
  - Email: standard email regex
- Two modes: `scan()` for detect-only (pipeline inputs) and `scan_and_redact()` for detect-and-replace (agent outputs)
- Ordered pattern processing: NI Number -> Sort Code -> Bank Account to prevent cross-pattern redaction interference
- Redaction character: `*` (configurable)

**Enforcement:** Intake node runs PII scan on every application. PII detection is recorded in the audit trail. The `pii_detected` flag is set in orchestrator state.

### GOV-03: Bias Detection

**Guarantee:** Agent outputs never reference protected characteristics or known proxy variables in lending decisions.

**Implementation:**
- `BiasChecker` (`src/guardrails/bias.py`) scans agent text for terms from `guardrails.yaml`
- Protected characteristics (9 terms from Equality Act 2010):
  - age, gender, race, ethnicity, religion, disability, sexual orientation, marital status, pregnancy
- Proxy variables (5 terms):
  - postcode, first name, school attended, accent, nationality
- Matching strategy:
  - Single-word terms use `\b` word-boundary regex to prevent false positives ("postage" does not trigger "age")
  - Multi-word terms use substring matching on lowercased text

**Enforcement:** Returns `BiasCheckResult` with `bias_detected` flag, lists of found protected characteristics and proxy variables, and human-readable detail string.

### GOV-04: Immutable Audit Trail

**Guarantee:** Every action in the pipeline is recorded in a tamper-evident, hash-chained audit trail suitable for regulatory submission.

**Implementation:**
- `AuditTrail` (`src/governance/audit.py`) records 30-50 entries per lending request
- SHA-256 hash chain: each entry's hash is computed from `previous_hash + entry_id + timestamp + stage + action`
- Chain root: `GENESIS` as the well-known starting hash
- Only immutable fields are hashed (details dict is excluded to avoid non-deterministic serialisation)
- `sort_keys=True` in `json.dumps` for deterministic hash computation
- `verify_chain()` walks the entire chain from GENESIS, recomputing hashes -- any modification breaks all subsequent entries
- `export_json()` produces a JSON string with `request_id`, `entry_count`, `chain_valid` metadata, and all entries -- suitable for 7-year retention and FCA/PRA submission

**Enforcement:** Audit entries are accumulated via the `Annotated[list, operator.add]` reducer in OrchestratorState. Every node (intake, agents, grounding checkpoints, decision, escalation) emits audit trail entries.

### GOV-05: Deterministic Decision Matrix

**Guarantee:** Final lending decisions are produced by a deterministic, auditable matrix -- no LLM calls, no randomness.

**Implementation:**
- `apply_decision_matrix()` (`src/orchestrator_decision.py`) maps three inputs to one of three outcomes:

| Analyst Recommendation | Reviewer Agrees | Compliance Passed | Decision |
|----------------------|----------------|-------------------|----------|
| APPROVE | True | True | APPROVED |
| APPROVE | True | False | REFERRED_TO_UNDERWRITER |
| APPROVE | False | True | REFERRED_TO_UNDERWRITER |
| APPROVE | False | False | REJECTED |
| REJECT | True | True | REJECTED |
| REJECT | True | False | REJECTED |
| REJECT | False | True | REFERRED_TO_UNDERWRITER |
| REJECT | False | False | REJECTED |
| REFER_TO_UNDERWRITER | * | * | REFERRED_TO_UNDERWRITER |

- **Fail-closed default:** Unknown analyst recommendations route to `REFERRED_TO_UNDERWRITER` (human review), never silently approved or rejected
- Confidence score: unweighted average of 4 signals (analyst credit_score/100, reviewer quality_score, compliance pass 1.0/0.0, average grounding score). Missing signals fall back to neutral 0.5.

### GOV-06: Escalation Triggers

**Guarantee:** Cases meeting specific risk criteria are automatically escalated to human underwriter review, overriding the matrix outcome.

**Implementation:**
- `check_escalation_triggers()` (`src/orchestrator_decision.py`) evaluates 6 config-driven triggers from `guardrails.yaml`:

| Trigger | Condition | Threshold |
|---------|-----------|-----------|
| `high_value_loan` | `loan_amount > threshold` | 500,000 |
| `deteriorating_sector` | `sector_outlook contains "deteriorating"` | substring match |
| `compliance_failure` | `compliance_overall_passed == false` | boolean |
| `low_reviewer_confidence` | `reviewer_confidence < threshold` | 0.5 (maps from HIGH=0.9, MEDIUM=0.7, LOW=0.3) |
| `grounding_failure` | `grounding_retries_exhausted` | max_retries reached for any checkpoint |
| `low_average_grounding` | `average_grounding_score < threshold` | 0.75 |

- **Config-driven extension:** Trigger names come from YAML; each maps to an evaluator closure. Adding a new trigger requires only a YAML entry and a matching evaluator function.
- **Per-trigger isolation:** Each trigger check is wrapped in its own try/except so a missing or malformed state field skips that trigger rather than crashing the pipeline.
- **Belt-and-braces:** `escalation_node` re-runs `check_escalation_triggers` as a safety net for cases routed directly (skipping `decision_node`).

### GOV-07: Circuit Breaker (ORCH-06)

**Guarantee:** When the vector database becomes unavailable, the pipeline degrades gracefully rather than retrying indefinitely.

**Implementation:**
- Module-level circuit breaker in `orchestrator_nodes.py` tracks consecutive failures per grounding checkpoint
- Threshold: 3 consecutive failures before tripping
- When tripped: subsequent calls return a zero-score grounding result with `circuit_broken=True` without calling the vector DB
- The call that trips the breaker carries `circuit_broken=True`; subsequent calls fast-path via `_circuit_breaker_is_open()`
- Successful calls reset the per-checkpoint failure counter
- Circuit-broken grounding entries route to escalate (not retry) -- once the breaker has tripped, retries cannot help

---

## Audit Trail

### Entry Structure

Each `AuditEntry` (Pydantic model in `src/models/governance.py`) contains:

| Field | Type | Description |
|-------|------|-------------|
| `entry_id` | str | Unique UUID per entry |
| `timestamp` | datetime | ISO 8601 timestamp |
| `stage` | str | Pipeline stage (INTAKE, ANALYSIS, REVIEW, etc.) |
| `action` | str | Specific action (input_received, analysis_complete, grounding_check, etc.) |
| `agent` | str (optional) | Agent name if applicable |
| `details` | dict | Additional context (not included in hash) |
| `duration_ms` | float (optional) | Execution duration |
| `token_count` | int (optional) | LLM token usage |
| `hash` | str | SHA-256 hash linking to previous entry |

### Hash Chain Verification

```
GENESIS --> hash(GENESIS + entry_0) --> hash(hash_0 + entry_1) --> ... --> hash(hash_n-1 + entry_n)
```

Only immutable fields participate in the hash computation: `entry_id`, `timestamp`, `stage`, `action`. The `details` dict is excluded because its serialisation order is non-deterministic. `sort_keys=True` ensures consistent JSON serialisation of the hash input.

### Lifecycle Actions

A typical request produces 30-50 audit entries covering:

- `input_received` -- application validated
- `pii_scan_complete` -- PII detection results
- `analysis_complete` -- analyst agent finished
- `grounding_check` -- grounding verification at each checkpoint
- `review_complete` -- reviewer agent finished
- `compliance_complete` -- compliance agent finished
- `matrix_applied` -- decision matrix inputs and output
- `decision_rendered` -- final decision with confidence
- `escalation_triggered` -- escalation triggers that fired (if any)
- `human_review_required` -- escalation terminal entry (if escalated)

---

## Configuration

All governance behaviour is driven by four YAML configuration files loaded via `ConfigLoader` (`src/config/settings.py`) with `lru_cache` for single-read semantics:

### guardrails.yaml

Controls grounding thresholds, PII patterns, bias terms, and escalation triggers.

```yaml
grounding:
  threshold: 0.7           # Per-claim embedding similarity cutoff
  max_retries: 2            # Re-prompt attempts when >20% claims fail
  ungrounded_claim_limit: 0.2  # Maximum ratio of ungrounded claims

pii:
  enabled: true
  patterns: [...]           # 6 regex patterns for UK PII types
  redaction_char: "*"

bias:
  enabled: true
  protected_characteristics: [...]  # 9 Equality Act 2010 terms
  proxy_variables: [...]            # 5 known proxy variables

escalation:
  triggers: [...]           # 6 named triggers with conditions
```

### agents.yaml

Configures agent roles, LLM settings, and tool permissions.

```yaml
analyst:
  role: "Credit Risk Analyst"
  llm_model: "gpt-4o"
  temperature: 0.1
  max_iterations: 5
  permissions: [rag_financial_lookup, rag_sector_analysis, credit_scorer, risk_calculator, historical_comparator]

reviewer:
  role: "Senior Credit Review Officer"
  llm_model: "gpt-4o"
  temperature: 0.1
  max_iterations: 5
  permissions: [rag_financial_lookup, rag_sector_analysis, rag_policy_lookup, stress_tester, historical_comparator]

compliance:
  role: "Regulatory Compliance Officer"
  llm_model: "gpt-4o"
  temperature: 0.0
  max_iterations: 3
  permissions: [rag_policy_lookup, concentration_checker]
```

### scoring.yaml

Defines credit scoring weights, stress test scenarios, and concentration limits.

- Credit scoring: 8 weighted factors summing to 1.0 (enforced by `model_validator`)
- Stress testing: 5 scenarios with revenue shocks, cost increases, and default rate multipliers
- Concentration: single-name limit 5%, sector limit 25%

### config.yaml

Application-wide settings: Weaviate connection, embedding model, observability flags, processing limits.

```yaml
providers:
  weaviate_url: "http://localhost:8080"
  embedding_model: "text-embedding-3-small"
  embedding_dimensions: 1536

processing:
  max_concurrent_requests: 5
  request_timeout_seconds: 300
  llm_retry_attempts: 3
  llm_retry_backoff_seconds: 2
```
