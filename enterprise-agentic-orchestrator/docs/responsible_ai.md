# Responsible AI Principles

**Enterprise Agentic Orchestrator -- Ethical AI in Regulated Lending**

## Principles Overview

This system is designed for credit risk assessment in UK financial services -- a domain where AI decisions directly affect people's access to capital. The responsible AI principles embedded in this architecture reflect both regulatory obligations (FCA Consumer Duty, Equality Act 2010, EU AI Act) and a commitment to building AI systems that are fair, transparent, and accountable.

These principles are not aspirational statements bolted onto the documentation after the fact. They are architectural constraints enforced through code, configuration, and pipeline design at every stage of the assessment process.

### Core Principles

1. **Fairness** -- Lending decisions must not discriminate on protected characteristics
2. **Transparency** -- Every decision must be explainable to applicants, auditors, and regulators
3. **Accountability** -- Every action is recorded in a tamper-evident audit trail
4. **Grounding** -- AI claims must be traceable to source data, not hallucinated
5. **Human oversight** -- High-risk or uncertain cases are escalated to human underwriters
6. **Privacy** -- Personal data is detected, recorded, and protected throughout the pipeline

---

## Protected Characteristics

### Exclusion by Design

The system is architected to ensure that protected characteristics under the Equality Act 2010 are **never used as factors** in lending decisions. This is enforced at multiple levels:

**1. Input design:** The `LoanApplication` Pydantic model does not include fields for age, gender, race, ethnicity, religion, disability, sexual orientation, marital status, or pregnancy. These characteristics cannot enter the pipeline because the data schema does not accept them.

**2. Credit scoring:** The credit scorer (`src/tools/credit_scorer.py`) uses 8 purely financial factors:
- Profit margin, debt-to-asset ratio, revenue trend, cash coverage, years trading, sector outlook, CCJ history, security coverage
- None of these factors correlate with protected characteristics by construction (all are business financial metrics)

**3. Agent output scanning:** The `BiasChecker` (`src/guardrails/bias.py`) scans all agent-generated text for references to:
- **9 protected characteristics:** age, gender, race, ethnicity, religion, disability, sexual orientation, marital status, pregnancy
- **5 proxy variables:** postcode, first name, school attended, accent, nationality

Proxy variables are included because they can serve as indirect indicators of protected characteristics (e.g., postcode correlating with ethnicity, first name correlating with gender or ethnicity).

**4. Matching precision:** Single-word terms use `\b` word-boundary regex to prevent false positives -- "postage" does not trigger on "age", "manager" does not trigger on "age". Multi-word terms use substring matching on lowercased text.

### Limitations

- The bias checker is term-based, not semantic. An agent could express discriminatory reasoning using synonyms or circumlocutions that the checker would not detect. This is acknowledged as a known limitation (see Limitations section below).
- Proxy variable detection flags presence, but does not automatically block output. The system records the finding for human review.
- The credit scoring factors (particularly sector outlook and years trading) could theoretically exhibit disparate impact across demographic groups. This would require ongoing monitoring against real portfolio outcomes, which is out of scope for this portfolio demonstration.

---

## Grounding and Hallucination Prevention

### The Problem

LLM-based agents can generate plausible but fabricated financial claims -- invented revenue figures, non-existent regulations, or fictional historical precedents. In regulated lending, ungrounded AI output is not merely incorrect; it is a compliance violation that could lead to unfair lending decisions.

### The Approach

The system implements a **grounding-first architecture** with verification at three mandatory checkpoints:

1. **Post-analyst grounding** -- Verify the analyst's financial claims against retrieved documents
2. **Post-reviewer grounding** -- Verify the reviewer's validation claims against retrieved documents
3. **Post-compliance grounding** -- Verify the compliance officer's regulatory citations against retrieved documents

### Verification Method

The `GroundingChecker` (`src/guardrails/grounding.py`) uses **embedding similarity** to verify grounding:

1. **Claim extraction:** Agent output is split into individual claims via sentence splitting (period-space or period-newline boundaries, minimum 10 characters per claim)
2. **Batch embedding:** All claims and all source document texts are embedded in exactly 2 API calls (one for claims, one for sources) using OpenAI `text-embedding-3-small`
3. **Cosine similarity:** Pairwise similarity is computed via dot product on unit-normalised embeddings
4. **Threshold evaluation:** Claims with similarity below 0.7 to their best-matching source are flagged as ungrounded
5. **Aggregate assessment:** If more than 20% of claims are ungrounded, the output fails grounding verification

### Retry and Escalation

When grounding fails:
- The agent is re-prompted (up to 2 retries per checkpoint)
- Retry routing is handled by LangGraph conditional edges
- If retries are exhausted, the case is escalated to human review
- Circuit breaker prevents infinite retry loops when the vector database is unavailable

### Known Limitations

- Embedding similarity operates at the sentence level, not the numerical level. A claim "revenue was 1.5M" matched against a source saying "revenue was 1.4M" may score high similarity despite the factual discrepancy. Numerical precision verification is a known gap.
- The 0.7 threshold is a configuration parameter, not empirically optimised against a labelled ground truth set. Tuning against real grounding labels would improve accuracy.
- Grounding verification depends on the quality and completeness of the retrieved source documents. If relevant documents are not retrieved, valid claims may be flagged as ungrounded.

---

## Human-in-the-Loop Escalation

### Design Philosophy

The system does not make final lending decisions in isolation. It identifies cases that require human judgment and routes them to qualified underwriters. This is not a fallback for system failures -- it is a deliberate design choice reflecting the principle that high-stakes financial decisions should have human oversight.

### Escalation Triggers

Six conditions automatically escalate a case to human review (configured in `guardrails.yaml`):

| Trigger | Rationale |
|---------|-----------|
| **High-value loan** (> 500,000) | Large exposures warrant additional scrutiny regardless of automated assessment |
| **Deteriorating sector** | Sector-level risk requires judgment about timing and exposure management |
| **Compliance failure** | Any regulatory check failure requires human evaluation of severity and remediation |
| **Low reviewer confidence** | When the independent reviewer has low confidence, the assessment is uncertain |
| **Grounding failure** | When agent claims cannot be verified against source data, the assessment is unreliable |
| **Low average grounding** (< 0.75) | Even if no single checkpoint failed, a low overall grounding score indicates reduced reliability |

### Escalation Behaviour

When any trigger fires:
- The decision matrix outcome is overridden to `ESCALATED`
- The `requires_escalation` flag is set in orchestrator state
- All trigger reasons are recorded in the reasoning trace and audit trail
- The case is routed to the `escalation_node`, which assembles a comprehensive human-review summary
- The response API returns HTTP 202 (Accepted) rather than 200 (OK) to signal the case requires human action

### Fail-Closed Design

The decision matrix is deliberately **fail-closed**:
- Unknown analyst recommendations route to `REFERRED_TO_UNDERWRITER` (human review), never silently approved or rejected
- The escalation node re-runs all trigger checks as a belt-and-braces safety net
- Any unhandled exception in decision or escalation nodes produces an error delta that routes to human review

---

## Explainability and Transparency

### Per-Decision Explainability

Every assessment produces a structured explainability report accessible via `GET /api/v1/decisions/{id}/explain`:

- **Analysis summary:** The analyst's financial assessment, credit score, risk metrics, and recommendation
- **Review summary:** The reviewer's independent validation, stress test results, and agree/disagree assessment
- **Compliance summary:** All 5 regulatory check results with pass/fail and specific regulation citations
- **Grounding scores:** Per-checkpoint grounding scores showing how well each agent's output was verified
- **Reasoning trace:** Human-readable narrative of the decision logic: "Analyst recommendation: APPROVE. Reviewer agrees: True. Compliance passed: True. Decision: APPROVED. Confidence: 0.82. No escalation triggers."

### Deterministic Decision Logic

The final lending decision is produced by a deterministic 9-row matrix (not an LLM). This means:
- The same inputs always produce the same output
- The logic can be explained to a regulator without referencing AI behaviour
- The decision can be audited by inspecting three boolean inputs and one lookup table
- No probabilistic or learned components affect the final outcome

### Source Traceability

Every claim in every agent output can be traced to a specific source document through the grounding verification system. Source citations include:
- Document ID
- Document type (collection name)
- Chunk text (first 200 characters)
- Relevance score

---

## Data Privacy

### Synthetic-Only Data

This system uses exclusively synthetic data. No real borrower, company, or financial data is used at any point in the pipeline. All 50 loan applications, 200 historical decisions, and supporting documents are generated using seeded random generators for reproducibility.

### PII Detection and Redaction

Even with synthetic data, the PII detection system is fully implemented and operational:

- **Input scanning:** The intake node scans every application for 6 types of UK PII (NI numbers, sort codes, bank accounts, dates of birth, phone numbers, email addresses)
- **Output redaction:** `scan_and_redact()` replaces PII matches with configurable redaction characters before agent output is persisted
- **Audit recording:** PII detection events are recorded in the audit trail with types detected (not the PII values themselves)
- **Pattern ordering:** NI Number -> Sort Code -> Bank Account processing order prevents cross-pattern interference during redaction

### Data Retention

Assessment results are persisted as JSON files. The audit trail's `export_json()` method produces a format suitable for 7-year regulatory retention requirements (FCA/PRA).

---

## Limitations and Ongoing Considerations

This system is a portfolio demonstration of responsible AI architecture for regulated lending. The following limitations should be understood:

### Known Technical Limitations

1. **Numerical grounding gap:** Embedding similarity cannot verify numerical precision. A claim of "revenue was 1.5M" may score as grounded against a source saying "revenue was 1.4M" because the sentence-level semantics are similar. Dedicated numerical fact-checking would require a separate verification pass.

2. **Term-based bias detection:** The bias checker uses keyword matching, not semantic analysis. Discriminatory reasoning expressed through synonyms, euphemisms, or statistical proxies not in the configured list would not be detected.

3. **Static proxy variable list:** The proxy variable list (postcode, first name, school attended, accent, nationality) is manually curated. Research has shown that many seemingly neutral features can serve as proxies for protected characteristics. The current list may be incomplete.

4. **No disparate impact analysis:** The system does not perform statistical analysis of decision outcomes across demographic groups. In production, this would be essential for detecting indirect discrimination that cannot be caught by term-based scanning.

5. **Single-model dependency:** All agents use the same LLM provider (Azure OpenAI GPT-4o). The system's fairness and accuracy are bounded by the base model's biases and capabilities. A production system would benefit from model diversity and regular bias auditing of the underlying LLM.

6. **Grounding threshold not empirically calibrated:** The 0.7 similarity threshold and 20% ungrounded claim limit are configuration parameters, not values derived from empirical evaluation against labelled datasets. Production deployment would require threshold calibration against known-good and known-bad grounding examples.

### Production Considerations

For production deployment in a regulated environment, the following additional measures would be recommended:

- **Ongoing bias monitoring:** Regular analysis of decision distributions across demographic segments to detect disparate impact
- **Model validation cadre:** Independent validation of LLM behaviour against a curated test set of lending scenarios
- **Regulatory change management:** Process for updating regulatory policy documents and compliance checks when regulations change
- **Human oversight training:** Training materials for underwriters receiving escalated cases, covering the system's capabilities and limitations
- **Feedback loop:** Mechanism for human underwriters to provide feedback on escalated cases, improving future automated assessment quality
- **Red team testing:** Adversarial testing of agent outputs to identify ways the grounding and bias detection systems can be circumvented

### Ethical Commitment

This system demonstrates that governance can be built into AI architecture from the start -- not added as an afterthought. The multi-agent design, grounding verification, bias detection, and human-in-the-loop escalation are not features; they are the foundation on which the entire system is built. Every design decision prioritises safety, fairness, and transparency over convenience or performance.
