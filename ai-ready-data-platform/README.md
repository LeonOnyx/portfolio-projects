# AI-Ready Data Platform

**Governed RAG for Regulated Industries**

> An end-to-end data platform demonstrating the data foundations that must exist before enterprise AI can be trusted. Medallion architecture, automated governance, data quality controls, and a governed RAG interface with row-level security — built for industries where getting AI wrong has regulatory consequences.

## The Thesis

Most enterprise AI programmes fail at the data layer, not the model layer. Organisations invest in LLMs and agentic frameworks while sitting on fragmented, ungoverned, undocumented data estates. The result: AI that hallucinates, leaks sensitive data, or produces outputs nobody trusts.

This project demonstrates how to build a data platform that makes AI viable — not by adding AI on top of broken data, but by architecting the foundation that AI requires to be safe, accurate, and auditable.

## What This Project Demonstrates

A complete data platform pipeline that ingests raw data, transforms it through a medallion architecture, applies governance and quality controls at every layer, and exposes a governed RAG interface where an LLM can answer questions grounded only in trusted, classified, access-controlled data.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Data Sources                          │
│  CSV/JSON │ APIs │ Databases │ Streaming │ Documents     │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 Bronze Layer (Raw)                        │
│  Raw ingestion │ Schema detection │ Source tagging        │
│  Data lineage capture │ Immutable audit log              │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                Silver Layer (Cleansed)                    │
│  Data quality checks │ Deduplication │ Type enforcement   │
│  PII detection & classification │ Standardisation        │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │          Governance Controls                        │  │
│  │  Classification │ Sensitivity labels │ Lineage      │  │
│  │  Quality scores │ Freshness tracking                │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 Gold Layer (Business)                     │
│  Business entities │ Aggregations │ Feature store         │
│  Semantic models │ Access control (RLS)                   │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │         Row-Level Security (RLS)                    │  │
│  │  Role-based access │ Column masking │ Audit trail   │  │
│  └────────────────────────────────────────────────────┘  │
└──────────┬───────────────────────────────┬──────────────┘
           │                               │
┌──────────▼──────────┐     ┌──────────────▼──────────────┐
│   Analytics Layer    │     │      Governed RAG Layer      │
│                      │     │                              │
│  Power BI Dashboard  │     │  Document chunking           │
│  Data quality metrics│     │  Embedding generation        │
│  Lineage explorer    │     │  Vector store (Weaviate)     │
│  Governance scorecard│     │  Retrieval with RLS          │
│                      │     │  LLM grounded response       │
│                      │     │  Source attribution           │
│                      │     │  Grounding verification       │
└──────────────────────┘     └─────────────────────────────┘
```

### Key Design Principles

1. **Governance before AI** — Data classification, lineage, and quality controls are applied at ingestion, not retrofitted after deployment
2. **Row-level security on RAG** — The LLM can only access data the requesting user is authorised to see
3. **Source attribution** — Every RAG response includes citations to the specific gold-layer records that informed it
4. **Grounding verification** — Responses are checked against source data; ungrounded content is flagged and rejected
5. **Observable data quality** — Real-time dashboard showing quality scores, freshness, lineage, and governance compliance across the estate

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Compute** | PySpark (Databricks / Fabric) | Distributed data processing across all layers |
| **Storage** | Delta Lake | ACID transactions, time travel, schema evolution |
| **Orchestration** | Azure Data Factory / Airflow | Pipeline scheduling and dependency management |
| **Governance** | Custom (mirrors Purview patterns) | Classification, lineage, quality scoring, access control |
| **Data Quality** | Great Expectations + custom rules | Automated quality checks at bronze→silver→gold transitions |
| **Vector Database** | Weaviate | Embedding storage for RAG retrieval |
| **Embeddings** | Azure OpenAI / Sentence Transformers | Document and record embedding generation |
| **LLM** | Azure OpenAI / Claude | Grounded response generation |
| **RAG Framework** | LlamaIndex | Document processing, chunking, retrieval pipeline |
| **Visualisation** | Power BI (embedded) | Governance dashboard, data quality scorecard |
| **Language** | Python 3.11+ | All components |
| **Cloud** | Azure (primary), Databricks (alternative) | Infrastructure |

## Project Structure

```
ai-ready-data-platform/
├── src/
│   ├── ingestion/
│   │   ├── batch_ingestor.py       # Batch data ingestion (CSV, JSON, Parquet, API)
│   │   ├── stream_ingestor.py      # Streaming ingestion (Kafka, Event Hub)
│   │   ├── schema_detector.py      # Automatic schema inference and validation
│   │   └── source_registry.py      # Data source registration and metadata
│   ├── transforms/
│   │   ├── bronze_to_silver.py     # Cleansing, deduplication, type enforcement
│   │   ├── silver_to_gold.py       # Business entity creation, aggregation
│   │   ├── feature_store.py        # Feature engineering for ML consumption
│   │   └── spark_utils.py          # PySpark helper functions
│   ├── governance/
│   │   ├── classifier.py           # Data sensitivity classification engine
│   │   ├── lineage_tracker.py      # End-to-end data lineage capture
│   │   ├── quality_engine.py       # Data quality rules and scoring
│   │   ├── pii_detector.py         # PII detection and masking
│   │   ├── access_control.py       # Row-level security and column masking
│   │   └── audit_logger.py         # Immutable governance audit trail
│   ├── rag/
│   │   ├── document_processor.py   # Document chunking and preparation
│   │   ├── embedding_engine.py     # Embedding generation and management
│   │   ├── vector_store.py         # Weaviate integration for storage and retrieval
│   │   ├── retriever.py            # Governed retrieval with RLS enforcement
│   │   ├── grounding_checker.py    # Output grounding verification
│   │   └── response_generator.py   # LLM response with source attribution
│   └── api/
│       ├── query_endpoint.py       # REST API for governed RAG queries
│       ├── governance_api.py       # Governance metrics and reporting API
│       └── auth.py                 # Authentication and authorisation
├── config/
│   ├── governance_rules.yaml       # Data classification and quality rules
│   ├── access_policies.yaml        # RLS policies and role definitions
│   ├── pipeline_config.yaml        # Pipeline scheduling and parameters
│   └── config.yaml                 # Application configuration
├── tests/
│   ├── test_transforms.py          # Bronze→Silver→Gold transformation tests
│   ├── test_governance.py          # Classification, lineage, quality tests
│   ├── test_rag.py                 # RAG accuracy and grounding tests
│   ├── test_rls.py                 # Row-level security enforcement tests
│   └── test_quality.py             # Data quality rule validation
├── dashboards/
│   ├── governance_scorecard.pbix   # Power BI governance dashboard
│   └── data_quality_report.pbix    # Data quality metrics dashboard
├── docs/
│   ├── architecture.md             # Platform architecture documentation
│   ├── governance_framework.md     # Governance principles and controls
│   ├── medallion_patterns.md       # Medallion architecture design decisions
│   ├── governed_rag.md             # How RLS is enforced through the RAG pipeline
│   └── deployment.md              # Deployment guide
├── notebooks/
│   ├── 01_bronze_ingestion.ipynb   # Interactive bronze layer walkthrough
│   ├── 02_silver_cleansing.ipynb   # Silver layer transformation demo
│   ├── 03_gold_modelling.ipynb     # Gold layer business entity creation
│   ├── 04_governance_demo.ipynb    # Governance controls demonstration
│   └── 05_governed_rag_demo.ipynb  # End-to-end governed RAG walkthrough
├── sample_data/
│   ├── raw/                        # Sample raw data files
│   └── reference/                  # Reference data and lookup tables
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── .env.example
└── README.md
```

## Use Cases

### 1. Financial Services — Governed Customer Analytics
A bank ingests customer transaction data, applies PII classification and sensitivity labelling, transforms through medallion layers with quality controls, and exposes a RAG interface where relationship managers can ask questions about customer behaviour — but only see data for customers in their assigned portfolio (RLS enforced).

### 2. Healthcare — Clinical Data Platform
A healthcare provider consolidates clinical, operational, and patient feedback data. Governance controls ensure HIPAA/GDPR compliance, PII is detected and masked, and a governed RAG interface allows clinical staff to query patient pathways while respecting consent and access boundaries.

### 3. Pharmaceutical — Regulatory Compliance
A pharma company builds a data estate for regulatory submissions. Every data point is lineage-tracked from source through transformation to final report. A governed RAG interface allows regulatory affairs teams to query submission data with full audit trail and source attribution.

## Governance Framework

This platform implements governance as architecture, not afterthought:

### Data Classification
- Automatic sensitivity classification at ingestion (Public, Internal, Confidential, Restricted)
- PII detection using pattern matching and NLP-based entity recognition
- Classification labels propagated through all transformations

### Data Quality
- Automated quality rules at every medallion layer transition
- Quality scores computed per dataset, per record, per column
- Quality gates that prevent low-quality data from reaching the gold layer
- Historical quality trending and anomaly detection

### Data Lineage
- End-to-end lineage from source to gold layer to RAG output
- Column-level lineage tracking
- Impact analysis for upstream changes

### Access Control
- Row-level security (RLS) enforced at the gold layer and through the RAG pipeline
- Column masking for sensitive fields
- Role-based access policies defined in YAML configuration
- Every access attempt logged in the audit trail

### RAG Governance
- The LLM only receives retrieved context that passes the user's RLS policy
- Every response includes source citations to specific gold-layer records
- Grounding checker validates that the response is faithful to retrieved context
- Ungrounded or hallucinated content is detected and rejected

## Roadmap

- [x] Project structure and architecture documentation
- [ ] Bronze layer: batch ingestion with schema detection
- [ ] Bronze layer: source tagging and lineage capture
- [ ] Silver layer: PySpark cleansing and deduplication
- [ ] Silver layer: PII detection and classification
- [ ] Silver layer: data quality engine (Great Expectations)
- [ ] Gold layer: business entity models
- [ ] Gold layer: row-level security implementation
- [ ] Governance dashboard (Power BI)
- [ ] RAG pipeline: document processing and embedding
- [ ] RAG pipeline: governed retrieval with RLS
- [ ] RAG pipeline: grounding verification
- [ ] API layer: query endpoint with auth
- [ ] Interactive notebooks (5x walkthrough)
- [ ] Docker deployment
- [ ] Comprehensive test suite

## Author

**Leon Gordon** — Principal Data & AI Architect | 5x Microsoft Data Platform MVP | Oxford Saïd AI Programme

- LinkedIn: [linkedin.com/in/leon-gordon](https://linkedin.com/in/leon-gordon)
- Microsoft Build & Ignite featured customer success story
- Specialist in enterprise data governance and AI-ready data estates for regulated industries

## License

MIT
