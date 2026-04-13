#!/usr/bin/env python
"""CLI entry point for ingesting synthetic data into Weaviate.

Loads pre-generated JSON files from disk (or generates fresh data) and
runs the full ingestion pipeline: content composition, chunking,
embedding, and batch insertion into Weaviate collections.

Usage from project root:

    python scripts/ingest_documents.py
    python scripts/ingest_documents.py --data-dir data/synthetic
    python scripts/ingest_documents.py --seed 42 --num-applications 100 --num-decisions 400
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure project root is on sys.path so ``from src.`` imports work
# regardless of the working directory when invoked.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest synthetic data into Weaviate for the Enterprise Agentic Orchestrator RAG pipeline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation if generating fresh (default: 42)",
    )
    parser.add_argument(
        "--num-applications",
        type=int,
        default=50,
        help="Number of loan applications to generate if no JSON files found (default: 50)",
    )
    parser.add_argument(
        "--num-decisions",
        type=int,
        default=200,
        help="Number of historical decisions to generate if no JSON files found (default: 200)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/synthetic",
        help='Directory containing pre-generated JSON files (default: "data/synthetic")',
    )
    return parser.parse_args()


def _load_json(path: str) -> list:
    """Load a JSON array from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_or_generate(args: argparse.Namespace) -> tuple[list, list, list, list]:
    """Load data from JSON files or generate fresh data.

    Returns (applications, sector_reports, regulatory_docs, historical_decisions)
    as lists of plain dicts.
    """
    data_dir = os.path.join(_PROJECT_ROOT, args.data_dir)

    apps_path = os.path.join(data_dir, "loan_applications.json")
    sectors_path = os.path.join(data_dir, "sector_reports.json")
    policies_path = os.path.join(data_dir, "regulatory_policies.json")
    decisions_path = os.path.join(data_dir, "historical_decisions.json")

    # Check if all four JSON files exist
    all_exist = all(
        os.path.isfile(p)
        for p in [apps_path, sectors_path, policies_path, decisions_path]
    )

    if all_exist:
        print(f"Loading pre-generated data from {data_dir} ...")
        applications = _load_json(apps_path)
        sector_reports = _load_json(sectors_path)
        regulatory_docs = _load_json(policies_path)
        historical_decisions = _load_json(decisions_path)
        print(
            f"  Loaded: {len(applications)} applications, "
            f"{len(sector_reports)} sector reports, "
            f"{len(regulatory_docs)} regulatory policies, "
            f"{len(historical_decisions)} historical decisions"
        )
    else:
        print(f"JSON files not found in {data_dir}. Generating fresh data (seed={args.seed}) ...")

        from src.generators import (
            generate_historical_decisions,
            generate_loan_applications,
            generate_regulatory_docs,
            generate_sector_reports,
        )

        app_models = generate_loan_applications(n=args.num_applications, seed=args.seed)
        applications = [app.model_dump(mode="json") for app in app_models]
        sector_reports = generate_sector_reports()
        regulatory_docs = generate_regulatory_docs()
        historical_decisions = generate_historical_decisions(
            app_models, n=args.num_decisions, seed=args.seed,
        )
        print(
            f"  Generated: {len(applications)} applications, "
            f"{len(sector_reports)} sector reports, "
            f"{len(regulatory_docs)} regulatory policies, "
            f"{len(historical_decisions)} historical decisions"
        )

    return applications, sector_reports, regulatory_docs, historical_decisions


def main() -> None:
    args = _parse_args()

    # Load or generate data
    applications, sector_reports, regulatory_docs, historical_decisions = _load_or_generate(args)

    # Connect to Weaviate and ingest
    try:
        import weaviate

        host = os.environ.get("WEAVIATE_HOST", "localhost")
        port = int(os.environ.get("WEAVIATE_HTTP_PORT", "8080"))
        grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))
        print(f"\nConnecting to Weaviate at {host}:{port} ...")
        client = weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)
    except Exception as exc:
        print(
            f"\nWeaviate not reachable: {exc}\n"
            "Run: docker-compose up -d && python scripts/setup_weaviate.py"
        )
        sys.exit(1)

    try:
        from src.rag.ingestion import ingest_all

        print()
        counts = ingest_all(
            client,
            applications=applications,
            sector_reports=sector_reports,
            regulatory_docs=regulatory_docs,
            historical_decisions=historical_decisions,
        )

        total = sum(counts.values())
        print(f"\nIngested {total} total objects across {len(counts)} collections:")
        for name, count in counts.items():
            print(f"  {name}: {count}")
        print("\nDone.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
