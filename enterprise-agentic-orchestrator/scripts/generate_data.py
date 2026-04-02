#!/usr/bin/env python
"""CLI entry point for generating the complete synthetic dataset.

Orchestrates all four generators and writes JSON files to the output
directory.  Runnable from the project root:

    python scripts/generate_data.py --seed 42 --output-dir data/synthetic

The generated files are consumed by Phase 3 RAG ingestion.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure project root is on sys.path so ``from src.…`` imports work
# regardless of the working directory when invoked.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.generators import (  # noqa: E402
    generate_historical_decisions,
    generate_loan_applications,
    generate_regulatory_docs,
    generate_sector_reports,
)
from src.models.loan import LoanApplication  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the complete synthetic dataset for the Enterprise Agentic Orchestrator.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/synthetic",
        help='Output directory for JSON files (default: "data/synthetic")',
    )
    parser.add_argument(
        "--num-applications",
        type=int,
        default=50,
        help="Number of loan applications to generate (default: 50)",
    )
    parser.add_argument(
        "--num-decisions",
        type=int,
        default=200,
        help="Number of historical decisions to generate (default: 200)",
    )
    return parser.parse_args()


def _write_json(data: list, path: str) -> int:
    """Write *data* as formatted JSON and return file size in bytes."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return os.path.getsize(path)


def main() -> None:
    args = _parse_args()

    # Resolve output directory relative to project root
    output_dir = os.path.join(_PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating synthetic dataset (seed={args.seed}) ...")
    print(f"Output directory: {output_dir}\n")

    # 1. Loan applications (Pydantic models)
    applications = generate_loan_applications(n=args.num_applications, seed=args.seed)
    apps_data = [app.model_dump(mode="json") for app in applications]
    apps_path = os.path.join(output_dir, "loan_applications.json")
    apps_size = _write_json(apps_data, apps_path)
    print(f"  loan_applications.json: {len(apps_data)} items, {apps_size:,} bytes")

    # 2. Sector reports (plain dicts)
    sector_reports = generate_sector_reports()
    reports_path = os.path.join(output_dir, "sector_reports.json")
    reports_size = _write_json(sector_reports, reports_path)
    print(f"  sector_reports.json:    {len(sector_reports)} items, {reports_size:,} bytes")

    # 3. Regulatory policies (plain dicts)
    regulatory_docs = generate_regulatory_docs()
    policies_path = os.path.join(output_dir, "regulatory_policies.json")
    policies_size = _write_json(regulatory_docs, policies_path)
    print(f"  regulatory_policies.json: {len(regulatory_docs)} items, {policies_size:,} bytes")

    # 4. Historical decisions (plain dicts, cross-references applications)
    decisions = generate_historical_decisions(
        applications, n=args.num_decisions, seed=args.seed,
    )
    decisions_path = os.path.join(output_dir, "historical_decisions.json")
    decisions_size = _write_json(decisions, decisions_path)
    print(f"  historical_decisions.json: {len(decisions)} items, {decisions_size:,} bytes")

    total_size = apps_size + reports_size + policies_size + decisions_size
    print(f"\nTotal: {len(apps_data) + len(sector_reports) + len(regulatory_docs) + len(decisions)} items, {total_size:,} bytes")

    # 5. Validation: round-trip Pydantic check on loan applications
    print("\nValidating loan_applications.json through Pydantic models ...")
    with open(apps_path, encoding="utf-8") as f:
        loaded = json.load(f)
    for item in loaded:
        LoanApplication(**item)
    print(f"  All {len(loaded)} loan applications validated successfully.")

    print("\nDone.")


if __name__ == "__main__":
    main()
