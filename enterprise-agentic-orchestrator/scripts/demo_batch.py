#!/usr/bin/env python
"""Demo: Batch loan application credit risk assessment.

Processes all 50 loan applications from the synthetic dataset through the
Enterprise Agentic Orchestrator API and prints summary statistics including
decision distribution, average grounding scores, escalation rate, and
total processing time.

Usage:
    python scripts/demo_batch.py                        # All 50 applications
    python scripts/demo_batch.py --max 10               # First 10 only
    python scripts/demo_batch.py --url http://host:8000 # Custom API URL

Requires the API to be running (docker-compose up).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter

# Ensure project root is on sys.path so ``from src.…`` imports work
# regardless of the working directory when invoked.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import httpx  # noqa: E402


# =====================================================================
# Health check
# =====================================================================


def wait_for_health(base_url: str, max_wait: int = 60) -> bool:
    """Poll GET /api/v1/health until the API responds.

    Retries every 2 seconds up to *max_wait* seconds.  Returns ``True``
    when the health endpoint responds (any status code), ``False`` if
    connection is refused or the timeout is exceeded.
    """
    url = f"{base_url}/api/v1/health"
    deadline = time.monotonic() + max_wait
    print(f"Waiting for API at {url} ", end="", flush=True)

    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=5)
            print(f" ready (status={resp.status_code})")
            return True
        except httpx.ConnectError:
            print(".", end="", flush=True)
            time.sleep(2)
        except httpx.TimeoutException:
            print(".", end="", flush=True)
            time.sleep(2)
        except Exception:
            print(".", end="", flush=True)
            time.sleep(2)

    print(" timed out!")
    return False


# =====================================================================
# Data loading
# =====================================================================


def load_all_applications(data_dir: str, max_count: int = 0) -> list[dict]:
    """Load loan applications from the synthetic dataset.

    Parameters
    ----------
    data_dir:
        Path to the directory containing ``loan_applications.json``.
    max_count:
        Maximum number of applications to return.  ``0`` means all.

    Returns
    -------
    list[dict]
        Application payloads with ``application_id`` and
        ``submitted_at`` stripped (the API generates those server-side).

    Raises
    ------
    SystemExit
        If the data file does not exist or is empty.
    """
    filepath = os.path.join(data_dir, "loan_applications.json")
    if not os.path.isfile(filepath):
        print(f"ERROR: Data file not found: {filepath}")
        print("Run 'python scripts/generate_data.py' first to create synthetic data.")
        sys.exit(1)

    with open(filepath) as fh:
        applications = json.load(fh)

    if not applications:
        print("ERROR: No applications found in data file.")
        sys.exit(1)

    # Strip server-generated fields from each application
    for app in applications:
        app.pop("application_id", None)
        app.pop("submitted_at", None)

    if max_count > 0:
        applications = applications[:max_count]

    return applications


# =====================================================================
# Batch processing
# =====================================================================


def process_batch(
    base_url: str, applications: list[dict], timeout: int
) -> list[dict]:
    """Submit each application sequentially and collect results.

    Each result dict contains:
    - ``index``: zero-based application index
    - ``company``: company name for display
    - ``status_code``: HTTP status code (0 for connection errors)
    - ``result``: the parsed response body
    - ``elapsed``: seconds taken for this request

    Processing continues even if individual applications fail.
    """
    url = f"{base_url}/api/v1/assess"
    total = len(applications)
    results: list[dict] = []

    for i, app in enumerate(applications):
        company = app.get("applicant", {}).get("company_name", f"Application {i}")
        amount = app.get("loan", {}).get("amount_requested", "N/A")
        progress = f"[{i + 1}/{total}]"

        print(f"  {progress} {company} (GBP {amount})...", end=" ", flush=True)

        start = time.monotonic()
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(url, json=app)
                try:
                    body = resp.json()
                except Exception:
                    body = {"raw": resp.text}
                status_code = resp.status_code
        except httpx.ConnectError as exc:
            status_code = 0
            body = {"error": f"Connection refused: {exc}"}
        except httpx.TimeoutException as exc:
            status_code = 0
            body = {"error": f"Timed out: {exc}"}
        except Exception as exc:
            status_code = 0
            body = {"error": f"Unexpected: {exc}"}

        elapsed = time.monotonic() - start

        # Print inline status
        if status_code == 200:
            decision = body.get("decision", "?")
            confidence = body.get("confidence_score", 0.0)
            print(f"{decision} ({confidence:.0%}) [{elapsed:.1f}s]")
        elif status_code == 202:
            print(f"ESCALATED [{elapsed:.1f}s]")
        elif status_code == 0:
            print(f"ERROR (connection) [{elapsed:.1f}s]")
        else:
            print(f"ERROR ({status_code}) [{elapsed:.1f}s]")

        results.append({
            "index": i,
            "company": company,
            "status_code": status_code,
            "result": body,
            "elapsed": elapsed,
        })

    return results


# =====================================================================
# Summary statistics
# =====================================================================

_LINE = "=" * 70


def print_batch_summary(results: list[dict], total_elapsed: float) -> None:
    """Print a formatted summary of batch processing results."""
    print()
    print(_LINE)
    print("BATCH PROCESSING SUMMARY")
    print(_LINE)

    total = len(results)
    if total == 0:
        print("  No results to summarise.")
        print(_LINE)
        return

    # Categorise results
    decided = [r for r in results if r["status_code"] == 200]
    escalated = [r for r in results if r["status_code"] == 202]
    errors = [r for r in results if r["status_code"] not in (200, 202)]

    print(f"  Total Applications:  {total}")
    print(f"  Decided (200):       {len(decided)}")
    print(f"  Escalated (202):     {len(escalated)}")
    print(f"  Errors:              {len(errors)}")
    print()

    # Decision distribution
    if decided:
        decisions = Counter(
            r["result"].get("decision", "UNKNOWN") for r in decided
        )
        print("DECISION DISTRIBUTION:")
        for decision, count in decisions.most_common():
            pct = count / total * 100
            bar = "#" * int(pct / 2)
            print(f"  {decision:<30s} {count:>3d} ({pct:5.1f}%) {bar}")
        print()

    # Escalation rate
    escalation_rate = len(escalated) / total * 100 if total > 0 else 0.0
    print(f"ESCALATION RATE: {escalation_rate:.1f}% ({len(escalated)}/{total})")
    if escalated:
        print("  Escalated companies:")
        for r in escalated:
            print(f"    - {r['company']}")
    print()

    # Error rate
    error_rate = len(errors) / total * 100 if total > 0 else 0.0
    print(f"ERROR RATE: {error_rate:.1f}% ({len(errors)}/{total})")
    if errors:
        print("  Failed companies:")
        for r in errors:
            err_msg = r["result"].get("error", r["result"].get("detail", "unknown"))
            print(f"    - {r['company']}: {err_msg}")
    print()

    # Grounding scores
    _print_grounding_summary(decided)

    # Confidence scores
    if decided:
        confidences = [
            r["result"].get("confidence_score", 0.0) for r in decided
        ]
        confidences = [c for c in confidences if isinstance(c, (int, float))]
        if confidences:
            print("CONFIDENCE SCORES:")
            print(f"  Mean:    {statistics.mean(confidences):.2%}")
            print(f"  Median:  {statistics.median(confidences):.2%}")
            if len(confidences) >= 2:
                print(f"  Std Dev: {statistics.stdev(confidences):.2%}")
            print(f"  Min:     {min(confidences):.2%}")
            print(f"  Max:     {max(confidences):.2%}")
            print()

    # Timing
    request_times = [r["elapsed"] for r in results]
    print("TIMING:")
    print(f"  Total elapsed:       {total_elapsed:.1f}s")
    print(f"  Per-request mean:    {statistics.mean(request_times):.1f}s")
    print(f"  Per-request median:  {statistics.median(request_times):.1f}s")
    print(f"  Fastest:             {min(request_times):.1f}s")
    print(f"  Slowest:             {max(request_times):.1f}s")
    print(f"  Throughput:          {total / total_elapsed:.2f} apps/s" if total_elapsed > 0 else "")
    print()

    print(_LINE)


def _print_grounding_summary(decided: list[dict]) -> None:
    """Extract and summarise grounding scores across all decided results."""
    # Collect scores by checkpoint
    checkpoint_scores: dict[str, list[float]] = {}
    for r in decided:
        for gs in r["result"].get("grounding_scores", []):
            checkpoint = gs.get("checkpoint", gs.get("checkpoint_name", "unknown"))
            score = gs.get("score", gs.get("grounding_score", None))
            if score is not None:
                try:
                    checkpoint_scores.setdefault(checkpoint, []).append(float(score))
                except (TypeError, ValueError):
                    pass

    if not checkpoint_scores:
        print("GROUNDING SCORES: No grounding data available")
        print()
        return

    print("GROUNDING SCORES (averages by checkpoint):")
    all_scores: list[float] = []
    for checkpoint, scores in sorted(checkpoint_scores.items()):
        avg = statistics.mean(scores)
        grounded_count = sum(1 for s in scores if s >= 0.7)
        total_count = len(scores)
        all_scores.extend(scores)
        print(
            f"  {checkpoint:<30s} avg={avg:.3f}  "
            f"grounded={grounded_count}/{total_count}"
        )

    if all_scores:
        overall_avg = statistics.mean(all_scores)
        overall_grounded = sum(1 for s in all_scores if s >= 0.7)
        print(f"  {'OVERALL':<30s} avg={overall_avg:.3f}  "
              f"grounded={overall_grounded}/{len(all_scores)}")
    print()


# =====================================================================
# CLI entry point
# =====================================================================


def main() -> None:
    """Parse CLI arguments, load applications, process batch, show summary."""
    parser = argparse.ArgumentParser(
        description="Demo: Batch loan application credit risk assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/demo_batch.py                        # All 50 applications\n"
            "  python scripts/demo_batch.py --max 10               # First 10 only\n"
            "  python scripts/demo_batch.py --url http://host:8000 # Custom API URL\n"
        ),
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=0,
        dest="max_count",
        help="Max applications to process, 0 = all (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(_PROJECT_ROOT, "data", "synthetic"),
        help="Path to synthetic data directory (default: data/synthetic)",
    )

    args = parser.parse_args()

    print("Enterprise Agentic Orchestrator -- Batch Processing Demo")
    print()

    # 1. Wait for API health
    if not wait_for_health(args.url):
        print("ERROR: API is not available. Is the server running?")
        print("  Start with: docker-compose up")
        sys.exit(1)

    # 2. Load applications
    applications = load_all_applications(args.data_dir, args.max_count)
    print(f"Loaded {len(applications)} applications from {args.data_dir}")
    print()

    # 3. Process batch
    print("Processing applications:")
    batch_start = time.monotonic()
    results = process_batch(args.url, applications, args.timeout)
    batch_elapsed = time.monotonic() - batch_start

    # 4. Print summary
    print_batch_summary(results, batch_elapsed)


if __name__ == "__main__":
    main()
