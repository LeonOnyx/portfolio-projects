#!/usr/bin/env python
"""Demo: Single loan application credit risk assessment.

Processes one loan application through the Enterprise Agentic Orchestrator
API and prints formatted output showing the decision, risk metrics,
grounding scores, and compliance results.

Usage:
    python scripts/demo_single.py                    # First application
    python scripts/demo_single.py --index 5          # 6th application
    python scripts/demo_single.py --url http://host:8000  # Custom API URL

Requires the API to be running (docker-compose up).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

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


def load_application(data_dir: str, index: int) -> dict:
    """Load a single loan application from the synthetic dataset.

    Parameters
    ----------
    data_dir:
        Path to the directory containing ``loan_applications.json``.
    index:
        Zero-based index of the application to load.

    Returns
    -------
    dict
        The application payload with ``application_id`` and
        ``submitted_at`` stripped (the API generates those server-side).

    Raises
    ------
    SystemExit
        If the file does not exist or the index is out of bounds.
    """
    filepath = os.path.join(data_dir, "loan_applications.json")
    if not os.path.isfile(filepath):
        print(f"ERROR: Data file not found: {filepath}")
        print("Run 'python scripts/generate_data.py' first to create synthetic data.")
        sys.exit(1)

    with open(filepath) as fh:
        applications = json.load(fh)

    if index < 0 or index >= len(applications):
        print(f"ERROR: Index {index} out of range (0-{len(applications) - 1})")
        sys.exit(1)

    app = applications[index]
    # Strip server-generated fields
    app.pop("application_id", None)
    app.pop("submitted_at", None)
    return app


# =====================================================================
# API submission
# =====================================================================


def submit_application(
    base_url: str, application: dict, timeout: int
) -> tuple[int, dict]:
    """POST a loan application to the assessment endpoint.

    Returns
    -------
    tuple[int, dict]
        ``(status_code, response_json)``.  On connection or timeout
        errors the status code is ``0`` and the dict contains an
        ``"error"`` key describing the failure.
    """
    url = f"{base_url}/api/v1/assess"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=application)
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text}
            return resp.status_code, body
    except httpx.ConnectError as exc:
        return 0, {"error": f"Connection refused: {exc}"}
    except httpx.TimeoutException as exc:
        return 0, {"error": f"Request timed out after {timeout}s: {exc}"}
    except Exception as exc:
        return 0, {"error": f"Unexpected error: {exc}"}


# =====================================================================
# Output formatting
# =====================================================================

_LINE = "=" * 70


def print_single_result(status_code: int, result: dict) -> None:
    """Print a formatted summary of the assessment result."""
    print()
    print(_LINE)
    print("CREDIT RISK ASSESSMENT RESULT")
    print(_LINE)

    # --- Connection / server error ---
    if status_code == 0:
        print(f"  ERROR: {result.get('error', 'Unknown error')}")
        print(_LINE)
        return

    if status_code >= 400:
        _print_error_result(status_code, result)
        return

    # --- Escalation (202) ---
    if status_code == 202:
        _print_escalation_result(result)
        return

    # --- Decided (200) ---
    _print_decided_result(status_code, result)


def _print_decided_result(status_code: int, result: dict) -> None:
    """Format output for a 200 (decided) response."""
    request_id = result.get("request_id", "N/A")
    application_id = result.get("application_id", "N/A")
    decision = result.get("decision", "N/A")
    confidence = result.get("confidence_score", 0.0)

    print(f"  Request ID:    {request_id}")
    print(f"  Application:   {application_id}")
    print(f"  Status Code:   {status_code}")
    print(f"  Decision:      {decision}")
    print(f"  Confidence:    {confidence:.2%}")
    print()

    # Reasoning
    reasoning = result.get("reasoning", "")
    if reasoning:
        print("REASONING:")
        # Word-wrap reasoning at ~68 chars
        for line in _wrap_text(reasoning, 66):
            print(f"  {line}")
        print()

    # Grounding scores
    grounding_scores = result.get("grounding_scores", [])
    if grounding_scores:
        print("GROUNDING SCORES:")
        for gs in grounding_scores:
            checkpoint = gs.get("checkpoint", gs.get("checkpoint_name", "unknown"))
            score = gs.get("score", gs.get("grounding_score", 0.0))
            try:
                score_val = float(score)
                is_grounded = score_val >= 0.7
                label = "grounded" if is_grounded else "UNGROUNDED"
                print(f"  {checkpoint}: {score_val:.3f} ({label})")
            except (TypeError, ValueError):
                print(f"  {checkpoint}: {score}")
        print()

    # Compliance -- extract from audit trail
    _print_compliance_section(result.get("audit_trail", []))

    # Audit trail
    _print_audit_trail(result.get("audit_trail", []))

    print(_LINE)


def _print_escalation_result(result: dict) -> None:
    """Format output for a 202 (escalated) response."""
    request_id = result.get("request_id", "N/A")
    application_id = result.get("application_id", "N/A")
    status = result.get("status", "escalated")
    message = result.get("message", "")

    print(f"  Request ID:    {request_id}")
    print(f"  Application:   {application_id}")
    print(f"  Status Code:   202")
    print(f"  Status:        {status}")
    print(f"  Message:       {message}")
    print()

    reasoning = result.get("reasoning", "")
    if reasoning:
        print("REASONING:")
        for line in _wrap_text(reasoning, 66):
            print(f"  {line}")
        print()

    triggers = result.get("escalation_triggers", [])
    if triggers:
        print("ESCALATION TRIGGERS:")
        for t in triggers:
            print(f"  - {t}")
        print()

    _print_audit_trail(result.get("audit_trail", []))

    print(_LINE)


def _print_error_result(status_code: int, result: dict) -> None:
    """Format output for 4xx/5xx error responses."""
    print(f"  Status Code:   {status_code}")
    if status_code == 400:
        print(f"  Error:         {result.get('error', 'validation_error')}")
        print(f"  Detail:        {result.get('detail', 'N/A')}")
        errors = result.get("errors", [])
        if errors:
            print(f"  Errors ({len(errors)}):")
            for e in errors[:5]:
                if isinstance(e, dict):
                    print(f"    - {e.get('error', e)}")
                else:
                    print(f"    - {e}")
    else:
        print(f"  Error:         {result.get('error', 'internal_error')}")
        print(f"  Detail:        {result.get('detail', 'N/A')}")
    print(_LINE)


def _print_compliance_section(audit_trail: list[dict]) -> None:
    """Extract and print compliance-related audit trail entries."""
    compliance_entries = [
        e for e in audit_trail
        if isinstance(e, dict) and str(e.get("stage", "")).upper() == "COMPLIANCE"
    ]
    if compliance_entries:
        print("COMPLIANCE:")
        for entry in compliance_entries:
            action = entry.get("action", "N/A")
            details = entry.get("details", "")
            print(f"  [{action}]")
            if details:
                if isinstance(details, dict):
                    for k, v in details.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"    {details}")
        print()


def _print_audit_trail(audit_trail: list[dict]) -> None:
    """Print the audit trail with truncation for long trails."""
    if not audit_trail:
        return

    n = len(audit_trail)
    print(f"AUDIT TRAIL ({n} entries):")

    if n <= 15:
        entries_to_show = audit_trail
    else:
        entries_to_show = audit_trail[:10]
        print("  ... (showing first 10)")

    for entry in entries_to_show:
        if not isinstance(entry, dict):
            continue
        stage = entry.get("stage", "?")
        action = entry.get("action", "?")
        duration = entry.get("duration_ms", 0)
        try:
            duration_val = float(duration)
            print(f"  [{stage}] {action} ({duration_val:.0f}ms)")
        except (TypeError, ValueError):
            print(f"  [{stage}] {action}")

    if n > 15:
        print(f"  ... ({n - 15} entries omitted)")
        print("  ... (showing last 5)")
        for entry in audit_trail[-5:]:
            if not isinstance(entry, dict):
                continue
            stage = entry.get("stage", "?")
            action = entry.get("action", "?")
            duration = entry.get("duration_ms", 0)
            try:
                duration_val = float(duration)
                print(f"  [{stage}] {action} ({duration_val:.0f}ms)")
            except (TypeError, ValueError):
                print(f"  [{stage}] {action}")

    print()


def _wrap_text(text: str, width: int) -> list[str]:
    """Simple word-wrap using stdlib only."""
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + (1 if current else 0) > width:
            if current:
                lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + (1 if len(current) > 1 else 0)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


# =====================================================================
# CLI entry point
# =====================================================================


def main() -> None:
    """Parse CLI arguments, load application, submit, and display result."""
    parser = argparse.ArgumentParser(
        description="Demo: Single loan application credit risk assessment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/demo_single.py                    # First application\n"
            "  python scripts/demo_single.py --index 5          # 6th application\n"
            "  python scripts/demo_single.py --url http://host:8000  # Custom API URL\n"
        ),
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Application index from loan_applications.json (default: 0)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(_PROJECT_ROOT, "data", "synthetic"),
        help="Path to synthetic data directory (default: data/synthetic)",
    )

    args = parser.parse_args()

    print("Enterprise Agentic Orchestrator -- Single Application Demo")
    print()

    # 1. Wait for API health
    if not wait_for_health(args.url):
        print("ERROR: API is not available. Is the server running?")
        print("  Start with: docker-compose up")
        sys.exit(1)

    # 2. Load application
    print(f"Loading application at index {args.index} from {args.data_dir}...")
    application = load_application(args.data_dir, args.index)

    company = application.get("applicant", {}).get("company_name", "Unknown")
    amount = application.get("loan", {}).get("amount_requested", "N/A")
    sector = application.get("applicant", {}).get("sector", "N/A")
    print(f"  Company:  {company}")
    print(f"  Sector:   {sector}")
    print(f"  Amount:   GBP {amount}")
    print()

    # 3. Submit to API
    print(f"Submitting to {args.url}/api/v1/assess (timeout={args.timeout}s)...")
    start = time.monotonic()
    status_code, result = submit_application(args.url, application, args.timeout)
    elapsed = time.monotonic() - start
    print(f"  Response received in {elapsed:.1f}s")

    # 4. Print formatted result
    print_single_result(status_code, result)


if __name__ == "__main__":
    main()
