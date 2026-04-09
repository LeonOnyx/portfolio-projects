"""Custom Prometheus metrics for credit risk assessment domain monitoring."""

import logging
import time
from contextlib import contextmanager

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

logger = logging.getLogger(__name__)

DECISION_COUNTER = Counter(
    "credit_decision_total",
    "Total credit decisions by outcome",
    labelnames=["outcome"],
)

GROUNDING_HISTOGRAM = Histogram(
    "grounding_score",
    "Distribution of grounding scores across assessments",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

ESCALATION_COUNTER = Counter(
    "credit_escalation_total",
    "Total assessments escalated to human review",
)

ASSESSMENT_LATENCY = Histogram(
    "assessment_duration_seconds",
    "End-to-end assessment processing latency",
    buckets=[1, 5, 10, 30, 60, 120, 300],
)


@contextmanager
def track_assessment_latency():
    """Context manager that observes elapsed time on ASSESSMENT_LATENCY."""
    start = time.monotonic()
    yield
    duration = time.monotonic() - start
    ASSESSMENT_LATENCY.observe(duration)


def record_assessment_metrics(result: dict) -> None:
    """Extract domain metrics from an orchestrator result and record them.

    Increments DECISION_COUNTER with the final decision outcome,
    ESCALATION_COUNTER if the result was escalated, and observes each
    grounding score on GROUNDING_HISTOGRAM.
    """
    decision = result.get("final_decision", "UNKNOWN")
    DECISION_COUNTER.labels(outcome=decision).inc()

    if result.get("requires_escalation", False):
        ESCALATION_COUNTER.inc()

    grounding_scores = result.get("grounding_scores", []) or []
    for entry in grounding_scores:
        if isinstance(entry, dict):
            score = entry.get("grounding_score") or entry.get("score")
            if score is not None:
                try:
                    GROUNDING_HISTOGRAM.observe(float(score))
                except (ValueError, TypeError):
                    pass


def get_metrics_text() -> tuple[bytes, str]:
    """Return Prometheus metrics as (body_bytes, content_type)."""
    return generate_latest(), CONTENT_TYPE_LATEST


def setup_instrumentator(app):
    """Attach prometheus-fastapi-instrumentator to a FastAPI app.

    Silently skips if the package is not installed (HTTP-level metrics
    will be unavailable, but the app starts normally).
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        instrumentator = Instrumentator(
            excluded_handlers=["/api/v1/health", "/api/v1/metrics"],
        )
        instrumentator.instrument(app)
        logger.info("prometheus-fastapi-instrumentator attached")
    except ImportError:
        logger.warning(
            "prometheus-fastapi-instrumentator not installed; "
            "HTTP-level metrics unavailable"
        )
    except Exception as exc:
        logger.warning("Instrumentator setup failed (%s)", exc)
