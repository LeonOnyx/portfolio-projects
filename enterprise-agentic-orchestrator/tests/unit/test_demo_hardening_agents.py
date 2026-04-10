"""Tests for Phase 11 agent and pipeline hardening fixes (P0-4, P0-5, P1-11, P1-12).

Covers:
- P0-4: ReviewerAgent extracts sources, not hardcoded empty list
- P0-5: Compliance error path returns overall_passed=False, not empty dict
- P1-11: Input sanitization via _sanitize_field in all agent prompt builders
- P1-12: BiasChecker wired post-hoc in orchestrator pipeline nodes
"""

import pytest


# -----------------------------------------------------------------------
# P0-4: Reviewer must extract sources, not hardcode empty list
# -----------------------------------------------------------------------


def test_reviewer_sources_not_hardcoded_empty():
    """P0-4: ReviewerAgent must extract sources, not hardcode empty list."""
    import ast
    import inspect

    from src.agents.reviewer import ReviewerAgent

    source = inspect.getsource(ReviewerAgent.execute)
    # Verify no 'sources_used=[]' literal in the AgentResponse construction
    assert "sources_used=[]" not in source, (
        "ReviewerAgent still has hardcoded sources_used=[]"
    )


# -----------------------------------------------------------------------
# P0-5: Compliance error path must return overall_passed=False
# -----------------------------------------------------------------------


def test_compliance_error_output_has_overall_passed():
    """P0-5: Compliance error path must return overall_passed=False, not empty dict."""
    source = open("src/agents/compliance.py").read()
    # Find the execute method's except block (the one with 'as exc')
    idx = source.find("except Exception as exc:")
    assert idx != -1, "execute method's except block not found"
    # Extract until the next method definition (def _build...)
    next_def = source.find("\n    def ", idx)
    error_section = source[idx:next_def] if next_def != -1 else source[idx:]
    assert "overall_passed" in error_section, (
        "Compliance error path must include overall_passed in output"
    )
    assert '"overall_passed": False' in error_section or "'overall_passed': False" in error_section, (
        "Compliance error path must set overall_passed to False"
    )


# -----------------------------------------------------------------------
# P1-12: BiasChecker wired in orchestrator pipeline nodes
# -----------------------------------------------------------------------


def test_bias_checker_wired_in_orchestrator_nodes():
    """P1-12: BiasChecker must be called post-hoc in orchestrator nodes."""
    source = open("src/orchestrator_nodes.py").read()
    assert "BiasChecker" in source, (
        "BiasChecker not referenced in orchestrator_nodes"
    )
    assert "_run_bias_check" in source, "Missing _run_bias_check helper"
    # Verify it's called in all three agent nodes
    assert source.count("bias_entries") >= 3, (
        "bias_entries should appear in analysis_node, review_node, and compliance_node"
    )


def test_run_bias_check_helper():
    """P1-12: _run_bias_check returns audit entries for clean output."""
    from src.orchestrator_nodes import _run_bias_check

    entries = _run_bias_check(
        {"reasoning": "The credit score is 75 which is acceptable."}, "ANALYSIS"
    )
    assert isinstance(entries, list)
    assert len(entries) >= 1
    assert entries[0]["action"] in ("bias_check_passed", "bias_check_warning")


def test_run_bias_check_detects_bias():
    """P1-12: _run_bias_check flags protected characteristics."""
    from src.orchestrator_nodes import _run_bias_check

    entries = _run_bias_check(
        {"reasoning": "Rejected due to applicant age and gender."},
        "ANALYSIS",
    )
    assert len(entries) >= 1
    assert entries[0]["action"] == "bias_check_warning"
    assert entries[0]["details"]["bias_detected"] is True


# -----------------------------------------------------------------------
# P1-11: Input sanitization in all agent prompt builders
# -----------------------------------------------------------------------


def test_sanitize_field_strips_control_chars():
    """P1-11: _sanitize_field must strip control characters."""
    from src.agents.analyst import _sanitize_field

    result = _sanitize_field("Hello\x00World\x1fTest")
    assert "\x00" not in result
    assert "\x1f" not in result
    assert "Hello" in result
    assert "World" in result


def test_sanitize_field_limits_length():
    """P1-11: _sanitize_field must limit string length."""
    from src.agents.analyst import _sanitize_field

    long_input = "A" * 10000
    result = _sanitize_field(long_input, max_length=500)
    assert len(result) <= 500


def test_sanitize_field_escapes_backticks():
    """P1-11: _sanitize_field must escape triple backticks."""
    from src.agents.analyst import _sanitize_field

    result = _sanitize_field("```injected code```")
    assert "```" not in result


def test_all_agents_have_sanitize_field():
    """P1-11: All agent prompt builders must use _sanitize_field."""
    for agent_file in [
        "src/agents/analyst.py",
        "src/agents/reviewer.py",
        "src/agents/compliance.py",
    ]:
        source = open(agent_file).read()
        assert "_sanitize_field" in source, (
            f"{agent_file} missing _sanitize_field"
        )
