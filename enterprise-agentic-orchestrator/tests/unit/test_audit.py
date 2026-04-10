"""Unit tests for audit trail hash chain (TEST-09).

Tests compute_content_hash, AuditTrail lifecycle (init, add_entry,
verify_chain, tamper detection, export_json, filtering), and edge cases.
"""

from __future__ import annotations

import json

import pytest

from src.governance.audit import AuditTrail, compute_content_hash


# ---------------------------------------------------------------------------
# compute_content_hash
# ---------------------------------------------------------------------------

class TestComputeContentHash:
    """Tests for compute_content_hash() utility."""

    def test_known_input_consistency(self):
        """Same input always produces same hash."""
        hash1 = compute_content_hash("hello world")
        hash2 = compute_content_hash("hello world")
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex digest

    def test_different_inputs_differ(self):
        hash1 = compute_content_hash("hello")
        hash2 = compute_content_hash("world")
        assert hash1 != hash2


# ---------------------------------------------------------------------------
# AuditTrail init
# ---------------------------------------------------------------------------

class TestAuditTrailInit:
    """Tests for AuditTrail construction."""

    def test_init(self):
        trail = AuditTrail(request_id="REQ-001")
        assert trail.request_id == "REQ-001"
        assert trail.entries == []
        assert trail._previous_hash == "GENESIS"

    def test_len_empty(self):
        trail = AuditTrail(request_id="REQ-001")
        assert len(trail) == 0


# ---------------------------------------------------------------------------
# add_entry
# ---------------------------------------------------------------------------

class TestAddEntry:
    """Tests for AuditTrail.add_entry()."""

    def test_entry_added(self):
        trail = AuditTrail(request_id="REQ-001")
        entry = trail.add_entry(stage="INTAKE", action="input_received")
        assert len(trail) == 1
        assert entry.stage == "INTAKE"
        assert entry.action == "input_received"
        assert entry.hash != ""
        assert entry.hash != "GENESIS"

    def test_second_entry_different_hash(self):
        trail = AuditTrail(request_id="REQ-001")
        entry1 = trail.add_entry(stage="INTAKE", action="input_received")
        entry2 = trail.add_entry(stage="ANALYSIS", action="analyst_started")
        assert entry1.hash != entry2.hash

    def test_entry_with_agent(self):
        trail = AuditTrail(request_id="REQ-001")
        entry = trail.add_entry(
            stage="ANALYSIS",
            action="analyst_started",
            agent="financial_analyst",
        )
        assert entry.agent == "financial_analyst"

    def test_entry_with_details(self):
        trail = AuditTrail(request_id="REQ-001")
        entry = trail.add_entry(
            stage="ANALYSIS",
            action="analyst_completed",
            details={"credit_score": 72},
        )
        assert entry.details["credit_score"] == 72

    def test_entry_with_hashes(self):
        trail = AuditTrail(request_id="REQ-001")
        entry = trail.add_entry(
            stage="INTAKE",
            action="input_received",
            input_hash="abc123",
            output_hash="def456",
        )
        assert entry.details["input_hash"] == "abc123"
        assert entry.details["output_hash"] == "def456"


# ---------------------------------------------------------------------------
# verify_chain
# ---------------------------------------------------------------------------

class TestVerifyChain:
    """Tests for AuditTrail.verify_chain()."""

    def test_valid_chain(self):
        trail = AuditTrail(request_id="REQ-001")
        trail.add_entry(stage="INTAKE", action="input_received")
        trail.add_entry(stage="ANALYSIS", action="analyst_started")
        trail.add_entry(stage="ANALYSIS", action="analyst_completed")
        assert trail.verify_chain() is True

    def test_tampered_action(self):
        trail = AuditTrail(request_id="REQ-001")
        trail.add_entry(stage="INTAKE", action="input_received")
        trail.add_entry(stage="ANALYSIS", action="analyst_started")
        trail.add_entry(stage="ANALYSIS", action="analyst_completed")

        # Tamper with middle entry's action (immutable field in hash)
        trail.entries[1].action = "TAMPERED"
        assert trail.verify_chain() is False

    def test_tampered_timestamp(self):
        trail = AuditTrail(request_id="REQ-001")
        trail.add_entry(stage="INTAKE", action="input_received")

        from datetime import datetime, timedelta

        trail.entries[0].timestamp = datetime(2020, 1, 1)
        assert trail.verify_chain() is False

    def test_empty_trail_valid(self):
        trail = AuditTrail(request_id="REQ-001")
        assert trail.verify_chain() is True


# ---------------------------------------------------------------------------
# export_json
# ---------------------------------------------------------------------------

class TestExportJSON:
    """Tests for AuditTrail.export_json()."""

    def test_export_valid_json(self):
        trail = AuditTrail(request_id="REQ-001")
        trail.add_entry(stage="INTAKE", action="input_received")
        trail.add_entry(stage="ANALYSIS", action="analyst_started")

        exported = trail.export_json()
        data = json.loads(exported)

        assert data["request_id"] == "REQ-001"
        assert data["entry_count"] == 2
        assert data["chain_valid"] is True
        assert len(data["entries"]) == 2

    def test_export_tampered_chain_valid_false(self):
        trail = AuditTrail(request_id="REQ-001")
        trail.add_entry(stage="INTAKE", action="input_received")
        trail.entries[0].action = "TAMPERED"

        exported = trail.export_json()
        data = json.loads(exported)
        assert data["chain_valid"] is False

    def test_export_empty_trail(self):
        trail = AuditTrail(request_id="REQ-001")
        exported = trail.export_json()
        data = json.loads(exported)

        assert data["entry_count"] == 0
        assert data["chain_valid"] is True
        assert data["entries"] == []


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestFiltering:
    """Tests for get_entries_by_stage() and get_entries_by_agent()."""

    def test_get_entries_by_stage(self):
        trail = AuditTrail(request_id="REQ-001")
        trail.add_entry(stage="INTAKE", action="input_received")
        trail.add_entry(stage="ANALYSIS", action="analyst_started")
        trail.add_entry(stage="ANALYSIS", action="analyst_completed")
        trail.add_entry(stage="REVIEW", action="reviewer_started")

        analysis_entries = trail.get_entries_by_stage("ANALYSIS")
        assert len(analysis_entries) == 2
        assert all(e.stage == "ANALYSIS" for e in analysis_entries)

    def test_get_entries_by_agent(self):
        trail = AuditTrail(request_id="REQ-001")
        trail.add_entry(stage="ANALYSIS", action="started", agent="analyst")
        trail.add_entry(stage="ANALYSIS", action="completed", agent="analyst")
        trail.add_entry(stage="REVIEW", action="started", agent="reviewer")

        analyst_entries = trail.get_entries_by_agent("analyst")
        assert len(analyst_entries) == 2
        assert all(e.agent == "analyst" for e in analyst_entries)

    def test_filter_returns_empty_for_unknown(self):
        trail = AuditTrail(request_id="REQ-001")
        trail.add_entry(stage="INTAKE", action="input_received")

        assert trail.get_entries_by_stage("NONEXISTENT") == []
        assert trail.get_entries_by_agent("nobody") == []

    def test_len_matches_entries(self):
        trail = AuditTrail(request_id="REQ-001")
        trail.add_entry(stage="INTAKE", action="a1")
        trail.add_entry(stage="INTAKE", action="a2")
        trail.add_entry(stage="INTAKE", action="a3")
        assert len(trail) == 3
