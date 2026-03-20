"""Tests for FindingsDatabase — Parquet persistence and query API."""

from __future__ import annotations

import pandas as pd
import pytest

from correlation_engine.discovery.findings import Finding
from correlation_engine.store.findings_db import FindingsDatabase


# ── helpers ────────────────────────────────────────────────────────

def _make_finding(**overrides) -> Finding:
    """Create a Finding with sensible defaults, overridable."""
    defaults = dict(
        scan_id="scan-001",
        scanned_at="2025-06-15T12:00:00+00:00",
        series_a="GDP",
        series_b="SPY",
        series_a_name="Gross Domestic Product",
        series_b_name="S&P 500 ETF",
        correlation=0.85,
        correlation_method="pearson",
        optimal_lag=3,
        lag_correlation=0.88,
        granger_p_value=0.02,
        granger_direction="a_causes_b",
        rolling_zscore=2.5,
        regime_change_detected=True,
        trigger_types=["high_correlation", "regime_change"],
        interestingness_score=0.78,
        is_new=True,
        template_summary="Strong correlation found.",
        llm_summary=None,
        lookback_days=3650,
        frequency="M",
    )
    defaults.update(overrides)
    return Finding(**defaults)


# ── tests ──────────────────────────────────────────────────────────

class TestFindingsDatabase:

    @pytest.fixture()
    def db(self, tmp_path):
        return FindingsDatabase(base_path=tmp_path / "findings")

    def test_save_and_load_latest(self, db):
        """Save findings then load them back."""
        findings = [
            _make_finding(series_a="A", series_b="B", interestingness_score=0.9),
            _make_finding(series_a="C", series_b="D", interestingness_score=0.5),
            _make_finding(series_a="E", series_b="F", interestingness_score=0.7),
        ]
        db.save_findings(findings, scan_id="s1", scanned_at="2025-06-15T12:00:00+00:00")

        loaded = db.load_latest(n=2)
        assert len(loaded) == 2
        assert loaded[0].interestingness_score >= loaded[1].interestingness_score

    def test_load_latest_empty_db(self, db):
        """Empty DB should return empty list."""
        assert db.load_latest() == []

    def test_save_creates_parquet(self, db):
        """Save should create a .parquet file."""
        findings = [_make_finding()]
        db.save_findings(findings, scan_id="s1", scanned_at="2025-06-15T12:00:00+00:00")

        parquets = list(db.base_path.glob("*.parquet"))
        assert len(parquets) == 1

    def test_index_updated(self, db):
        """Save should update the index.json."""
        db.save_findings([_make_finding()], scan_id="s1", scanned_at="2025-06-15T12:00:00+00:00")
        db.save_findings([_make_finding()], scan_id="s2", scanned_at="2025-06-16T12:00:00+00:00")

        scans = db.load_all_scans()
        assert len(scans) == 2
        assert "scan_id" in scans.columns

    def test_load_pair_history_both_orderings(self, db):
        """Pair history should find (A, B) regardless of storage order."""
        f1 = _make_finding(series_a="GDP", series_b="SPY", interestingness_score=0.8)
        f2 = _make_finding(series_a="GDP", series_b="SPY", interestingness_score=0.6)
        db.save_findings([f1], scan_id="s1", scanned_at="2025-06-15T12:00:00+00:00")
        db.save_findings([f2], scan_id="s2", scanned_at="2025-06-16T12:00:00+00:00")

        # Query in original order
        hist1 = db.load_pair_history("GDP", "SPY")
        assert len(hist1) == 2

        # Query in reversed order
        hist2 = db.load_pair_history("SPY", "GDP")
        assert len(hist2) == 2

    def test_load_pair_history_no_match(self, db):
        """Pair not in DB should return empty DataFrame."""
        db.save_findings([_make_finding()], scan_id="s1", scanned_at="2025-06-15T12:00:00+00:00")
        hist = db.load_pair_history("AAPL", "MSFT")
        assert len(hist) == 0

    def test_was_seen_before_true(self, db):
        """Returns True after pair has been saved."""
        db.save_findings(
            [_make_finding(series_a="GDP", series_b="SPY")],
            scan_id="s1",
            scanned_at="2025-06-15T12:00:00+00:00",
        )
        assert db.was_seen_before("GDP", "SPY") is True
        assert db.was_seen_before("SPY", "GDP") is True  # reversed order

    def test_was_seen_before_false(self, db):
        """Returns False for a brand-new pair."""
        assert db.was_seen_before("AAPL", "MSFT") is False

    def test_was_seen_before_lookback_limit(self, db):
        """Only checks the last N scans."""
        # Save pair in scan 1, but with lookback=1, scan 2 doesn't have it
        db.save_findings(
            [_make_finding(series_a="A", series_b="B")],
            scan_id="s1", scanned_at="2025-01-01T00:00:00+00:00",
        )
        db.save_findings(
            [_make_finding(series_a="C", series_b="D")],
            scan_id="s2", scanned_at="2025-01-02T00:00:00+00:00",
        )

        # lookback=1 → only scan s2
        assert db.was_seen_before("A", "B", lookback_scans=1) is False
        # lookback=2 → includes scan s1
        assert db.was_seen_before("A", "B", lookback_scans=2) is True

    def test_load_all_scans_empty(self, db):
        """Empty DB returns empty DataFrame with correct columns."""
        scans = db.load_all_scans()
        assert isinstance(scans, pd.DataFrame)
        assert len(scans) == 0

    def test_get_scan_count(self, db):
        assert db.get_scan_count() == 0
        db.save_findings([_make_finding()], scan_id="s1", scanned_at="2025-01-01T00:00:00+00:00")
        assert db.get_scan_count() == 1

    def test_directory_auto_creation(self, tmp_path):
        """DB should create the base directory if it doesn't exist."""
        deep_path = tmp_path / "a" / "b" / "c" / "findings"
        db = FindingsDatabase(base_path=deep_path)
        assert deep_path.exists()

    def test_round_trip_finding_integrity(self, db):
        """Save and load should preserve all Finding fields."""
        original = _make_finding(
            series_a="GDP",
            series_b="UNRATE",
            correlation=-0.72,
            optimal_lag=6,
            lag_correlation=-0.80,
            granger_p_value=0.003,
            granger_direction="a_causes_b",
            rolling_zscore=2.8,
            regime_change_detected=True,
            trigger_types=["high_correlation", "regime_change", "granger_causality"],
            interestingness_score=0.88,
            is_new=True,
            template_summary="GDP and UNRATE are inversely correlated.",
            llm_summary=None,
        )
        db.save_findings([original], scan_id="s1", scanned_at="2025-06-15T12:00:00+00:00")
        loaded = db.load_latest(n=1)[0]

        assert loaded.series_a == original.series_a
        assert loaded.series_b == original.series_b
        assert loaded.correlation == pytest.approx(original.correlation)
        assert loaded.optimal_lag == original.optimal_lag
        assert loaded.trigger_types == original.trigger_types
        assert loaded.granger_p_value == pytest.approx(original.granger_p_value)
        assert loaded.regime_change_detected is True
        assert loaded.is_new is True

    def test_multiple_findings_per_scan(self, db):
        """Save many findings in one scan and load them all."""
        findings = [
            _make_finding(series_a=f"S{i}", series_b=f"S{i+1}", interestingness_score=i/100)
            for i in range(50)
        ]
        db.save_findings(findings, scan_id="s1", scanned_at="2025-06-15T12:00:00+00:00")

        loaded = db.load_latest(n=100)
        assert len(loaded) == 50
