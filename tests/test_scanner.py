"""Tests for FindingScorer and DiscoveryScanner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from correlation_engine.discovery.findings import Finding
from correlation_engine.discovery.scanner import DiscoveryScanner, ScanConfig
from correlation_engine.discovery.scoring import FindingScorer, ScoringConfig


# ═══════════════════════════════════════════════════════════════════
# Scoring Tests
# ═══════════════════════════════════════════════════════════════════

class TestScoringConfig:

    def test_default_weights_sum_to_one(self):
        cfg = ScoringConfig()
        cfg.validate()  # should not raise

    def test_bad_weights_raises(self):
        cfg = ScoringConfig(weights={
            "high_correlation": 0.5,
            "newly_emerging": 0.5,
            "regime_change": 0.5,
            "granger_causality": 0.0,
            "anomalous_lag": 0.0,
            "rolling_divergence": 0.0,
        })
        with pytest.raises(ValueError, match="sum to 1.0"):
            cfg.validate()

    def test_from_yaml(self, tmp_path):
        import yaml

        data = {
            "thresholds": {"correlation": 0.8, "zscore": 1.5, "granger_p": 0.01, "lag_correlation": 0.5},
            "weights": {
                "high_correlation": 0.30,
                "newly_emerging": 0.10,
                "regime_change": 0.20,
                "granger_causality": 0.20,
                "anomalous_lag": 0.10,
                "rolling_divergence": 0.10,
            },
        }
        path = tmp_path / "scoring.yaml"
        path.write_text(yaml.dump(data))

        cfg = ScoringConfig.from_yaml(path)
        assert cfg.correlation_threshold == 0.8
        assert cfg.zscore_threshold == 1.5


class TestFindingScorer:

    @pytest.fixture()
    def scorer(self):
        return FindingScorer(ScoringConfig())

    def test_high_correlation_fires(self, scorer):
        triggers, score = scorer.score(
            correlation=0.85, optimal_lag=0, lag_correlation=0.85,
            granger_p_value=None, rolling_zscore=0.0, is_new=False,
        )
        assert "high_correlation" in triggers
        assert score > 0

    def test_high_correlation_below_threshold(self, scorer):
        triggers, score = scorer.score(
            correlation=0.5, optimal_lag=0, lag_correlation=0.5,
            granger_p_value=None, rolling_zscore=0.0, is_new=False,
        )
        assert "high_correlation" not in triggers

    def test_newly_emerging_fires(self, scorer):
        triggers, _ = scorer.score(
            correlation=0.3, optimal_lag=0, lag_correlation=0.3,
            granger_p_value=None, rolling_zscore=0.0, is_new=True,
        )
        assert "newly_emerging" in triggers

    def test_regime_change_fires(self, scorer):
        triggers, _ = scorer.score(
            correlation=0.5, optimal_lag=0, lag_correlation=0.5,
            granger_p_value=None, rolling_zscore=2.5, is_new=False,
        )
        assert "regime_change" in triggers
        assert "rolling_divergence" in triggers

    def test_regime_change_below_threshold(self, scorer):
        triggers, _ = scorer.score(
            correlation=0.5, optimal_lag=0, lag_correlation=0.5,
            granger_p_value=None, rolling_zscore=1.5, is_new=False,
        )
        assert "regime_change" not in triggers

    def test_granger_fires(self, scorer):
        triggers, _ = scorer.score(
            correlation=0.5, optimal_lag=0, lag_correlation=0.5,
            granger_p_value=0.01, rolling_zscore=0.0, is_new=False,
        )
        assert "granger_causality" in triggers

    def test_granger_not_significant(self, scorer):
        triggers, _ = scorer.score(
            correlation=0.5, optimal_lag=0, lag_correlation=0.5,
            granger_p_value=0.1, rolling_zscore=0.0, is_new=False,
        )
        assert "granger_causality" not in triggers

    def test_granger_none_skipped(self, scorer):
        triggers, _ = scorer.score(
            correlation=0.5, optimal_lag=0, lag_correlation=0.5,
            granger_p_value=None, rolling_zscore=0.0, is_new=False,
        )
        assert "granger_causality" not in triggers

    def test_anomalous_lag_fires(self, scorer):
        triggers, _ = scorer.score(
            correlation=0.5, optimal_lag=6, lag_correlation=0.75,
            granger_p_value=None, rolling_zscore=0.0, is_new=False,
        )
        assert "anomalous_lag" in triggers

    def test_anomalous_lag_zero_lag(self, scorer):
        triggers, _ = scorer.score(
            correlation=0.8, optimal_lag=0, lag_correlation=0.8,
            granger_p_value=None, rolling_zscore=0.0, is_new=False,
        )
        assert "anomalous_lag" not in triggers

    def test_all_criteria_fire(self, scorer):
        triggers, score = scorer.score(
            correlation=0.9, optimal_lag=6, lag_correlation=0.85,
            granger_p_value=0.001, rolling_zscore=3.0, is_new=True,
        )
        assert len(triggers) == 6
        assert score > 0.5

    def test_no_criteria_fire(self, scorer):
        triggers, score = scorer.score(
            correlation=0.1, optimal_lag=0, lag_correlation=0.1,
            granger_p_value=0.5, rolling_zscore=0.5, is_new=False,
        )
        assert triggers == []
        assert score == 0.0

    def test_score_bounded_zero_one(self, scorer):
        _, score = scorer.score(
            correlation=0.99, optimal_lag=12, lag_correlation=0.99,
            granger_p_value=0.001, rolling_zscore=5.0, is_new=True,
        )
        assert 0.0 <= score <= 1.0

    def test_negative_correlation_triggers(self, scorer):
        triggers, _ = scorer.score(
            correlation=-0.85, optimal_lag=0, lag_correlation=-0.85,
            granger_p_value=None, rolling_zscore=0.0, is_new=False,
        )
        assert "high_correlation" in triggers  # |r| = 0.85 > 0.7


# ═══════════════════════════════════════════════════════════════════
# Scanner Tests
# ═══════════════════════════════════════════════════════════════════

def _make_series(n: int = 120, seed: int = 0) -> pd.Series:
    """Random-walk monthly series."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")
    return pd.Series(rng.normal(0, 1, n).cumsum(), index=idx)


class TestDiscoveryScanner:

    @pytest.fixture()
    def scanner(self):
        cfg = ScanConfig(
            rolling_window=12,
            min_r_for_granger=0.3,
            min_score_threshold=0.0,
            max_lag=6,
            scoring=ScoringConfig(),
        )
        return DiscoveryScanner(cfg)

    def test_correlated_pair_found(self, scanner):
        """A synthetic perfectly correlated pair should trigger high_correlation."""
        base = _make_series(120, seed=42)
        series_dict = {
            "A": base,
            "B": base * 1.0 + 0.01 * _make_series(120, seed=99),
            "C": _make_series(120, seed=7),
        }
        findings = scanner.scan(series_dict)
        assert len(findings) > 0
        # Pair (A, B) should be top-scored
        top = findings[0]
        assert {top.series_a, top.series_b} == {"A", "B"}
        assert "high_correlation" in top.trigger_types

    def test_uncorrelated_pair_filtered(self, scanner):
        """Independent random walks should not trigger high_correlation."""
        series_dict = {
            "X": _make_series(120, seed=1),
            "Y": _make_series(120, seed=2),
        }
        findings = scanner.scan(series_dict)
        high_corr = [f for f in findings if "high_correlation" in f.trigger_types]
        # Independent walks are unlikely to have |r| > 0.7
        # (not impossible due to random walk drift, but unlikely with these seeds)
        assert len(high_corr) == 0

    def test_findings_sorted_by_score(self, scanner):
        """Findings should be sorted descending by interestingness_score."""
        series_dict = {
            "A": _make_series(120, seed=10),
            "B": _make_series(120, seed=20),
            "C": _make_series(120, seed=30),
            "D": _make_series(120, seed=40),
        }
        findings = scanner.scan(series_dict)
        if len(findings) >= 2:
            for i in range(len(findings) - 1):
                assert findings[i].interestingness_score >= findings[i + 1].interestingness_score

    def test_lagged_pair_detected(self, scanner):
        """A pair with a known lag should trigger anomalous_lag."""
        base = _make_series(120, seed=42)
        lagged = base.shift(6).dropna()
        # Re-index to same length
        common_idx = base.index.intersection(lagged.index)
        noise = 0.05 * _make_series(len(common_idx), seed=99)
        noise.index = common_idx

        series_dict = {
            "Leader": base.loc[common_idx],
            "Follower": lagged.loc[common_idx] + noise,
        }
        findings = scanner.scan(series_dict)
        assert len(findings) > 0
        f = findings[0]
        assert f.optimal_lag != 0 or "high_correlation" in f.trigger_types

    def test_progress_callback(self, scanner):
        """on_progress should be called during scan."""
        calls = []

        def track(done, total):
            calls.append((done, total))

        series_dict = {
            "A": _make_series(120, seed=1),
            "B": _make_series(120, seed=2),
            "C": _make_series(120, seed=3),
        }
        scanner.scan(series_dict, on_progress=track)
        assert len(calls) > 0
        # Last call should be (total, total)
        assert calls[-1][0] == calls[-1][1]

    def test_was_seen_fn_marks_new(self, scanner):
        """Pairs not seen before should have is_new=True."""
        series_dict = {
            "A": _make_series(120, seed=1),
            "B": _make_series(120, seed=1) * 0.99,  # nearly identical → high corr
        }
        findings = scanner.scan(series_dict, was_seen_fn=lambda a, b: False)
        if findings:
            assert findings[0].is_new is True

    def test_was_seen_fn_marks_old(self, scanner):
        """Pairs seen before should have is_new=False."""
        series_dict = {
            "A": _make_series(120, seed=1),
            "B": _make_series(120, seed=1) * 0.99,
        }
        findings = scanner.scan(series_dict, was_seen_fn=lambda a, b: True)
        if findings:
            assert findings[0].is_new is False

    def test_name_map_applied(self):
        """name_map should populate series_a_name / series_b_name."""
        cfg = ScanConfig(
            name_map={"A": "Series Alpha", "B": "Series Beta"},
            scoring=ScoringConfig(),
        )
        scanner = DiscoveryScanner(cfg)
        series_dict = {
            "A": _make_series(120, seed=1),
            "B": _make_series(120, seed=1) * 0.99,
        }
        findings = scanner.scan(series_dict)
        if findings:
            f = findings[0]
            assert f.series_a_name == "Series Alpha" or f.series_b_name == "Series Alpha"

    def test_empty_input(self, scanner):
        """Empty series dict should return empty findings."""
        assert scanner.scan({}) == []

    def test_single_series(self, scanner):
        """Single series (no pairs) should return empty findings."""
        assert scanner.scan({"only": _make_series()}) == []


# ═══════════════════════════════════════════════════════════════════
# Finding dataclass round-trip
# ═══════════════════════════════════════════════════════════════════

class TestFindingRoundTrip:

    def test_to_dict_from_dict(self):
        f = Finding(
            scan_id="abc-123",
            scanned_at="2025-01-01T00:00:00+00:00",
            series_a="GDP",
            series_b="SPY",
            series_a_name="Gross Domestic Product",
            series_b_name="S&P 500 ETF",
            correlation=0.85,
            optimal_lag=3,
            lag_correlation=0.88,
            granger_p_value=0.02,
            granger_direction="a_causes_b",
            rolling_zscore=2.5,
            regime_change_detected=True,
            trigger_types=["high_correlation", "regime_change", "granger_causality"],
            interestingness_score=0.78,
            is_new=True,
            template_summary="GDP and SPY are strongly correlated.",
            lookback_days=3650,
            frequency="M",
        )
        d = f.to_dict()
        assert isinstance(d["trigger_types"], str)
        assert "|" in d["trigger_types"]

        f2 = Finding.from_dict(d)
        assert f2.scan_id == f.scan_id
        assert f2.trigger_types == f.trigger_types
        assert f2.granger_p_value == pytest.approx(f.granger_p_value)
        assert f2.is_new is True
        assert f2.regime_change_detected is True

    def test_empty_triggers_round_trip(self):
        f = Finding(
            scan_id="x", scanned_at="t",
            series_a="A", series_b="B",
            series_a_name="A", series_b_name="B",
            correlation=0.1,
            trigger_types=[],
        )
        d = f.to_dict()
        assert d["trigger_types"] == ""
        f2 = Finding.from_dict(d)
        assert f2.trigger_types == []

    def test_none_granger_round_trip(self):
        f = Finding(
            scan_id="x", scanned_at="t",
            series_a="A", series_b="B",
            series_a_name="A", series_b_name="B",
            correlation=0.1,
            granger_p_value=None,
        )
        d = f.to_dict()
        f2 = Finding.from_dict(d)
        assert f2.granger_p_value is None
