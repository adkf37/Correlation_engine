"""Tests for core correlation analysis (Phase 4)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib.figure import Figure as MplFigure

from correlation_engine.analysis.correlation import (
    cluster_correlation_matrix,
    compute_correlation_matrix,
)
from correlation_engine.analysis.lag import (
    compute_cross_correlation,
    compute_lead_lag_matrix,
)
from correlation_engine.viz.heatmap import plot_correlation_heatmap
from correlation_engine.viz.lag_plots import plot_ccf, plot_lead_lag_matrix


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_df(n=300, seed=42):
    """Create a clean DataFrame with known correlation structure."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    base = rng.standard_normal(n)
    df = pd.DataFrame(
        {
            "a": base,
            "b": 0.9 * base + 0.1 * rng.standard_normal(n),  # highly correlated with a
            "c": -0.7 * base + 0.3 * rng.standard_normal(n),  # negatively correlated
            "d": rng.standard_normal(n),  # uncorrelated
        },
        index=dates,
    )
    return df


def _make_lagged_df(n=400, lag=3, seed=42):
    """Create two series where X leads Y by `lag` periods.

    y[t] = x[t - lag] + noise, so a positive CCF peak should appear at +lag.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    x = rng.standard_normal(n)
    y = np.zeros(n)
    y[lag:] = x[:-lag]
    y[:lag] = rng.standard_normal(lag)
    y += 0.2 * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": y}, index=dates)


# ---------------------------------------------------------------------------
# Correlation matrix tests
# ---------------------------------------------------------------------------

class TestCorrelationMatrix:
    def test_pearson_basic(self):
        df = _make_clean_df()
        corr = compute_correlation_matrix(df, method="pearson")
        assert corr.shape == (4, 4)
        # Diagonal should be 1.0
        np.testing.assert_allclose(np.diag(corr.values), 1.0)

    def test_high_positive_correlation(self):
        df = _make_clean_df()
        corr = compute_correlation_matrix(df)
        assert corr.loc["a", "b"] > 0.85

    def test_negative_correlation(self):
        df = _make_clean_df()
        corr = compute_correlation_matrix(df)
        assert corr.loc["a", "c"] < -0.5

    def test_near_zero_correlation(self):
        df = _make_clean_df()
        corr = compute_correlation_matrix(df)
        assert abs(corr.loc["a", "d"]) < 0.2

    def test_spearman(self):
        df = _make_clean_df()
        corr = compute_correlation_matrix(df, method="spearman")
        assert corr.shape == (4, 4)

    def test_kendall(self):
        df = _make_clean_df()
        corr = compute_correlation_matrix(df, method="kendall")
        assert corr.shape == (4, 4)

    def test_invalid_method_raises(self):
        df = _make_clean_df()
        with pytest.raises(ValueError, match="Unknown method"):
            compute_correlation_matrix(df, method="magic")

    def test_symmetric(self):
        df = _make_clean_df()
        corr = compute_correlation_matrix(df)
        pd.testing.assert_frame_equal(corr, corr.T)


class TestClusterCorrelationMatrix:
    def test_reordering(self):
        df = _make_clean_df()
        corr = compute_correlation_matrix(df)
        reordered, linkage = cluster_correlation_matrix(corr)
        # Same shape and values, different order
        assert reordered.shape == corr.shape
        assert set(reordered.columns) == set(corr.columns)
        assert linkage is not None

    def test_correlated_series_adjacent(self):
        """a and b are highly correlated — clustering should place them adjacent."""
        df = _make_clean_df()
        corr = compute_correlation_matrix(df)
        reordered, _ = cluster_correlation_matrix(corr)
        cols = list(reordered.columns)
        idx_a = cols.index("a")
        idx_b = cols.index("b")
        assert abs(idx_a - idx_b) == 1


# ---------------------------------------------------------------------------
# Cross-correlation / lag tests
# ---------------------------------------------------------------------------

class TestCrossCorrelation:
    def test_peak_at_correct_lag(self):
        df = _make_lagged_df(lag=3)
        ccf_df = compute_cross_correlation(df["x"], df["y"], max_lag=10)
        # Peak absolute correlation should be at lag=3
        peak_row = ccf_df.loc[ccf_df["correlation"].abs().idxmax()]
        assert int(peak_row["lag"]) == 3

    def test_output_shape(self):
        df = _make_lagged_df()
        ccf_df = compute_cross_correlation(df["x"], df["y"], max_lag=12)
        # Should have 2*max_lag + 1 rows
        assert len(ccf_df) == 25
        assert list(ccf_df.columns) == ["lag", "correlation"]

    def test_self_correlation_peaks_at_zero(self):
        df = _make_lagged_df()
        ccf_df = compute_cross_correlation(df["x"], df["x"], max_lag=10)
        peak_row = ccf_df.loc[ccf_df["correlation"].abs().idxmax()]
        assert int(peak_row["lag"]) == 0
        assert peak_row["correlation"] > 0.99

    def test_different_lag_values(self):
        """Test with lag=5 and lag=7 to ensure generality."""
        for lag in [5, 7]:
            df = _make_lagged_df(lag=lag)
            ccf_df = compute_cross_correlation(df["x"], df["y"], max_lag=12)
            peak_row = ccf_df.loc[ccf_df["correlation"].abs().idxmax()]
            assert int(peak_row["lag"]) == lag


class TestLeadLagMatrix:
    def test_output_structure(self):
        df = _make_clean_df()
        result = compute_lead_lag_matrix(df, max_lag=6)
        assert "optimal_lag" in result.columns.get_level_values(0)
        assert "max_correlation" in result.columns.get_level_values(0)

    def test_diagonal_is_zero_lag(self):
        df = _make_clean_df()
        result = compute_lead_lag_matrix(df, max_lag=6)
        lag_matrix = result["optimal_lag"]
        for col in df.columns:
            assert lag_matrix.loc[col, col] == 0

    def test_detects_known_lag(self):
        df = _make_lagged_df(lag=3)
        result = compute_lead_lag_matrix(df, max_lag=6)
        lag_val = result["optimal_lag"].loc["x", "y"]
        assert lag_val == 3


# ---------------------------------------------------------------------------
# Visualization smoke tests
# ---------------------------------------------------------------------------

class TestHeatmapViz:
    def test_plotly_heatmap(self):
        df = _make_clean_df()
        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr, interactive=True)
        assert isinstance(fig, go.Figure)

    def test_seaborn_heatmap(self):
        df = _make_clean_df()
        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr, interactive=False)
        assert isinstance(fig, MplFigure)

    def test_unclustered_heatmap(self):
        df = _make_clean_df()
        corr = compute_correlation_matrix(df)
        fig = plot_correlation_heatmap(corr, interactive=True, clustered=False)
        assert isinstance(fig, go.Figure)


class TestLagViz:
    def test_plot_ccf(self):
        df = _make_lagged_df()
        ccf_df = compute_cross_correlation(df["x"], df["y"], max_lag=10)
        fig = plot_ccf(ccf_df, title="Test CCF")
        assert isinstance(fig, go.Figure)

    def test_plot_lead_lag_matrix(self):
        df = _make_clean_df()
        ll_df = compute_lead_lag_matrix(df, max_lag=6)
        fig = plot_lead_lag_matrix(ll_df)
        assert isinstance(fig, go.Figure)
