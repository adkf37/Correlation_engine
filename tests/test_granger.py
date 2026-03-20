"""Tests for Granger causality analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from correlation_engine.analysis.granger import (
    granger_causality_matrix,
    granger_causality_test,
    granger_summary,
)


@pytest.fixture()
def causal_df():
    """Y = 0.5 * X_lagged_3 + noise → X Granger-causes Y at lag 3."""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2000-01-01", periods=n, freq="D")
    x = rng.standard_normal(n)
    noise = rng.standard_normal(n) * 0.5
    y = np.zeros(n)
    for t in range(3, n):
        y[t] = 0.5 * x[t - 3] + noise[t]
    return pd.DataFrame({"x": x, "y": y}, index=idx)


@pytest.fixture()
def four_series_df():
    rng = np.random.default_rng(7)
    n = 300
    idx = pd.date_range("2010-01-01", periods=n, freq="D")
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.4 * x[t - 2] + rng.normal(0, 0.5)
    return pd.DataFrame({
        "x": x,
        "y": y,
        "c": rng.standard_normal(n),
        "d": rng.standard_normal(n),
    }, index=idx)


class TestGrangerPairwise:
    def test_causal_pair_rejects_null(self, causal_df):
        res = granger_causality_test(causal_df, target="y", predictor="x", max_lag=6)
        assert res["reject_null"] is True
        assert res["p_value"] < 0.05

    def test_reverse_direction_weaker(self, causal_df):
        """Y→X should be weaker than X→Y."""
        fwd = granger_causality_test(causal_df, target="y", predictor="x", max_lag=6)
        rev = granger_causality_test(causal_df, target="x", predictor="y", max_lag=6)
        assert fwd["p_value"] < rev["p_value"]

    def test_non_stationary_warns(self):
        rng = np.random.default_rng(5)
        n = 300
        idx = pd.date_range("2000", periods=n, freq="D")
        # Random walk (non-stationary)
        x = rng.standard_normal(n).cumsum()
        y = rng.standard_normal(n).cumsum()
        df = pd.DataFrame({"x": x, "y": y}, index=idx)
        with pytest.warns(UserWarning, match="non-stationary"):
            granger_causality_test(df, target="y", predictor="x", max_lag=4)

    def test_returns_expected_keys(self, causal_df):
        res = granger_causality_test(causal_df, target="y", predictor="x", max_lag=4)
        for key in ("optimal_lag", "f_stat", "p_value", "reject_null", "differenced"):
            assert key in res


class TestGrangerMatrix:
    def test_shape(self, four_series_df):
        mat = granger_causality_matrix(four_series_df, max_lag=4)
        assert mat.shape == (4, 4)

    def test_diagonal_is_nan(self, four_series_df):
        mat = granger_causality_matrix(four_series_df, max_lag=4)
        assert mat.loc["x", "x"] != mat.loc["x", "x"]  # NaN != NaN

    def test_asymmetric(self, four_series_df):
        """Granger matrix is NOT symmetric — direction matters."""
        mat = granger_causality_matrix(four_series_df, max_lag=4)
        # x→y and y→x should differ
        assert mat.loc["y", "x"] != mat.loc["x", "y"]


class TestGrangerSummary:
    def test_returns_significant_pairs(self, four_series_df):
        mat = granger_causality_matrix(four_series_df, max_lag=4)
        summary = granger_summary(mat, alpha=0.05)
        # x should Granger-cause y
        predictors = [s["predictor"] for s in summary]
        targets = [s["target"] for s in summary]
        assert "x" in predictors and "y" in targets

    def test_sorted_by_pvalue(self, four_series_df):
        mat = granger_causality_matrix(four_series_df, max_lag=4)
        summary = granger_summary(mat, alpha=0.5)  # lenient to get multiple hits
        pvals = [s["p_value"] for s in summary]
        assert pvals == sorted(pvals)
