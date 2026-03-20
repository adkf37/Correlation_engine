"""Tests for DCC-GARCH model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from correlation_engine.analysis.dcc_garch import DccResult, fit_dcc_garch


@pytest.fixture()
def returns_df():
    """Three volatile return series with ~300 obs."""
    rng = np.random.default_rng(42)
    n = 350
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # Simulate heteroskedastic returns
    vol = 0.01 + 0.005 * np.abs(np.sin(np.arange(n) / 30))
    r1 = rng.normal(0, 1, n) * vol
    r2 = 0.6 * r1 + rng.normal(0, 1, n) * vol * 0.8
    r3 = rng.normal(0, 1, n) * vol * 1.2
    return pd.DataFrame({"a": r1, "b": r2, "c": r3}, index=idx)


class TestDccGarch:
    def test_smoke_fit(self, returns_df):
        result = fit_dcc_garch(returns_df)
        assert isinstance(result, DccResult)

    def test_result_structure(self, returns_df):
        result = fit_dcc_garch(returns_df)
        # conditional_correlations: 3 choose 2 = 3 pairs
        assert len(result.conditional_correlations) == 3
        # conditional_volatilities: 3 columns
        assert result.conditional_volatilities.shape[1] == 3
        # model_params has a, b
        assert "a" in result.model_params
        assert "b" in result.model_params

    def test_correlations_bounded(self, returns_df):
        result = fit_dcc_garch(returns_df)
        for pair, series in result.conditional_correlations.items():
            assert series.min() >= -1.0 - 1e-6
            assert series.max() <= 1.0 + 1e-6

    def test_dcc_params_valid(self, returns_df):
        result = fit_dcc_garch(returns_df)
        a = result.model_params["a"]
        b = result.model_params["b"]
        assert a >= 0
        assert b >= 0
        assert a + b < 1

    def test_warns_many_series(self):
        rng = np.random.default_rng(1)
        n = 300
        idx = pd.date_range("2020", periods=n, freq="B")
        df = pd.DataFrame(
            rng.standard_normal((n, 12)) * 0.01,
            index=idx,
            columns=[f"s{i}" for i in range(12)],
        )
        with pytest.warns(UserWarning, match="computationally expensive"):
            fit_dcc_garch(df)

    def test_warns_few_observations(self):
        rng = np.random.default_rng(2)
        n = 100
        idx = pd.date_range("2020", periods=n, freq="B")
        df = pd.DataFrame({
            "a": rng.standard_normal(n) * 0.01,
            "b": rng.standard_normal(n) * 0.01,
        }, index=idx)
        with pytest.warns(UserWarning, match="250"):
            fit_dcc_garch(df)
