"""Tests for the preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from correlation_engine.preprocessing.align import align_frequencies
from correlation_engine.preprocessing.missing import handle_missing, report_missing
from correlation_engine.preprocessing.pipeline import PreprocessingPipeline
from correlation_engine.preprocessing.transform import (
    check_stationarity,
    check_stationarity_all,
    make_stationary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_daily_df(n=500, seed=42):
    """Create a daily DataFrame with some NaNs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n)
    df = pd.DataFrame(
        {
            "a": np.cumsum(rng.standard_normal(n)),
            "b": np.cumsum(rng.standard_normal(n)),
            "c": rng.standard_normal(n),  # stationary
        },
        index=dates,
    )
    # Sprinkle ~5% NaNs
    mask = rng.random(df.shape) < 0.05
    df = df.mask(mask)
    return df


# ---------------------------------------------------------------------------
# align_frequencies tests
# ---------------------------------------------------------------------------

class TestAlignFrequencies:
    def test_daily_to_monthly(self):
        df = _make_daily_df()
        result = align_frequencies(df, target_freq="M", method="last")
        assert len(result) < len(df)
        # Monthly data should have ~n/21 rows
        assert len(result) >= 20

    def test_daily_to_weekly(self):
        df = _make_daily_df()
        result = align_frequencies(df, target_freq="W", method="mean")
        assert len(result) < len(df)

    def test_daily_to_quarterly(self):
        df = _make_daily_df()
        result = align_frequencies(df, target_freq="Q", method="last")
        assert len(result) <= 10  # ~2 years of daily data → ~8 quarters

    def test_invalid_freq_raises(self):
        df = _make_daily_df()
        with pytest.raises(ValueError, match="Unknown target_freq"):
            align_frequencies(df, target_freq="X")

    def test_invalid_method_raises(self):
        df = _make_daily_df()
        with pytest.raises(ValueError, match="Unknown method"):
            align_frequencies(df, target_freq="M", method="median")

    def test_all_agg_methods(self):
        df = _make_daily_df()
        for method in ("last", "first", "mean", "sum"):
            result = align_frequencies(df, target_freq="M", method=method)
            assert len(result) > 0


# ---------------------------------------------------------------------------
# missing data tests
# ---------------------------------------------------------------------------

class TestMissing:
    def test_report_missing(self):
        df = _make_daily_df()
        report = report_missing(df)
        assert "count_missing" in report.columns
        assert "pct_missing" in report.columns
        assert (report["pct_missing"] >= 0).all()

    def test_interpolate_fills_all(self):
        df = _make_daily_df()
        assert df.isna().any().any()  # has NaNs
        result = handle_missing(df, strategy="interpolate")
        assert not result.isna().any().any()

    def test_ffill(self):
        df = pd.DataFrame(
            {"x": [1.0, np.nan, np.nan, 4.0]},
            index=pd.date_range("2020-01-01", periods=4),
        )
        result = handle_missing(df, strategy="ffill")
        assert result["x"].tolist() == [1.0, 1.0, 1.0, 4.0]

    def test_bfill(self):
        df = pd.DataFrame(
            {"x": [1.0, np.nan, np.nan, 4.0]},
            index=pd.date_range("2020-01-01", periods=4),
        )
        result = handle_missing(df, strategy="bfill")
        assert result["x"].tolist() == [1.0, 4.0, 4.0, 4.0]

    def test_drop_rows(self):
        df = _make_daily_df()
        result = handle_missing(df, strategy="drop_rows")
        assert not result.isna().any().any()
        assert len(result) < len(df)

    def test_drop_threshold(self):
        df = pd.DataFrame(
            {
                "good": [1.0, 2.0, 3.0, 4.0],
                "bad": [np.nan, np.nan, np.nan, 1.0],  # 75% missing
            },
            index=pd.date_range("2020-01-01", periods=4),
        )
        result = handle_missing(df, strategy="ffill", drop_threshold=50)
        assert "bad" not in result.columns
        assert "good" in result.columns

    def test_invalid_strategy_raises(self):
        df = _make_daily_df()
        with pytest.raises(ValueError, match="Unknown strategy"):
            handle_missing(df, strategy="magic")


# ---------------------------------------------------------------------------
# stationarity tests
# ---------------------------------------------------------------------------

class TestStationarity:
    def test_random_walk_not_stationary(self):
        rng = np.random.default_rng(42)
        walk = pd.Series(
            np.cumsum(rng.standard_normal(500)),
            index=pd.bdate_range("2020-01-02", periods=500),
        )
        result = check_stationarity(walk)
        assert result["is_stationary"] == False  # noqa: E712 — may be np.bool_

    def test_white_noise_is_stationary(self):
        rng = np.random.default_rng(42)
        noise = pd.Series(
            rng.standard_normal(500),
            index=pd.bdate_range("2020-01-02", periods=500),
        )
        result = check_stationarity(noise)
        assert result["is_stationary"] is True

    def test_check_stationarity_all(self):
        df = _make_daily_df()
        # Fill NaNs first so ADF can run
        df = handle_missing(df, strategy="interpolate")
        report = check_stationarity_all(df)
        assert "is_stationary" in report.columns
        assert len(report) == len(df.columns)

    def test_short_series_handled(self):
        short = pd.Series([1.0, 2.0], index=pd.date_range("2020-01-01", periods=2))
        result = check_stationarity(short)
        assert result["is_stationary"] is False
        assert np.isnan(result["statistic"])


class TestMakeStationary:
    def test_diff(self):
        df = _make_daily_df()
        df = handle_missing(df, strategy="interpolate")
        result, report = make_stationary(df, method="diff")
        assert len(result) == len(df) - 1
        assert all("difference" in v for v in report.values())

    def test_log_diff(self):
        # Use positive data
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {"price": 100 + np.cumsum(rng.standard_normal(200))},
            index=pd.bdate_range("2020-01-02", periods=200),
        )
        df["price"] = df["price"].clip(lower=1)  # ensure positive
        result, report = make_stationary(df, method="log_diff")
        assert len(result) == len(df) - 1

    def test_log_diff_negative_fallback(self):
        df = pd.DataFrame(
            {"x": [-1.0, 2.0, 3.0, 4.0, 5.0] * 20},
            index=pd.bdate_range("2020-01-02", periods=100),
        )
        with pytest.warns(UserWarning, match="non-positive"):
            result, report = make_stationary(df, method="log_diff")
        assert "fallback" in report["x"]

    def test_detrend(self):
        df = _make_daily_df()
        df = handle_missing(df, strategy="interpolate")
        result, report = make_stationary(df, method="detrend")
        assert len(result) == len(df)
        assert all("detrend" in v for v in report.values())

    def test_differenced_becomes_stationary(self):
        """A random walk should become stationary after differencing."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {"walk": np.cumsum(rng.standard_normal(500))},
            index=pd.bdate_range("2020-01-02", periods=500),
        )
        result, _ = make_stationary(df, method="diff")
        adf = check_stationarity(result["walk"])
        assert adf["is_stationary"] is True

    def test_invalid_method_raises(self):
        df = _make_daily_df()
        with pytest.raises(ValueError, match="Unknown method"):
            make_stationary(df, method="magic")


# ---------------------------------------------------------------------------
# Pipeline integration tests
# ---------------------------------------------------------------------------

class TestPreprocessingPipeline:
    def test_full_pipeline(self):
        """Raw daily data with NaNs → monthly, filled, differenced."""
        df = _make_daily_df()
        pipeline = PreprocessingPipeline([
            ("align", {"target_freq": "M"}),
            ("missing", {"strategy": "interpolate"}),
            ("transform", {"method": "diff"}),
        ])
        result = pipeline.run(df)
        # Should have no NaNs
        assert not result.isna().any().any()
        # Should be monthly-ish
        assert len(result) < 30
        assert len(result) > 0

    def test_pipeline_report(self):
        df = _make_daily_df()
        pipeline = PreprocessingPipeline([
            ("align", {"target_freq": "M"}),
            ("missing", {"strategy": "interpolate"}),
            ("transform", {"method": "diff"}),
        ])
        pipeline.run(df)
        report = pipeline.report()
        assert "align" in report
        assert "missing" in report
        assert "transform" in report
        assert "rows_before" in report["align"]
        assert "rows_after" in report["align"]

    def test_pipeline_single_step(self):
        df = _make_daily_df()
        pipeline = PreprocessingPipeline([
            ("missing", {"strategy": "drop_rows"}),
        ])
        result = pipeline.run(df)
        assert not result.isna().any().any()

    def test_pipeline_empty_steps(self):
        df = _make_daily_df()
        pipeline = PreprocessingPipeline([])
        result = pipeline.run(df)
        pd.testing.assert_frame_equal(result, df)

    def test_invalid_step_raises(self):
        with pytest.raises(ValueError, match="Unknown step"):
            PreprocessingPipeline([("bogus", {})])
