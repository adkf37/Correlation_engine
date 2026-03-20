"""Tests for the data ingestion layer."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from correlation_engine.ingest.base import BaseLoader
from correlation_engine.ingest.cache import DataCache
from correlation_engine.ingest.csv_loader import CsvLoader
from correlation_engine.ingest.fred import FredLoader
from correlation_engine.ingest.yahoo import YahooLoader

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


# ---------------------------------------------------------------------------
# BaseLoader contract tests
# ---------------------------------------------------------------------------

class ConcreteLoader(BaseLoader):
    """Minimal concrete loader for testing validation."""

    def load(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._validate_output(df)


class TestBaseLoaderValidation:
    def test_valid_dataframe(self):
        loader = ConcreteLoader()
        df = pd.DataFrame(
            {"a": [1.0, 2.0], "b": [3.0, 4.0]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        )
        result = loader.load(df)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "date"

    def test_rejects_non_datetime_index(self):
        loader = ConcreteLoader()
        df = pd.DataFrame({"a": [1.0, 2.0]}, index=[0, 1])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            loader.load(df)

    def test_rejects_duplicate_dates(self):
        loader = ConcreteLoader()
        df = pd.DataFrame(
            {"a": [1.0, 2.0]},
            index=pd.to_datetime(["2020-01-01", "2020-01-01"]),
        )
        with pytest.raises(ValueError, match="duplicate"):
            loader.load(df)

    def test_coerces_to_numeric(self):
        loader = ConcreteLoader()
        df = pd.DataFrame(
            {"a": ["1.5", "2.5"]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02"]),
        )
        result = loader.load(df)
        assert result["a"].dtype == np.float64


# ---------------------------------------------------------------------------
# CsvLoader tests
# ---------------------------------------------------------------------------

class TestCsvLoader:
    def test_load_macro_sample(self):
        loader = CsvLoader()
        df = loader.load(SAMPLE_DIR / "macro_indicators.csv")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert set(df.columns) >= {"gdp_growth", "cpi", "unemployment", "interest_rate"}
        assert len(df) > 100

    def test_load_equity_sample(self):
        loader = CsvLoader()
        df = loader.load(SAMPLE_DIR / "equity_prices.csv")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "stock_a" in df.columns
        assert len(df) > 2000

    def test_load_multiple_files(self):
        loader = CsvLoader()
        df = loader.load([
            SAMPLE_DIR / "macro_indicators.csv",
            SAMPLE_DIR / "equity_prices.csv",
        ])
        # Should contain columns from both files
        assert "gdp_growth" in df.columns
        assert "stock_a" in df.columns

    def test_explicit_date_column(self):
        loader = CsvLoader()
        df = loader.load(SAMPLE_DIR / "macro_indicators.csv", date_column="date")
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_auto_detect_date_column(self):
        # Create a temp CSV with a non-obvious date column name
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        try:
            tmp.write("timestamp,value\n2020-01-01,1.0\n2020-01-02,2.0\n")
            tmp.close()
            loader = CsvLoader()
            df = loader.load(tmp.name)
            assert isinstance(df.index, pd.DatetimeIndex)
        finally:
            os.unlink(tmp.name)

    def test_tsv_delimiter(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False)
        try:
            tmp.write("date\tvalue\n2020-01-01\t1.0\n2020-01-02\t2.0\n")
            tmp.close()
            loader = CsvLoader()
            df = loader.load(tmp.name, delimiter="\t")
            assert len(df) == 2
        finally:
            os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# FredLoader tests (mocked)
# ---------------------------------------------------------------------------

class TestFredLoader:
    def _mock_fred_series(self, series_id: str, **kwargs):
        """Return a fake FRED series."""
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        rng = np.random.default_rng(hash(series_id) % 2**32)
        return pd.Series(rng.standard_normal(24), index=dates, name=series_id)

    @patch("fredapi.Fred")
    def test_load_single_series(self, MockFred):
        mock_instance = MagicMock()
        mock_instance.get_series.side_effect = self._mock_fred_series
        MockFred.return_value = mock_instance

        loader = FredLoader(api_key="test_key")
        df = loader.load(series_ids=["GDP"], start="2020-01-01", end="2021-12-01")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "GDP" in df.columns

    @patch("fredapi.Fred")
    def test_load_multiple_series(self, MockFred):
        mock_instance = MagicMock()
        mock_instance.get_series.side_effect = self._mock_fred_series
        MockFred.return_value = mock_instance

        loader = FredLoader(api_key="test_key")
        df = loader.load(
            series_ids=["GDP", "CPIAUCSL", "UNRATE"],
            request_delay=0,  # no delay in tests
        )
        assert len(df.columns) == 3
        assert set(df.columns) == {"GDP", "CPIAUCSL", "UNRATE"}

    def test_missing_api_key_raises(self):
        loader = FredLoader(api_key=None)
        loader._api_key = None
        with pytest.raises(RuntimeError, match="FRED API key not found"):
            loader.load(series_ids=["GDP"])


# ---------------------------------------------------------------------------
# YahooLoader tests (mocked)
# ---------------------------------------------------------------------------

class TestYahooLoader:
    def _make_mock_data(self, tickers):
        """Create fake multi-ticker download output."""
        dates = pd.bdate_range("2023-01-02", periods=100)
        rng = np.random.default_rng(99)
        if len(tickers) > 1:
            arrays = {
                ("Close", t): 100 + np.cumsum(rng.standard_normal(100))
                for t in tickers
            }
            df = pd.DataFrame(arrays, index=dates)
            df.columns = pd.MultiIndex.from_tuples(df.columns)
        else:
            df = pd.DataFrame(
                {"Close": 100 + np.cumsum(rng.standard_normal(100))},
                index=dates,
            )
        return df

    @patch("yfinance.download")
    def test_load_single_ticker(self, mock_download):
        mock_download.return_value = self._make_mock_data(["AAPL"])
        loader = YahooLoader()
        df = loader.load(tickers=["AAPL"])
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "AAPL" in df.columns

    @patch("yfinance.download")
    def test_load_multiple_tickers(self, mock_download):
        tickers = ["AAPL", "MSFT", "GOOG"]
        mock_download.return_value = self._make_mock_data(tickers)
        loader = YahooLoader()
        df = loader.load(tickers=tickers)
        assert len(df.columns) == 3

    @patch("yfinance.download")
    def test_empty_download_raises(self, mock_download):
        mock_download.return_value = pd.DataFrame()
        loader = YahooLoader()
        with pytest.raises(ValueError, match="No data returned"):
            loader.load(tickers=["INVALID_TICKER"])


# ---------------------------------------------------------------------------
# DataCache tests
# ---------------------------------------------------------------------------

class TestDataCache:
    def _make_df(self):
        return pd.DataFrame(
            {"a": [1.0, 2.0, 3.0]},
            index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        )

    def test_put_and_get(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path, max_age=3600)
        df = self._make_df()
        key = cache.make_key("csv", path="test.csv")
        cache.put(key, df)
        result = cache.get(key)
        assert result is not None
        pd.testing.assert_frame_equal(df, result)

    def test_get_missing_returns_none(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path)
        assert cache.get("nonexistent") is None

    def test_ttl_expiry(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path, max_age=1)
        df = self._make_df()
        key = cache.make_key("csv", path="test.csv")
        cache.put(key, df)

        # Immediately: should be fresh
        assert cache.get(key) is not None

        # After TTL: should be stale
        time.sleep(1.1)
        assert cache.get(key) is None

    def test_invalidate(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path)
        df = self._make_df()
        key = cache.make_key("csv", path="test.csv")
        cache.put(key, df)
        assert cache.invalidate(key) is True
        assert cache.get(key) is None
        assert cache.invalidate(key) is False  # already gone

    def test_clear(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path)
        df = self._make_df()
        cache.put("k1", df)
        cache.put("k2", df)
        removed = cache.clear()
        assert removed == 2
        assert cache.get("k1") is None

    def test_make_key_deterministic(self, tmp_path):
        cache = DataCache(cache_dir=tmp_path)
        k1 = cache.make_key("fred", series=["GDP"], start="2020-01-01")
        k2 = cache.make_key("fred", series=["GDP"], start="2020-01-01")
        k3 = cache.make_key("fred", series=["CPI"], start="2020-01-01")
        assert k1 == k2
        assert k1 != k3
