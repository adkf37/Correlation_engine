"""Tests for Watchlist — data universe loading with mocked loaders."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from correlation_engine.discovery.watchlist import Watchlist


# ── helpers ────────────────────────────────────────────────────────

def _make_monthly_df(columns: list[str], n: int = 120) -> pd.DataFrame:
    """Synthetic monthly DataFrame with random-walk data."""
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(42)
    data = rng.normal(100, 10, size=(n, len(columns))).cumsum(axis=0)
    return pd.DataFrame(data, index=idx, columns=columns)


_MINI_CONFIG = {
    "fred_series": [
        {"id": "GDP", "name": "Gross Domestic Product", "category": "output", "frequency": "Q"},
        {"id": "UNRATE", "name": "Unemployment Rate", "category": "employment", "frequency": "M"},
        {"id": "CPIAUCSL", "name": "CPI", "category": "inflation", "frequency": "M"},
    ],
    "yahoo_tickers": [
        {"id": "SPY", "name": "S&P 500 ETF", "category": "us_equity", "frequency": "D"},
        {"id": "TLT", "name": "20+ Year Treasury ETF", "category": "fixed_income", "frequency": "D"},
    ],
}


# ── tests ──────────────────────────────────────────────────────────


class TestWatchlist:
    """Tests using mocked loaders — no real API calls."""

    @pytest.fixture()
    def watchlist(self, tmp_path):
        """Watchlist with mocked config and cache dir."""
        import yaml

        config_path = tmp_path / "universe.yaml"
        config_path.write_text(yaml.dump(_MINI_CONFIG), encoding="utf-8")
        return Watchlist(config_path=config_path, cache_ttl_hours=1)

    @patch("correlation_engine.discovery.watchlist.YahooLoader")
    @patch("correlation_engine.discovery.watchlist.FredLoader")
    def test_load_returns_dict_of_series(self, MockFred, MockYahoo, watchlist):
        """load() should return a dict[str, pd.Series] after preprocessing."""
        fred_df = _make_monthly_df(["GDP", "UNRATE", "CPIAUCSL"])
        yahoo_df = _make_monthly_df(["SPY", "TLT"])

        fred_inst = MockFred.return_value
        fred_inst.load.return_value = fred_df

        yahoo_inst = MockYahoo.return_value
        yahoo_inst.load.return_value = yahoo_df

        result = watchlist.load(start_date="2015-01-01")

        assert isinstance(result, dict)
        assert len(result) > 0
        for key, series in result.items():
            assert isinstance(series, pd.Series), f"{key} is not a Series"
            assert isinstance(series.index, pd.DatetimeIndex)

    @patch("correlation_engine.discovery.watchlist.YahooLoader")
    @patch("correlation_engine.discovery.watchlist.FredLoader")
    def test_all_series_no_nans(self, MockFred, MockYahoo, watchlist):
        """After preprocessing, no NaN values should remain."""
        fred_df = _make_monthly_df(["GDP", "UNRATE", "CPIAUCSL"])
        yahoo_df = _make_monthly_df(["SPY", "TLT"])

        MockFred.return_value.load.return_value = fred_df
        MockYahoo.return_value.load.return_value = yahoo_df

        result = watchlist.load()

        for key, series in result.items():
            assert not series.isna().any(), f"{key} contains NaN after preprocessing"

    @patch("correlation_engine.discovery.watchlist.YahooLoader")
    @patch("correlation_engine.discovery.watchlist.FredLoader")
    def test_partial_failure_continues(self, MockFred, MockYahoo, watchlist):
        """If FRED load fails, Yahoo data should still be returned."""
        fred_inst = MockFred.return_value
        fred_inst.load.side_effect = RuntimeError("No API key")

        yahoo_df = _make_monthly_df(["SPY", "TLT"])
        MockYahoo.return_value.load.return_value = yahoo_df

        result = watchlist.load()

        # Should still have Yahoo series
        assert len(result) > 0

    @patch("correlation_engine.discovery.watchlist.YahooLoader")
    @patch("correlation_engine.discovery.watchlist.FredLoader")
    def test_short_series_dropped(self, MockFred, MockYahoo, watchlist):
        """Series with fewer than 24 months after preprocessing are skipped."""
        # Give FRED only 12 rows of data (too short after diff → 11 obs)
        fred_df = _make_monthly_df(["GDP", "UNRATE", "CPIAUCSL"], n=12)
        yahoo_df = _make_monthly_df(["SPY", "TLT"], n=120)

        fred_inst = MockFred.return_value
        fred_inst.load.return_value = fred_df

        MockYahoo.return_value.load.return_value = yahoo_df

        result = watchlist.load()

        # FRED series should be dropped (too short), Yahoo should remain
        for key in result:
            assert len(result[key]) >= 24

    @patch("correlation_engine.discovery.watchlist.YahooLoader")
    @patch("correlation_engine.discovery.watchlist.FredLoader")
    def test_metadata_populated(self, MockFred, MockYahoo, watchlist):
        """After load, metadata property should have entries."""
        fred_df = _make_monthly_df(["GDP", "UNRATE", "CPIAUCSL"])
        yahoo_df = _make_monthly_df(["SPY", "TLT"])

        MockFred.return_value.load.return_value = fred_df
        MockYahoo.return_value.load.return_value = yahoo_df

        watchlist.load()
        meta = watchlist.metadata

        assert isinstance(meta, pd.DataFrame)
        assert len(meta) > 0
        assert "source" in meta.columns

    def test_config_loaded(self, watchlist):
        """Config should be parsed from YAML on construction."""
        cfg = watchlist.config
        assert "fred_series" in cfg
        assert "yahoo_tickers" in cfg
        assert len(cfg["fred_series"]) == 3
        assert len(cfg["yahoo_tickers"]) == 2

    @patch("correlation_engine.discovery.watchlist.YahooLoader")
    @patch("correlation_engine.discovery.watchlist.FredLoader")
    def test_empty_universe(self, MockFred, MockYahoo, tmp_path):
        """Empty config should return empty dict without crashing."""
        import yaml

        config_path = tmp_path / "empty.yaml"
        config_path.write_text(yaml.dump({"fred_series": [], "yahoo_tickers": []}))

        wl = Watchlist(config_path=config_path)
        result = wl.load()
        assert result == {}
