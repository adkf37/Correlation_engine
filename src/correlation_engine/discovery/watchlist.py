"""Watchlist — loads, caches, and preprocesses the full data universe."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from correlation_engine.ingest.cache import DataCache
from correlation_engine.ingest.fred import FredLoader
from correlation_engine.ingest.yahoo import YahooLoader
from correlation_engine.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # src/correlation_engine/discovery -> project root
_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "universe.yaml"


class Watchlist:
    """Fetch and preprocess all series defined in the universe config.

    Parameters
    ----------
    config_path : str or Path
        Path to ``universe.yaml``.
    cache_ttl_hours : int
        Hours before cached API data expires.
    """

    def __init__(
        self,
        config_path: str | Path = _DEFAULT_CONFIG,
        cache_ttl_hours: int = 24,
    ):
        self._config_path = Path(config_path)
        self._cache = DataCache(max_age=cache_ttl_hours * 3600)
        self._config = self._load_config()
        self._metadata: list[dict] = []

    # ── public API ────────────────────────────────────────────────────

    def load(
        self,
        start_date: str = "2015-01-01",
        end_date: str | None = None,
    ) -> dict[str, pd.Series]:
        """Fetch all configured series, preprocess, and return as a dict.

        Each value is a ``pd.Series`` with monthly ``DatetimeIndex``,
        forward-filled, and stationarity-transformed.
        """
        raw_frames: dict[str, pd.DataFrame] = {}
        self._metadata = []

        # ── FRED series ──
        fred_entries = self._config.get("fred_series", [])
        if fred_entries:
            raw_frames["fred"] = self._fetch_fred(
                fred_entries, start_date, end_date,
            )

        # ── Yahoo tickers ──
        yahoo_entries = self._config.get("yahoo_tickers", [])
        if yahoo_entries:
            raw_frames["yahoo"] = self._fetch_yahoo(
                yahoo_entries, start_date, end_date,
            )

        # Merge all into one wide DataFrame
        all_frames = [df for df in raw_frames.values() if df is not None and not df.empty]
        if not all_frames:
            logger.warning("No series loaded from any source.")
            return {}

        merged = pd.concat(all_frames, axis=1)

        # ── Preprocess ──
        pipeline = PreprocessingPipeline([
            ("align", {"target_freq": "M", "method": "last"}),
            ("missing", {"strategy": "ffill"}),
            ("transform", {"method": "diff"}),
        ])
        clean = pipeline.run(merged)

        # Drop any columns that are entirely NaN after preprocessing
        clean = clean.dropna(axis=1, how="all")

        # Build final output dict
        result: dict[str, pd.Series] = {}
        for col in clean.columns:
            s = clean[col].dropna()
            if len(s) >= 24:  # need at least 2 years of monthly data
                result[col] = s
            else:
                logger.warning(
                    "Skipping '%s': only %d observations after preprocessing.",
                    col, len(s),
                )

        loaded = len(result)
        total = len(fred_entries) + len(yahoo_entries)
        logger.info(
            "Watchlist loaded: %d / %d series ready for scanning.", loaded, total,
        )
        return result

    @property
    def metadata(self) -> pd.DataFrame:
        """DataFrame with series id, name, category, source info."""
        if not self._metadata:
            return pd.DataFrame(
                columns=["id", "name", "category", "source", "frequency"],
            )
        return pd.DataFrame(self._metadata)

    @property
    def config(self) -> dict:
        """Raw parsed config dict."""
        return self._config

    # ── private helpers ───────────────────────────────────────────────

    def _load_config(self) -> dict:
        with open(self._config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _fetch_fred(
        self,
        entries: list[dict],
        start: str,
        end: str | None,
    ) -> pd.DataFrame:
        ids = [e["id"] for e in entries]
        name_map = {e["id"]: e.get("name", e["id"]) for e in entries}

        # Check cache first
        cache_key = self._cache.make_key("fred_watchlist", series=ids, start=start, end=end)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("FRED data loaded from cache (%d columns).", len(cached.columns))
            for e in entries:
                self._metadata.append({**e, "source": "fred"})
            return cached

        loader = FredLoader()
        frames: dict[str, pd.Series] = {}
        for entry in entries:
            sid = entry["id"]
            try:
                df = loader.load(
                    series_ids=[sid],
                    start=start,
                    end=end,
                    request_delay=0.5,
                )
                frames[sid] = df[sid]
                self._metadata.append({**entry, "source": "fred"})
            except Exception as exc:
                logger.warning("Failed to load FRED series '%s': %s", sid, exc)

        if not frames:
            return pd.DataFrame()

        result = pd.DataFrame(frames)
        self._cache.put(cache_key, result)
        return result

    def _fetch_yahoo(
        self,
        entries: list[dict],
        start: str,
        end: str | None,
    ) -> pd.DataFrame:
        tickers = [e["id"] for e in entries]

        cache_key = self._cache.make_key("yahoo_watchlist", tickers=tickers, start=start, end=end)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("Yahoo data loaded from cache (%d columns).", len(cached.columns))
            for e in entries:
                self._metadata.append({**e, "source": "yahoo"})
            return cached

        loader = YahooLoader()
        try:
            df = loader.load(tickers=tickers, start=start, end=end)
            for e in entries:
                if e["id"] in df.columns:
                    self._metadata.append({**e, "source": "yahoo"})
                else:
                    logger.warning("Yahoo ticker '%s' returned no data.", e["id"])
            self._cache.put(cache_key, df)
            return df
        except Exception as exc:
            logger.warning("Failed to load Yahoo tickers: %s", exc)
            return pd.DataFrame()
