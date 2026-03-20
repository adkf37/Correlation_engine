"""FRED API data loader."""

from __future__ import annotations

import os
import time

import pandas as pd
from dotenv import find_dotenv, load_dotenv

from correlation_engine.ingest.base import BaseLoader

load_dotenv(find_dotenv(usecwd=True))


class FredLoader(BaseLoader):
    """Load time-series data from the FRED API.

    Requires a FRED API key set as the environment variable FRED_API_KEY
    (or in a .env file).
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.environ.get("FRED_API_KEY")

    def load(
        self,
        series_ids: list[str],
        start: str | None = None,
        end: str | None = None,
        request_delay: float = 0.2,
    ) -> pd.DataFrame:
        """Fetch one or more FRED series and return a merged DataFrame.

        Parameters
        ----------
        series_ids : list[str]
            FRED series identifiers (e.g. ['GDP', 'CPIAUCSL', 'UNRATE']).
        start : str, optional
            Start date in 'YYYY-MM-DD' format.
        end : str, optional
            End date in 'YYYY-MM-DD' format.
        request_delay : float
            Seconds to wait between API calls to respect rate limits.
        """
        if not self._api_key:
            raise RuntimeError(
                "FRED API key not found. Set FRED_API_KEY in your environment "
                "or .env file, or pass api_key= to FredLoader()."
            )

        from fredapi import Fred

        fred = Fred(api_key=self._api_key)
        frames: dict[str, pd.Series] = {}

        for i, sid in enumerate(series_ids):
            kwargs: dict = {}
            if start:
                kwargs["observation_start"] = start
            if end:
                kwargs["observation_end"] = end

            series = fred.get_series(sid, **kwargs)
            series.name = sid
            frames[sid] = series

            # Rate-limit courtesy delay (skip after last request)
            if i < len(series_ids) - 1:
                time.sleep(request_delay)

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        return self._validate_output(df)
