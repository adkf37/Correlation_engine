"""Yahoo Finance data loader."""

from __future__ import annotations

import pandas as pd

from correlation_engine.ingest.base import BaseLoader


class YahooLoader(BaseLoader):
    """Load historical price data from Yahoo Finance.

    Uses the yfinance library. No API key required.
    """

    def load(
        self,
        tickers: list[str],
        start: str | None = None,
        end: str | None = None,
        column: str = "Close",
    ) -> pd.DataFrame:
        """Fetch historical price data for one or more tickers.

        Parameters
        ----------
        tickers : list[str]
            Yahoo Finance ticker symbols (e.g. ['^GSPC', 'AAPL', 'MSFT']).
        start : str, optional
            Start date in 'YYYY-MM-DD' format.
        end : str, optional
            End date in 'YYYY-MM-DD' format.
        column : str
            Price column to extract. One of 'Open', 'High', 'Low',
            'Close', 'Volume'. Defaults to 'Close'.
        """
        import yfinance as yf

        kwargs: dict = {"tickers": tickers, "progress": False}
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end

        raw = yf.download(**kwargs)

        if raw.empty:
            raise ValueError(
                f"No data returned for tickers: {tickers}. "
                "Check that the ticker symbols are valid."
            )

        # yf.download returns multi-level columns for multiple tickers:
        # level 0 = Price (Close, Open, …), level 1 = Ticker
        if isinstance(raw.columns, pd.MultiIndex):
            df = raw[column].copy()
        else:
            # Single ticker — flat columns
            df = raw[[column]].copy()
            df.columns = tickers[:1]

        # Warn about tickers that returned no data
        empty_tickers = [t for t in tickers if t in df.columns and df[t].isna().all()]
        if empty_tickers:
            import warnings

            warnings.warn(
                f"Tickers returned no data (delisted or invalid?): {empty_tickers}",
                stacklevel=2,
            )

        df.index = pd.to_datetime(df.index)
        return self._validate_output(df)
