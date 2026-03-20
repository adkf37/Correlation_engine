"""CSV / TSV file loader."""

from pathlib import Path

import pandas as pd

from correlation_engine.ingest.base import BaseLoader


class CsvLoader(BaseLoader):
    """Load time-series data from local CSV/TSV files.

    Supports single or multiple files. Auto-detects the date column
    (first column that parses as datetime) and delimiter.
    """

    def load(
        self,
        path: str | Path | list[str | Path],
        date_column: str | None = None,
        delimiter: str | None = None,
    ) -> pd.DataFrame:
        """Load one or more CSV/TSV files and return a merged DataFrame.

        Parameters
        ----------
        path : str, Path, or list thereof
            File path(s) to load.
        date_column : str, optional
            Name of the date column. If None, auto-detects the first
            column that parses as datetime.
        delimiter : str, optional
            Column delimiter. If None, pandas will auto-detect.
        """
        paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]

        frames: list[pd.DataFrame] = []
        for p in paths:
            df = self._load_single(p, date_column=date_column, delimiter=delimiter)
            frames.append(df)

        if len(frames) == 1:
            result = frames[0]
        else:
            result = pd.concat(frames, axis=1)
            # Drop duplicate columns (same series loaded from multiple files)
            result = result.loc[:, ~result.columns.duplicated()]

        return self._validate_output(result)

    def _load_single(
        self,
        path: Path,
        date_column: str | None = None,
        delimiter: str | None = None,
    ) -> pd.DataFrame:
        """Load a single CSV/TSV file."""
        read_kwargs: dict = {}
        if delimiter is not None:
            read_kwargs["sep"] = delimiter

        df = pd.read_csv(path, **read_kwargs)

        # Resolve the date column
        date_col = date_column or self._detect_date_column(df)
        if date_col is None:
            raise ValueError(
                f"Could not detect a date column in {path.name}. "
                "Pass date_column= explicitly."
            )

        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
        return df

    @staticmethod
    def _detect_date_column(df: pd.DataFrame) -> str | None:
        """Return the name of the first column that parses as datetime."""
        for col in df.columns:
            if df[col].dtype == "object" or "date" in col.lower() or "time" in col.lower():
                try:
                    pd.to_datetime(df[col])
                    return col
                except (ValueError, TypeError):
                    continue
        return None
