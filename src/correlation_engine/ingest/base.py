"""Abstract base class for all data loaders."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseLoader(ABC):
    """Base class for time-series data loaders.

    All loaders must return a DataFrame with:
    - DatetimeIndex (named 'date')
    - One column per series, dtype float64
    - No duplicate index values
    """

    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        """Load time-series data and return a cleaned DataFrame."""

    def _validate_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate that the output conforms to the BaseLoader contract."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                f"DataFrame index must be DatetimeIndex, got {type(df.index).__name__}"
            )

        if df.index.duplicated().any():
            raise ValueError("DataFrame index contains duplicate dates")

        # Coerce all columns to float64
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.index.name = "date"
        return df
