"""Missing data handling and reporting."""

from __future__ import annotations

import pandas as pd


def report_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Report missing-value statistics for each column.

    Returns a DataFrame with columns: count_missing, pct_missing, total_rows.
    """
    total = len(df)
    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "count_missing": missing,
            "pct_missing": (missing / total * 100).round(2),
            "total_rows": total,
        }
    )


def handle_missing(
    df: pd.DataFrame,
    strategy: str = "interpolate",
    drop_threshold: float | None = None,
) -> pd.DataFrame:
    """Fill or remove missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (may contain NaNs).
    strategy : str
        One of 'ffill', 'bfill', 'interpolate', 'drop_rows', 'drop_cols'.
    drop_threshold : float, optional
        If set, columns with more than this percentage of NaNs are dropped
        before applying the strategy. Value in range [0, 100].

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    result = df.copy()

    # Optionally drop columns exceeding the NaN threshold
    if drop_threshold is not None:
        pct = result.isna().sum() / len(result) * 100
        to_drop = pct[pct > drop_threshold].index.tolist()
        result = result.drop(columns=to_drop)

    if strategy == "ffill":
        result = result.ffill()
    elif strategy == "bfill":
        result = result.bfill()
    elif strategy == "interpolate":
        result = result.interpolate(method="time")
        # Interpolate can't fill leading/trailing NaNs — bfill/ffill the edges
        result = result.bfill().ffill()
    elif strategy == "drop_rows":
        result = result.dropna()
    elif strategy == "drop_cols":
        result = result.dropna(axis=1)
    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            "Choose from: 'ffill', 'bfill', 'interpolate', 'drop_rows', 'drop_cols'."
        )

    return result
