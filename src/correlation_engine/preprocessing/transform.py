"""Stationarity testing and transforms."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.signal import detrend as scipy_detrend
from statsmodels.tsa.stattools import adfuller


def check_stationarity(
    series: pd.Series,
    significance: float = 0.05,
) -> dict:
    """Run the Augmented Dickey-Fuller test on a single series.

    Returns
    -------
    dict with keys: statistic, pvalue, is_stationary, critical_values, n_obs.
    """
    clean = series.dropna()
    if len(clean) < 20:
        return {
            "statistic": np.nan,
            "pvalue": np.nan,
            "is_stationary": False,
            "critical_values": {},
            "n_obs": len(clean),
        }

    result = adfuller(clean, autolag="AIC")
    return {
        "statistic": result[0],
        "pvalue": result[1],
        "is_stationary": result[1] < significance,
        "critical_values": result[4],
        "n_obs": result[3],
    }


def check_stationarity_all(
    df: pd.DataFrame,
    significance: float = 0.05,
) -> pd.DataFrame:
    """Run ADF test on every column and return a summary table."""
    rows = []
    for col in df.columns:
        result = check_stationarity(df[col], significance=significance)
        rows.append({"series": col, **result})
    return pd.DataFrame(rows).set_index("series")


def make_stationary(
    df: pd.DataFrame,
    method: str = "diff",
) -> tuple[pd.DataFrame, dict]:
    """Transform series toward stationarity.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with DatetimeIndex.
    method : str
        'diff' (first difference), 'log_diff' (log returns), or 'detrend'
        (linear detrending).

    Returns
    -------
    tuple of (transformed DataFrame, report dict).
    The report dict maps column names to what was applied.
    """
    report: dict[str, str] = {}

    if method == "diff":
        result = df.diff()
        # Drop the leading NaN row created by differencing
        result = result.iloc[1:]
        for col in df.columns:
            report[col] = "first_difference"

    elif method == "log_diff":
        # Guard against non-positive values
        has_bad = (df <= 0).any().any()
        if has_bad:
            bad_cols = [c for c in df.columns if (df[c] <= 0).any()]
            warnings.warn(
                f"log_diff requires positive values. Columns with non-positive "
                f"values will use first difference instead: {bad_cols}",
                stacklevel=2,
            )
        result = pd.DataFrame(index=df.index)
        for col in df.columns:
            if (df[col] <= 0).any():
                result[col] = df[col].diff()
                report[col] = "first_difference (fallback from log_diff)"
            else:
                result[col] = np.log(df[col]).diff()
                report[col] = "log_difference"
        result = result.iloc[1:]

    elif method == "detrend":
        result = df.copy()
        for col in df.columns:
            clean = df[col].dropna()
            if len(clean) < 3:
                report[col] = "skipped (too few observations)"
                continue
            detrended = scipy_detrend(clean.values, type="linear")
            result.loc[clean.index, col] = detrended
            report[col] = "linear_detrend"

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'diff', 'log_diff', 'detrend'."
        )

    return result, report
