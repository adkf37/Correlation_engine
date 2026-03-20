"""Frequency alignment for time-series DataFrames."""

from __future__ import annotations

import pandas as pd

# Map user-friendly strings to pandas offset aliases
_FREQ_MAP = {
    "D": "D",
    "W": "W-FRI",
    "M": "ME",
    "Q": "QE",
    "Y": "YE",
}

_AGG_MAP = {
    "last": "last",
    "first": "first",
    "mean": "mean",
    "sum": "sum",
}


def align_frequencies(
    df: pd.DataFrame,
    target_freq: str = "M",
    method: str = "last",
) -> pd.DataFrame:
    """Resample all series to a common frequency.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with DatetimeIndex.
    target_freq : str
        Target frequency: 'D', 'W', 'M', 'Q', or 'Y'.
    method : str
        Aggregation method for downsampling: 'last', 'first', 'mean', 'sum'.
        For upsampling, linear interpolation is used regardless of this setting.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame at the target frequency.
    """
    if target_freq not in _FREQ_MAP:
        raise ValueError(
            f"Unknown target_freq '{target_freq}'. Choose from: {list(_FREQ_MAP)}"
        )
    if method not in _AGG_MAP:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(_AGG_MAP)}"
        )

    offset = _FREQ_MAP[target_freq]
    resampler = df.resample(offset)

    # Downsample via aggregation
    agg_fn = _AGG_MAP[method]
    result = getattr(resampler, agg_fn)()

    # For upsampling cases, interpolate the NaN gaps that appear
    if result.isna().any().any():
        result = result.interpolate(method="time")

    return result
