"""Rolling window correlations and window sensitivity analysis."""

from __future__ import annotations

import pandas as pd


def compute_rolling_correlation(
    x: pd.Series,
    y: pd.Series,
    window: int = 60,
    method: str = "pearson",
) -> pd.Series:
    """Rolling pairwise correlation between two series.

    Returns NaN for periods with fewer than *window* observations.
    """
    combined = pd.DataFrame({"x": x, "y": y}).dropna()
    if method in ("pearson", "spearman"):
        return combined["x"].rolling(window, min_periods=window).corr(combined["y"])
    # kendall: pandas rolling.corr doesn't support kendall, use apply
    from scipy.stats import kendalltau

    def _kt(w):
        y_slice = combined["y"].iloc[w.index[0]:w.index[-1] + 1]
        if len(y_slice) != len(w):
            return float("nan")
        return kendalltau(w.values, y_slice.values).statistic

    return combined["x"].rolling(window, min_periods=window).apply(_kt, raw=False)


def compute_rolling_matrix(
    df: pd.DataFrame,
    window: int = 60,
    method: str = "pearson",
) -> dict[tuple[str, str], pd.Series]:
    """Rolling correlation for all unique pairs.

    Returns
    -------
    dict mapping (col_a, col_b) → pd.Series of rolling correlations.
    """
    cols = df.columns.tolist()
    result: dict[tuple[str, str], pd.Series] = {}
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            result[(cols[i], cols[j])] = compute_rolling_correlation(
                df[cols[i]], df[cols[j]], window=window, method=method,
            )
    return result


def window_sensitivity(
    x: pd.Series,
    y: pd.Series,
    windows: list[int] | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute rolling correlation at multiple window sizes for comparison.

    Returns
    -------
    pd.DataFrame
        Index = dates, columns = window sizes, values = rolling correlation.
    """
    if windows is None:
        windows = [30, 60, 90, 120, 180]

    series_dict: dict[str, pd.Series] = {}
    for w in sorted(windows):
        series_dict[str(w)] = compute_rolling_correlation(x, y, window=w, method=method)

    return pd.DataFrame(series_dict)
