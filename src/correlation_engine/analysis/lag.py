"""Cross-correlation and lead/lag analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import ccf


def compute_cross_correlation(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 24,
) -> pd.DataFrame:
    """Compute cross-correlation between two series at multiple lags.

    Positive lag means X leads Y (Y is shifted forward relative to X).
    Negative lag means Y leads X.

    Parameters
    ----------
    x, y : pd.Series
        Two time series of equal length (aligned beforehand).
    max_lag : int
        Maximum lag in both directions.

    Returns
    -------
    pd.DataFrame
        Columns: 'lag', 'correlation'. Rows go from -max_lag to +max_lag.
    """
    x_clean = x.dropna()
    y_clean = y.dropna()

    # Align on shared index
    shared_idx = x_clean.index.intersection(y_clean.index)
    x_vals = x_clean.loc[shared_idx].values.astype(float)
    y_vals = y_clean.loc[shared_idx].values.astype(float)

    n = len(shared_idx)
    if n < max_lag + 1:
        max_lag = max(n - 1, 0)

    # ccf returns values starting at lag 0.
    # ccf(y, x, nlags=K) → array of length K for lags 0, 1, ..., K-1.
    # ccf(y, x) at lag k = corr(y[t], x[t-k]); peak at k means X leads Y by k.
    request_lags = max_lag + 1  # to include lag 0 through max_lag
    if max_lag > 0:
        pos_ccf = ccf(y_vals, x_vals, nlags=request_lags, alpha=None)
        neg_ccf = ccf(x_vals, y_vals, nlags=request_lags, alpha=None)
    else:
        corr0 = np.corrcoef(x_vals, y_vals)[0, 1]
        return pd.DataFrame({"lag": [0], "correlation": [corr0]})

    # pos_ccf[0] = lag 0, pos_ccf[k] = lag k
    # Assemble: [-max_lag, ..., -1, 0, 1, ..., max_lag]
    # Negative lags from neg_ccf (reversed, skip index 0 to avoid double lag-0)
    # Lag 0 from pos_ccf[0] (same as neg_ccf[0])
    # Positive lags from pos_ccf[1:]
    lags = list(range(-max_lag, 0)) + [0] + list(range(1, max_lag + 1))
    corrs = (
        list(neg_ccf[max_lag:0:-1])  # -max_lag, ..., -1
        + [pos_ccf[0]]                # 0
        + list(pos_ccf[1:max_lag + 1]) # 1, ..., max_lag
    )

    return pd.DataFrame({"lag": lags, "correlation": corrs})


def compute_lead_lag_matrix(
    df: pd.DataFrame,
    max_lag: int = 12,
) -> pd.DataFrame:
    """For each pair of series, find the lag with maximum absolute correlation.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns with levels ('optimal_lag', 'max_correlation').
        Rows and inner columns are series names.
    """
    cols = df.columns.tolist()
    n = len(cols)
    opt_lag = np.zeros((n, n), dtype=int)
    max_corr = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                opt_lag[i, j] = 0
                max_corr[i, j] = 1.0
                continue
            ccf_df = compute_cross_correlation(df[cols[i]], df[cols[j]], max_lag=max_lag)
            idx = ccf_df["correlation"].abs().idxmax()
            opt_lag[i, j] = int(ccf_df.loc[idx, "lag"])
            max_corr[i, j] = ccf_df.loc[idx, "correlation"]

    lag_df = pd.DataFrame(opt_lag, index=cols, columns=cols)
    corr_df = pd.DataFrame(max_corr, index=cols, columns=cols)

    result = pd.concat(
        {"optimal_lag": lag_df, "max_correlation": corr_df},
        axis=1,
    )
    return result
