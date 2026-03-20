"""Granger causality testing — pairwise and matrix."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def granger_causality_test(
    df: pd.DataFrame,
    target: str,
    predictor: str,
    max_lag: int = 12,
    significance: float = 0.05,
) -> dict:
    """Test whether *predictor* Granger-causes *target*.

    Pre-checks stationarity and auto-differences if needed.

    Returns
    -------
    dict with keys: optimal_lag, f_stat, p_value, reject_null, differenced
    """
    data = df[[target, predictor]].dropna()
    differenced = False

    # Stationarity check — auto-difference if needed
    for col in [target, predictor]:
        adf_p = adfuller(data[col], autolag="AIC")[1]
        if adf_p > significance:
            warnings.warn(
                f"'{col}' appears non-stationary (ADF p={adf_p:.4f}). "
                "Auto-differencing for Granger test."
            )
            data = data.diff().iloc[1:]
            differenced = True
            break  # re-difference both at once

    if len(data) < max_lag + 2:
        return {
            "optimal_lag": 0,
            "f_stat": np.nan,
            "p_value": 1.0,
            "reject_null": False,
            "differenced": differenced,
        }

    # grangercausalitytests expects [target, predictor]
    # Suppress its print output by catching verbose
    try:
        results = grangercausalitytests(
            data[[target, predictor]],
            maxlag=max_lag,
            verbose=False,
        )
    except Exception:
        return {
            "optimal_lag": 0,
            "f_stat": np.nan,
            "p_value": 1.0,
            "reject_null": False,
            "differenced": differenced,
        }

    # Pick lag with lowest p-value (F-test)
    best_lag = 1
    best_p = 1.0
    best_f = 0.0
    for lag, res in results.items():
        f_test = res[0]["ssr_ftest"]
        f_val, p_val = f_test[0], f_test[1]
        if p_val < best_p:
            best_p = p_val
            best_f = f_val
            best_lag = lag

    return {
        "optimal_lag": int(best_lag),
        "f_stat": float(best_f),
        "p_value": float(best_p),
        "reject_null": bool(best_p < significance),
        "differenced": differenced,
    }


def granger_causality_matrix(
    df: pd.DataFrame,
    max_lag: int = 12,
    significance: float = 0.05,
) -> pd.DataFrame:
    """Pairwise Granger causality p-values for all column combinations.

    Returns
    -------
    pd.DataFrame
        Rows = targets, columns = predictors.  Cell (i,j) = p-value
        for "column j Granger-causes column i".  Diagonal = NaN.
    """
    cols = df.columns.tolist()
    n = len(cols)
    pvals = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            res = granger_causality_test(
                df, target=cols[i], predictor=cols[j],
                max_lag=max_lag, significance=significance,
            )
            pvals[i, j] = res["p_value"]

    return pd.DataFrame(pvals, index=cols, columns=cols)


def granger_summary(
    granger_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> list[dict]:
    """Extract significant causal pairs from a Granger matrix.

    Returns list of dicts sorted by p-value, each with keys:
    predictor, target, p_value.
    """
    results: list[dict] = []
    cols = granger_matrix.columns.tolist()
    for target in cols:
        for predictor in cols:
            if target == predictor:
                continue
            p = granger_matrix.loc[target, predictor]
            if np.isnan(p):
                continue
            if p < alpha:
                results.append({
                    "predictor": predictor,
                    "target": target,
                    "p_value": float(p),
                })
    return sorted(results, key=lambda x: x["p_value"])
