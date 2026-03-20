"""Bootstrap confidence intervals and multiple-testing-corrected p-values."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def bootstrap_correlation_ci(
    x: pd.Series,
    y: pd.Series,
    method: str = "pearson",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> dict:
    """Block-bootstrap confidence interval for a pairwise correlation.

    Uses contiguous-block resampling to preserve autocorrelation.

    Parameters
    ----------
    x, y : pd.Series
        Two aligned time series.
    method : str
        'pearson', 'spearman', or 'kendall'.
    n_boot : int
        Number of bootstrap replications.
    alpha : float
        Significance level (CI = 1 - alpha).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: point_estimate, ci_lower, ci_upper, se
    """
    shared = x.dropna().index.intersection(y.dropna().index)
    xv = x.loc[shared].values.astype(float)
    yv = y.loc[shared].values.astype(float)
    n = len(xv)

    corr_fn = _corr_func(method)
    point_estimate = corr_fn(xv, yv)

    block_len = max(int(n ** (1 / 3)), 1)
    rng = np.random.default_rng(seed)

    boot_corrs = np.empty(n_boot)
    for b in range(n_boot):
        idx = _block_bootstrap_indices(n, block_len, rng)
        boot_corrs[b] = corr_fn(xv[idx], yv[idx])

    lo = np.nanpercentile(boot_corrs, 100 * alpha / 2)
    hi = np.nanpercentile(boot_corrs, 100 * (1 - alpha / 2))

    return {
        "point_estimate": float(point_estimate),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "se": float(np.nanstd(boot_corrs, ddof=1)),
    }


def bootstrap_correlation_matrix_ci(
    df: pd.DataFrame,
    method: str = "pearson",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bootstrap CI for every pair in *df*.

    Returns
    -------
    (point_estimates, ci_lower, ci_upper) — three NxN DataFrames.
    """
    cols = df.columns.tolist()
    n = len(cols)
    pe = np.ones((n, n))
    lo = np.ones((n, n))
    hi = np.ones((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            res = bootstrap_correlation_ci(
                df[cols[i]], df[cols[j]],
                method=method, n_boot=n_boot, alpha=alpha, seed=seed,
            )
            pe[i, j] = pe[j, i] = res["point_estimate"]
            lo[i, j] = lo[j, i] = res["ci_lower"]
            hi[i, j] = hi[j, i] = res["ci_upper"]

    kw = dict(index=cols, columns=cols)
    return pd.DataFrame(pe, **kw), pd.DataFrame(lo, **kw), pd.DataFrame(hi, **kw)


# ── p-value matrix & multiple-testing correction ──────────────────────


def compute_pvalue_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """P-value for every pairwise correlation.

    Returns
    -------
    pd.DataFrame  NxN of p-values (diagonal = 0.0).
    """
    cols = df.columns.tolist()
    n = len(cols)
    pvals = np.zeros((n, n))

    test_fn = _pvalue_func(method)

    for i in range(n):
        for j in range(i + 1, n):
            shared = df[[cols[i], cols[j]]].dropna()
            _, p = test_fn(shared.iloc[:, 0].values, shared.iloc[:, 1].values)
            pvals[i, j] = pvals[j, i] = p

    return pd.DataFrame(pvals, index=cols, columns=cols)


def adjust_pvalues(
    pvalue_matrix: pd.DataFrame,
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Apply multiple-testing correction to a p-value matrix.

    Parameters
    ----------
    method : str
        'bonferroni', 'fdr_bh', 'fdr_by', or 'holm'.
    """
    valid = ("bonferroni", "fdr_bh", "fdr_by", "holm")
    if method not in valid:
        raise ValueError(f"Unknown correction method '{method}'. Choose from {valid}")

    cols = pvalue_matrix.columns
    vals = pvalue_matrix.values.copy()
    n = len(cols)

    # Extract upper-triangle p-values (avoid double-counting)
    upper_idx = np.triu_indices(n, k=1)
    raw_p = vals[upper_idx]

    _, corrected, _, _ = multipletests(raw_p, method=method)

    adjusted = np.zeros_like(vals)
    adjusted[upper_idx] = corrected
    adjusted.T[upper_idx] = corrected  # mirror

    return pd.DataFrame(adjusted, index=cols, columns=cols)


def flag_significant(
    adjusted_pvalues: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Boolean matrix: True where correlation is statistically significant."""
    result = adjusted_pvalues < alpha
    np.fill_diagonal(result.values, False)
    return result


# ── helpers ───────────────────────────────────────────────────────────


def _block_bootstrap_indices(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """Generate block-bootstrap sample indices of length *n*."""
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
    return idx[:n]


def _corr_func(method: str):
    """Return a (x, y) -> float correlation function."""
    if method == "pearson":
        return lambda x, y: np.corrcoef(x, y)[0, 1]
    elif method == "spearman":
        return lambda x, y: stats.spearmanr(x, y).statistic
    elif method == "kendall":
        return lambda x, y: stats.kendalltau(x, y).statistic
    raise ValueError(f"Unknown method '{method}'")


def _pvalue_func(method: str):
    """Return a (x, y) -> (statistic, pvalue) function."""
    if method == "pearson":
        return stats.pearsonr
    elif method == "spearman":
        return stats.spearmanr
    elif method == "kendall":
        return stats.kendalltau
    raise ValueError(f"Unknown method '{method}'")
