"""DCC-GARCH: two-stage Dynamic Conditional Correlation estimation.

Stage 1 — fit univariate GARCH(p,q) to each series (via ``arch``).
Stage 2 — estimate DCC(1,1) parameters from standardised residuals
           using quasi-maximum-likelihood (Engle 2002).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class DccResult:
    """Container for DCC-GARCH output."""

    conditional_correlations: dict[tuple[str, str], pd.Series]
    conditional_volatilities: pd.DataFrame
    model_params: dict  # {'a': float, 'b': float}
    convergence_info: dict


def fit_dcc_garch(
    df: pd.DataFrame,
    p: int = 1,
    q: int = 1,
) -> DccResult:
    """Fit a DCC-GARCH(1,1) model.

    Parameters
    ----------
    df : pd.DataFrame
        Columns are return series (not price levels).
    p, q : int
        GARCH order for univariate stage.

    Returns
    -------
    DccResult
    """
    from arch.univariate import ConstantMean, GARCH

    cols = df.columns.tolist()
    k = len(cols)

    if k > 10:
        warnings.warn(
            f"DCC-GARCH with {k} series is computationally expensive. "
            "Consider reducing to ≤10 series."
        )
    if len(df) < 250:
        warnings.warn(
            f"Only {len(df)} observations — DCC-GARCH requires ≥250 "
            "for reliable estimation."
        )

    # Detect if data looks like price levels (all positive, low variance of first diffs)
    all_pos = (df > 0).all().all()
    if all_pos and df.mean().mean() > 10:
        warnings.warn(
            "Data appears to be price levels rather than returns. "
            "Consider passing log-returns instead."
        )

    data = df.dropna()
    T = len(data)

    # ── Stage 1: Univariate GARCH ────────────────────────────────────
    std_resids = np.empty((T, k))
    cond_vol = pd.DataFrame(index=data.index, columns=cols, dtype=float)

    for i, col in enumerate(cols):
        series = data[col] * 100  # rescale for numeric stability
        am = ConstantMean(series)
        am.volatility = GARCH(p=p, o=0, q=q)
        res = am.fit(disp="off", show_warning=False)
        std_resids[:, i] = res.std_resid
        cond_vol[col] = res.conditional_volatility / 100  # un-scale

    # ── Stage 2: DCC parameter estimation ────────────────────────────
    # Q̄ = unconditional correlation of standardised residuals
    Q_bar = np.corrcoef(std_resids, rowvar=False)

    # Optimise DCC(1,1) log-likelihood for (a, b)
    result = _fit_dcc_params(std_resids, Q_bar)
    a_hat, b_hat = result.x
    converged = result.success

    # ── Compute time-varying correlations ────────────────────────────
    Qt = np.empty((T, k, k))
    Rt = np.empty((T, k, k))
    Qt[0] = Q_bar.copy()
    Rt[0] = Q_bar.copy()

    for t in range(1, T):
        e = std_resids[t - 1].reshape(-1, 1)
        Qt[t] = (1 - a_hat - b_hat) * Q_bar + a_hat * (e @ e.T) + b_hat * Qt[t - 1]
        # Normalise: R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
        d = np.sqrt(np.diag(Qt[t]))
        d[d == 0] = 1.0
        D_inv = np.diag(1.0 / d)
        Rt[t] = D_inv @ Qt[t] @ D_inv

    # Package pair-wise conditional correlations
    cond_corrs: dict[tuple[str, str], pd.Series] = {}
    for i in range(k):
        for j in range(i + 1, k):
            pair = (cols[i], cols[j])
            cond_corrs[pair] = pd.Series(
                np.clip(Rt[:, i, j], -1, 1),
                index=data.index,
                name=f"{cols[i]}_vs_{cols[j]}",
            )

    return DccResult(
        conditional_correlations=cond_corrs,
        conditional_volatilities=cond_vol,
        model_params={"a": float(a_hat), "b": float(b_hat)},
        convergence_info={
            "converged": bool(converged),
            "message": result.message if hasattr(result, "message") else "",
            "n_obs": T,
            "n_series": k,
        },
    )


# ── internal helpers ──────────────────────────────────────────────────


def _fit_dcc_params(
    std_resids: np.ndarray,
    Q_bar: np.ndarray,
) -> object:
    """Estimate (a, b) by maximising DCC quasi-log-likelihood."""
    T, k = std_resids.shape

    def neg_loglik(params):
        a, b = params
        if a < 0 or b < 0 or a + b >= 1:
            return 1e12
        Qt = Q_bar.copy()
        ll = 0.0
        for t in range(1, T):
            e = std_resids[t - 1].reshape(-1, 1)
            Qt = (1 - a - b) * Q_bar + a * (e @ e.T) + b * Qt
            d = np.sqrt(np.diag(Qt))
            d[d == 0] = 1.0
            D_inv = np.diag(1.0 / d)
            Rt = D_inv @ Qt @ D_inv
            et = std_resids[t].reshape(-1, 1)
            sign, logdet = np.linalg.slogdet(Rt)
            if sign <= 0:
                return 1e12
            ll += logdet + (et.T @ np.linalg.solve(Rt, et) - et.T @ et).item()
        return ll

    result = minimize(
        neg_loglik,
        x0=[0.05, 0.90],
        method="L-BFGS-B",
        bounds=[(1e-6, 0.5), (1e-6, 0.9999)],
        options={"maxiter": 200},
    )
    # Ensure a + b < 1
    if result.x[0] + result.x[1] >= 1:
        result.x = np.array([0.01, 0.95])
    return result
