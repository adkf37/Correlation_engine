"""Correlation matrix computation and hierarchical clustering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def compute_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute pairwise correlation matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Columns are series, rows are observations.
    method : str
        'pearson', 'spearman', or 'kendall'.

    Returns
    -------
    pd.DataFrame
        NxN symmetric correlation matrix.
    """
    valid = ("pearson", "spearman", "kendall")
    if method not in valid:
        raise ValueError(f"Unknown method '{method}'. Choose from: {valid}")
    return df.corr(method=method)


def cluster_correlation_matrix(
    corr_matrix: pd.DataFrame,
    linkage_method: str = "ward",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Reorder a correlation matrix by hierarchical clustering.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Symmetric NxN correlation matrix.
    linkage_method : str
        Linkage algorithm (default: 'ward').

    Returns
    -------
    tuple of (reordered_matrix, linkage_matrix)
        The reordered correlation matrix and the raw linkage array
        (usable for dendrograms).
    """
    # Distance = 1 - |corr|  (0 = identical, 1 = uncorrelated)
    dist = 1 - np.abs(corr_matrix.values)
    np.fill_diagonal(dist, 0)  # ensure exact 0 on diagonal

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)
    order = leaves_list(Z)

    labels = corr_matrix.columns[order]
    reordered = corr_matrix.loc[labels, labels]
    return reordered, Z
