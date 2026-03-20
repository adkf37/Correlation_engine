"""Correlation heatmap visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure as MplFigure

from correlation_engine.analysis.correlation import cluster_correlation_matrix


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    interactive: bool = True,
    clustered: bool = True,
    title: str = "Correlation Matrix",
) -> go.Figure | MplFigure:
    """Render a correlation heatmap.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        NxN correlation matrix.
    interactive : bool
        True → Plotly figure, False → seaborn/matplotlib figure.
    clustered : bool
        If True, reorder rows/columns by hierarchical clustering.
    title : str
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure or matplotlib.figure.Figure
    """
    matrix = corr_matrix
    if clustered and len(corr_matrix) > 2:
        matrix, _ = cluster_correlation_matrix(corr_matrix)

    if interactive:
        return _plotly_heatmap(matrix, title)
    else:
        return _seaborn_heatmap(matrix, title)


def _plotly_heatmap(matrix: pd.DataFrame, title: str) -> go.Figure:
    labels = matrix.columns.tolist()
    z = matrix.values

    # Annotations with exact values
    text = [[f"{z[i][j]:.2f}" for j in range(len(labels))] for i in range(len(labels))]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45),
        width=600 + 30 * len(labels),
        height=500 + 30 * len(labels),
    )
    return fig


def _seaborn_heatmap(matrix: pd.DataFrame, title: str) -> MplFigure:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8 + 0.3 * len(matrix), 6 + 0.3 * len(matrix)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig
