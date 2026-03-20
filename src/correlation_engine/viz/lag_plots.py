"""Cross-correlation and lead/lag visualizations."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_ccf(
    ccf_df: pd.DataFrame,
    title: str | None = None,
) -> go.Figure:
    """Bar chart of cross-correlation vs. lag for one pair.

    Parameters
    ----------
    ccf_df : pd.DataFrame
        Output of compute_cross_correlation() with columns 'lag' and 'correlation'.
    title : str, optional
        Chart title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    colors = ["#d32f2f" if c < 0 else "#1976d2" for c in ccf_df["correlation"]]

    fig = go.Figure(
        go.Bar(
            x=ccf_df["lag"],
            y=ccf_df["correlation"],
            marker_color=colors,
        )
    )
    fig.update_layout(
        title=title or "Cross-Correlation Function",
        xaxis_title="Lag",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1.05, 1.05]),
    )
    # Highlight zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
    return fig


def plot_lead_lag_matrix(
    lead_lag_df: pd.DataFrame,
) -> go.Figure:
    """Heatmap of optimal lags with correlation annotations.

    Parameters
    ----------
    lead_lag_df : pd.DataFrame
        Output of compute_lead_lag_matrix() with MultiIndex columns
        ('optimal_lag', 'max_correlation').

    Returns
    -------
    plotly.graph_objects.Figure
    """
    lag_matrix = lead_lag_df["optimal_lag"]
    corr_matrix = lead_lag_df["max_correlation"]
    labels = lag_matrix.columns.tolist()

    # Text: show "lag=N (r=0.XX)" on hover
    text = [
        [
            f"lag={int(lag_matrix.iloc[i, j])}, r={corr_matrix.iloc[i, j]:.2f}"
            for j in range(len(labels))
        ]
        for i in range(len(labels))
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=lag_matrix.values,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Viridis",
            colorbar=dict(title="Optimal Lag"),
        )
    )
    fig.update_layout(
        title="Lead/Lag Matrix (optimal lag per pair)",
        xaxis=dict(tickangle=-45),
        width=600 + 30 * len(labels),
        height=500 + 30 * len(labels),
    )
    return fig
