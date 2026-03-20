"""Rolling correlation and window sensitivity visualizations."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def plot_rolling_correlation(
    rolling_series: pd.Series,
    pair_label: str | None = None,
) -> go.Figure:
    """Line plot of a single rolling correlation series."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_series.index,
        y=rolling_series.values,
        mode="lines",
        name=pair_label or "Rolling Correlation",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
    fig.update_layout(
        title=pair_label or "Rolling Correlation",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1.05, 1.05]),
    )
    return fig


def plot_rolling_multi(
    rolling_dict: dict[tuple[str, str], pd.Series],
    pairs: list[tuple[str, str]] | None = None,
) -> go.Figure:
    """Overlay multiple rolling correlation series."""
    fig = go.Figure()
    items = {k: v for k, v in rolling_dict.items() if k in pairs} if pairs else rolling_dict

    for (a, b), series in items.items():
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name=f"{a} vs {b}",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
    fig.update_layout(
        title="Rolling Correlations",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1.05, 1.05]),
    )
    return fig


def plot_window_sensitivity(
    sensitivity_df: pd.DataFrame,
    pair_label: str | None = None,
) -> go.Figure:
    """One line per window size showing how rolling correlation varies."""
    fig = go.Figure()
    for col in sensitivity_df.columns:
        fig.add_trace(go.Scatter(
            x=sensitivity_df.index,
            y=sensitivity_df[col],
            mode="lines",
            name=f"window={col}",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
    fig.update_layout(
        title=f"Window Sensitivity{f' — {pair_label}' if pair_label else ''}",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1.05, 1.05]),
    )
    return fig


def plot_significance_heatmap(
    corr_matrix: pd.DataFrame,
    pvalue_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> go.Figure:
    """Heatmap where non-significant cells are visually dimmed.

    Cells with adjusted p-value >= alpha are overlaid with an 'X'.
    """
    labels = corr_matrix.columns.tolist()
    z = corr_matrix.values
    n = len(labels)

    # Annotation text: value for significant, "ns" for non-significant
    text = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append("1.00")
            elif pvalue_matrix.iloc[i, j] < alpha:
                row.append(f"{z[i, j]:.2f}")
            else:
                row.append(f"<b>✗</b> {z[i, j]:.2f}")
        text.append(row)

    # Opacity mask: dim non-significant
    opacity = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            elif pvalue_matrix.iloc[i, j] < alpha:
                row.append(1.0)
            else:
                row.append(0.3)
        opacity.append(row)

    fig = go.Figure(data=go.Heatmap(
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
    ))
    fig.update_layout(
        title=f"Significance-Filtered Correlation (α={alpha})",
        xaxis=dict(tickangle=-45),
        width=600 + 30 * n,
        height=500 + 30 * n,
    )
    return fig
