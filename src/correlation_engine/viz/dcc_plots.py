"""DCC-GARCH result visualizations."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from correlation_engine.analysis.dcc_garch import DccResult


def plot_conditional_correlation(
    dcc_result: DccResult,
    pair: tuple[str, str],
    title: str | None = None,
) -> go.Figure:
    """Time-varying conditional correlation for a single pair."""
    series = dcc_result.conditional_correlations[pair]
    label = f"{pair[0]} vs {pair[1]}"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values,
        mode="lines", name=label,
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
    fig.update_layout(
        title=title or f"DCC Conditional Correlation: {label}",
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis=dict(range=[-1.05, 1.05]),
    )
    return fig


def plot_conditional_correlations_grid(
    dcc_result: DccResult,
    pairs: list[tuple[str, str]] | None = None,
) -> go.Figure:
    """Subplot grid of conditional correlations for multiple pairs."""
    items = pairs or list(dcc_result.conditional_correlations.keys())
    n = len(items)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    titles = [f"{a} vs {b}" for a, b in items]
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles)

    for idx, pair in enumerate(items):
        r, c = divmod(idx, cols)
        series = dcc_result.conditional_correlations[pair]
        fig.add_trace(
            go.Scatter(x=series.index, y=series.values, mode="lines", showlegend=False),
            row=r + 1, col=c + 1,
        )

    fig.update_layout(title="DCC Conditional Correlations", height=300 * rows)
    return fig


def plot_conditional_volatility(
    dcc_result: DccResult,
) -> go.Figure:
    """Conditional volatility (from Stage 1 GARCH) for each series."""
    vol = dcc_result.conditional_volatilities
    fig = go.Figure()
    for col in vol.columns:
        fig.add_trace(go.Scatter(
            x=vol.index, y=vol[col], mode="lines", name=col,
        ))
    fig.update_layout(
        title="GARCH Conditional Volatility",
        xaxis_title="Date",
        yaxis_title="Volatility",
    )
    return fig
