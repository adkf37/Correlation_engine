"""Correlation network graph visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx


def build_correlation_network(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.5,
) -> nx.Graph:
    """Build a graph where edges are correlations above *threshold*.

    Nodes = series names.  Edge attributes: weight (signed correlation),
    abs_weight, sign ('+' or '-').
    """
    G = nx.Graph()
    cols = corr_matrix.columns.tolist()
    G.add_nodes_from(cols)

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr_matrix.iloc[i, j]
            if abs(r) >= threshold:
                G.add_edge(
                    cols[i], cols[j],
                    weight=float(r),
                    abs_weight=float(abs(r)),
                    sign="+" if r > 0 else "-",
                )
    return G


def plot_correlation_network(
    graph: nx.Graph,
    layout: str = "spring",
    title: str = "Correlation Network",
) -> go.Figure:
    """Interactive Plotly network graph.

    Node size ~ degree.  Edge thickness ~ |correlation|.
    Edge colour: blue = positive, red = negative.
    """
    pos = _get_layout(graph, layout)

    # Edges
    edge_traces = []
    for u, v, d in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        color = "#1976d2" if d.get("sign") == "+" else "#d32f2f"
        width = 1 + 4 * d.get("abs_weight", 0.5)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color=color),
            hoverinfo="text",
            text=f"{u}↔{v}: {d.get('weight', 0):.2f}",
            showlegend=False,
        ))

    # Nodes
    degrees = dict(graph.degree())
    node_x = [pos[n][0] for n in graph.nodes()]
    node_y = [pos[n][1] for n in graph.nodes()]
    node_size = [10 + 5 * degrees.get(n, 0) for n in graph.nodes()]
    node_text = [f"{n} (deg={degrees.get(n, 0)})" for n in graph.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=node_size, color="#455a64", line=dict(width=1, color="white")),
        text=list(graph.nodes()),
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
    )
    return fig


def _get_layout(graph: nx.Graph, layout: str) -> dict:
    """Compute node positions for the given layout algorithm."""
    if layout == "circular":
        return nx.circular_layout(graph)
    elif layout == "kamada_kawai":
        try:
            if nx.is_connected(graph) and graph.number_of_nodes() > 0:
                return nx.kamada_kawai_layout(graph)
            return nx.spring_layout(graph, seed=42)
        except Exception:
            return nx.spring_layout(graph, seed=42)
    else:
        return nx.spring_layout(graph, seed=42)
