"""Tests for correlation network visualization."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from correlation_engine.viz.network import (
    build_correlation_network,
    plot_correlation_network,
)


@pytest.fixture()
def identity_corr():
    """Identity matrix — no off-diagonal correlations."""
    labels = ["a", "b", "c"]
    return pd.DataFrame(np.eye(3), index=labels, columns=labels)


@pytest.fixture()
def known_corr():
    """Known structure: a-b strongly correlated, a-c moderate, b-c weak."""
    labels = ["a", "b", "c", "d"]
    mat = np.array([
        [1.0, 0.9, 0.6, 0.1],
        [0.9, 1.0, 0.3, 0.05],
        [0.6, 0.3, 1.0, -0.7],
        [0.1, 0.05, -0.7, 1.0],
    ])
    return pd.DataFrame(mat, index=labels, columns=labels)


class TestBuildNetwork:
    def test_identity_no_edges(self, identity_corr):
        G = build_correlation_network(identity_corr, threshold=0.5)
        assert G.number_of_edges() == 0
        assert G.number_of_nodes() == 3

    def test_known_structure_threshold_05(self, known_corr):
        G = build_correlation_network(known_corr, threshold=0.5)
        # Edges above 0.5: a-b (0.9), a-c (0.6), c-d (0.7)
        assert G.number_of_edges() == 3

    def test_known_structure_threshold_08(self, known_corr):
        G = build_correlation_network(known_corr, threshold=0.8)
        # Only a-b (0.9)
        assert G.number_of_edges() == 1

    def test_edge_attributes(self, known_corr):
        G = build_correlation_network(known_corr, threshold=0.5)
        # a-b edge should have positive sign
        data = G.get_edge_data("a", "b")
        assert data["sign"] == "+"
        assert data["weight"] == pytest.approx(0.9)
        # c-d edge should have negative sign
        data_cd = G.get_edge_data("c", "d")
        assert data_cd["sign"] == "-"

    def test_threshold_zero_all_edges(self, known_corr):
        G = build_correlation_network(known_corr, threshold=0.0)
        # All 6 unique pairs (4 choose 2), but only those with |r| > 0
        assert G.number_of_edges() == 6


class TestPlotNetwork:
    def test_returns_figure(self, known_corr):
        G = build_correlation_network(known_corr, threshold=0.5)
        fig = plot_correlation_network(G)
        assert isinstance(fig, go.Figure)

    def test_layouts(self, known_corr):
        G = build_correlation_network(known_corr, threshold=0.5)
        for layout in ("spring", "circular", "kamada_kawai"):
            fig = plot_correlation_network(G, layout=layout)
            assert isinstance(fig, go.Figure)

    def test_empty_graph(self, identity_corr):
        G = build_correlation_network(identity_corr, threshold=0.5)
        fig = plot_correlation_network(G)
        assert isinstance(fig, go.Figure)
