"""Network Graph page — interactive correlation network."""

import streamlit as st

from correlation_engine.analysis.correlation import compute_correlation_matrix
from correlation_engine.viz.network import (
    build_correlation_network,
    plot_correlation_network,
)

st.title("🕸️ Correlation Network")

df = st.session_state.get("clean_data", st.session_state.get("raw_data"))
if df is None:
    st.warning("No data available. Load and optionally preprocess data first.")
    st.stop()

if len(df.columns) < 5:
    st.info(
        f"Only {len(df.columns)} series loaded. The network graph is most "
        "useful with 5+ series. For fewer series, the **Correlation Matrix** "
        "heatmap may be more informative."
    )

# ── Controls ──────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
method = c1.selectbox("Correlation method", ["pearson", "spearman", "kendall"],
                      key="net_method")
threshold = c2.slider("Threshold (|r| ≥)", 0.0, 1.0, 0.5, step=0.05, key="net_thresh")
layout = c3.selectbox("Layout", ["spring", "circular", "kamada_kawai"], key="net_layout")

# ── Compute ───────────────────────────────────────────────────────────
corr = st.session_state.get("corr_matrix")
if corr is None:
    corr = compute_correlation_matrix(df, method=method)

G = build_correlation_network(corr, threshold=threshold)

st.metric("Nodes", G.number_of_nodes())
st.metric("Edges", G.number_of_edges())

fig = plot_correlation_network(G, layout=layout,
                               title=f"Correlation Network (|r| ≥ {threshold})")
st.plotly_chart(fig, use_container_width=True)
