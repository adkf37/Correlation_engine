"""Correlation Matrix page — interactive heatmap with clustering toggle."""

import io

import streamlit as st

from correlation_engine.analysis.correlation import (
    compute_correlation_matrix,
)
from correlation_engine.viz.heatmap import plot_correlation_heatmap

st.title("🔥 Correlation Matrix")

# ── data guard ────────────────────────────────────────────────────────
df = st.session_state.get("clean_data", st.session_state.get("raw_data"))
if df is None:
    st.warning("No data available. Load and optionally preprocess data first.")
    st.stop()

# ── controls ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
method = col1.selectbox("Method", ["pearson", "spearman", "kendall"])
clustered = col2.checkbox("Cluster ordering", value=True)

# ── compute ───────────────────────────────────────────────────────────
corr = compute_correlation_matrix(df, method=method)
st.session_state["corr_matrix"] = corr

fig = plot_correlation_heatmap(corr, interactive=True, clustered=clustered,
                               title=f"{method.title()} Correlation Matrix")
st.plotly_chart(fig, use_container_width=True)

# ── exports ───────────────────────────────────────────────────────────
st.subheader("Export")
csv = corr.to_csv()
st.download_button("Download CSV", csv, file_name="correlation_matrix.csv", mime="text/csv")
