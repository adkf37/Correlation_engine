"""Significance Testing page — bootstrap CIs + p-value correction."""

import streamlit as st

from correlation_engine.analysis.correlation import compute_correlation_matrix
from correlation_engine.analysis.significance import (
    adjust_pvalues,
    bootstrap_correlation_matrix_ci,
    compute_pvalue_matrix,
    flag_significant,
)
from correlation_engine.viz.rolling_plots import plot_significance_heatmap

st.title("🎯 Significance Testing")

df = st.session_state.get("clean_data", st.session_state.get("raw_data"))
if df is None:
    st.warning("No data available. Load and optionally preprocess data first.")
    st.stop()

# ── P-value matrix ────────────────────────────────────────────────────
st.subheader("P-Value Matrix")
method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"],
                      key="sig_method")
correction = st.selectbox("Multiple testing correction",
                          ["fdr_bh", "bonferroni", "holm", "fdr_by"])
alpha = st.slider("Significance level (α)", 0.001, 0.10, 0.05, step=0.005)

if st.button("Compute p-values"):
    with st.spinner("Computing p-value matrix…"):
        raw_pvals = compute_pvalue_matrix(df, method=method)
        adj_pvals = adjust_pvalues(raw_pvals, method=correction)
        st.session_state["pval_matrix"] = adj_pvals

    sig = flag_significant(adj_pvals, alpha=alpha)
    n_sig = sig.sum().sum() // 2  # upper triangle only

    st.metric("Significant pairs", int(n_sig),
              delta=f"of {len(df.columns) * (len(df.columns)-1) // 2} total")

    corr = compute_correlation_matrix(df, method=method)
    fig = plot_significance_heatmap(corr, adj_pvals, alpha=alpha)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Adjusted p-values")
    st.dataframe(adj_pvals.style.format("{:.4f}"), use_container_width=True)

# ── Bootstrap CI ──────────────────────────────────────────────────────
st.divider()
st.subheader("Bootstrap Confidence Intervals")
n_boot = st.number_input("Bootstrap samples", 100, 5000, 500, step=100)

if st.button("Run Bootstrap"):
    with st.spinner(f"Running {n_boot} bootstrap replications (this may take a moment)…"):
        pe, lo, hi = bootstrap_correlation_matrix_ci(df, method=method,
                                                     n_boot=int(n_boot), seed=42)

    st.caption("Point Estimates")
    st.dataframe(pe.style.format("{:.3f}"), use_container_width=True)

    st.caption("Lower CI")
    st.dataframe(lo.style.format("{:.3f}"), use_container_width=True)

    st.caption("Upper CI")
    st.dataframe(hi.style.format("{:.3f}"), use_container_width=True)
