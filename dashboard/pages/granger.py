"""Granger Causality page — directional predictive testing."""

import streamlit as st
import pandas as pd

from correlation_engine.analysis.granger import (
    granger_causality_matrix,
    granger_causality_test,
    granger_summary,
)

st.title("➡️ Granger Causality")

df = st.session_state.get("clean_data", st.session_state.get("raw_data"))
if df is None:
    st.warning("No data available. Load and optionally preprocess data first.")
    st.stop()

cols = df.columns.tolist()

max_lag = st.slider("Max lag", 1, 24, 8, key="gc_maxlag")
alpha = st.slider("Significance level", 0.001, 0.10, 0.05, step=0.005, key="gc_alpha")

if st.button("Run Granger Tests"):
    with st.spinner("Running pairwise Granger tests…"):
        mat = granger_causality_matrix(df, max_lag=max_lag, significance=alpha)
        st.session_state["granger_matrix"] = mat

    # Heatmap of p-values
    st.subheader("P-Value Matrix (rows = target, cols = predictor)")
    st.dataframe(mat.style.format("{:.4f}").background_gradient(
        cmap="RdYlGn_r", vmin=0, vmax=0.1), use_container_width=True)

    # Significant pairs summary
    summary = granger_summary(mat, alpha=alpha)
    if summary:
        st.subheader(f"Significant Causal Pairs (α = {alpha})")
        st.dataframe(pd.DataFrame(summary), use_container_width=True)
    else:
        st.info("No significant Granger-causal relationships found at this α level.")

# ── Drill-down ────────────────────────────────────────────────────────
st.divider()
st.subheader("Pair Detail")
c1, c2 = st.columns(2)
target = c1.selectbox("Target", cols, index=0, key="gc_target")
predictor = c2.selectbox("Predictor", cols, index=min(1, len(cols) - 1), key="gc_pred")

if st.button("Test Pair", key="gc_pair"):
    with st.spinner("Testing…"):
        res = granger_causality_test(df, target=target, predictor=predictor,
                                     max_lag=max_lag, significance=alpha)
    st.json(res)
    if res["reject_null"]:
        st.success(
            f"**{predictor}** Granger-causes **{target}** "
            f"(lag={res['optimal_lag']}, p={res['p_value']:.4f})"
        )
    else:
        st.info(f"No significant Granger causality from {predictor} → {target}.")
