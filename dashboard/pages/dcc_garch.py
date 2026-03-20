"""DCC-GARCH page — dynamic conditional correlations."""

import streamlit as st

from correlation_engine.analysis.dcc_garch import fit_dcc_garch
from correlation_engine.viz.dcc_plots import (
    plot_conditional_correlation,
    plot_conditional_correlations_grid,
    plot_conditional_volatility,
)

st.title("📉 DCC-GARCH")

df = st.session_state.get("clean_data", st.session_state.get("raw_data"))
if df is None:
    st.warning("No data available. Load and optionally preprocess data first.")
    st.stop()

cols = df.columns.tolist()

st.markdown("""
DCC-GARCH estimates **time-varying conditional correlations** accounting for
volatility clustering.  Data should be **returns** (not price levels).
""")

selected = st.multiselect("Select series (≤10 recommended)", cols, default=cols[:min(5, len(cols))])

if len(selected) > 10:
    st.warning("More than 10 series: computation will be slow and convergence may degrade.")

if len(selected) < 2:
    st.info("Select at least 2 series.")
    st.stop()

n_obs = len(df[selected].dropna())
if n_obs < 250:
    st.warning(f"Only {n_obs} observations after dropna. DCC-GARCH recommends ≥250.")

if st.button("Fit DCC-GARCH"):
    with st.spinner("Fitting DCC-GARCH (this may take a while)…"):
        result = fit_dcc_garch(df[selected])
        st.session_state["dcc_result"] = result

    # Model summary
    st.subheader("Model Parameters")
    st.json(result.model_params)
    st.json(result.convergence_info)

    # Conditional correlations grid
    st.subheader("Conditional Correlations")
    fig_grid = plot_conditional_correlations_grid(result)
    st.plotly_chart(fig_grid, use_container_width=True)

    # Conditional volatility
    st.subheader("Conditional Volatility")
    fig_vol = plot_conditional_volatility(result)
    st.plotly_chart(fig_vol, use_container_width=True)

# ── Drill-down on cached result ──────────────────────────────────────
if "dcc_result" in st.session_state:
    st.divider()
    st.subheader("Single Pair Detail")
    pairs = list(st.session_state["dcc_result"].conditional_correlations.keys())
    pair_labels = [f"{a} vs {b}" for a, b in pairs]
    choice = st.selectbox("Pair", pair_labels)
    idx = pair_labels.index(choice)
    fig = plot_conditional_correlation(st.session_state["dcc_result"], pairs[idx])
    st.plotly_chart(fig, use_container_width=True)
