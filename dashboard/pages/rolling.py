"""Rolling Correlations page — time-varying relationships."""

import streamlit as st

from correlation_engine.analysis.rolling import (
    compute_rolling_correlation,
    compute_rolling_matrix,
    window_sensitivity,
)
from correlation_engine.viz.rolling_plots import (
    plot_rolling_correlation,
    plot_rolling_multi,
    plot_window_sensitivity,
)

st.title("📈 Rolling Correlations")

df = st.session_state.get("clean_data", st.session_state.get("raw_data"))
if df is None:
    st.warning("No data available. Load and optionally preprocess data first.")
    st.stop()

cols = df.columns.tolist()

# ── Controls ──────────────────────────────────────────────────────────
window = st.slider("Window size", 10, min(360, len(df) - 1), 60)

# ── Single pair ───────────────────────────────────────────────────────
st.subheader("Pair Rolling Correlation")
c1, c2 = st.columns(2)
x_col = c1.selectbox("Series X", cols, index=0, key="roll_x")
y_col = c2.selectbox("Series Y", cols, index=min(1, len(cols) - 1), key="roll_y")

rc = compute_rolling_correlation(df[x_col], df[y_col], window=window)
fig = plot_rolling_correlation(rc, pair_label=f"{x_col} vs {y_col}")
st.plotly_chart(fig, use_container_width=True)

# ── Window sensitivity ────────────────────────────────────────────────
if st.checkbox("Show window sensitivity"):
    ws = window_sensitivity(df[x_col], df[y_col],
                            windows=[30, 60, 90, 120, 180])
    fig_ws = plot_window_sensitivity(ws, pair_label=f"{x_col} vs {y_col}")
    st.plotly_chart(fig_ws, use_container_width=True)

# ── Multi-pair overlay ────────────────────────────────────────────────
st.subheader("All Pairs")
if st.button("Compute all pairs"):
    with st.spinner("Computing rolling correlations…"):
        rolling_dict = compute_rolling_matrix(df, window=window)
        st.session_state["rolling_dict"] = rolling_dict
    fig_all = plot_rolling_multi(rolling_dict)
    st.plotly_chart(fig_all, use_container_width=True)
