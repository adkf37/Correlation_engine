"""Lead / Lag analysis page — CCF bar chart + lead/lag matrix."""

import streamlit as st

from correlation_engine.analysis.lag import (
    compute_cross_correlation,
    compute_lead_lag_matrix,
)
from correlation_engine.viz.lag_plots import plot_ccf, plot_lead_lag_matrix

st.title("⏱️ Lead / Lag Analysis")

df = st.session_state.get("clean_data", st.session_state.get("raw_data"))
if df is None:
    st.warning("No data available. Load and optionally preprocess data first.")
    st.stop()

cols = df.columns.tolist()

# ── Pairwise CCF ─────────────────────────────────────────────────────
st.subheader("Cross-Correlation Function")
c1, c2, c3 = st.columns(3)
x_col = c1.selectbox("Series X", cols, index=0)
y_col = c2.selectbox("Series Y", cols, index=min(1, len(cols) - 1))
max_lag = c3.slider("Max lag", 1, 36, 12)

ccf_df = compute_cross_correlation(df[x_col], df[y_col], max_lag=max_lag)
fig = plot_ccf(ccf_df, title=f"CCF: {x_col} vs {y_col}")
st.plotly_chart(fig, use_container_width=True)

# Peak info
peak_row = ccf_df.loc[ccf_df["correlation"].abs().idxmax()]
lag_val = int(peak_row["lag"])
corr_val = peak_row["correlation"]
if lag_val > 0:
    st.info(f"**{x_col}** leads **{y_col}** by **{lag_val}** periods (r = {corr_val:.3f})")
elif lag_val < 0:
    st.info(f"**{y_col}** leads **{x_col}** by **{abs(lag_val)}** periods (r = {corr_val:.3f})")
else:
    st.info(f"Peak correlation at lag 0 (r = {corr_val:.3f})")

# ── Full matrix ──────────────────────────────────────────────────────
st.subheader("Lead/Lag Summary Matrix")
ll_df = compute_lead_lag_matrix(df, max_lag=max_lag)
fig2 = plot_lead_lag_matrix(ll_df)
st.plotly_chart(fig2, use_container_width=True)
