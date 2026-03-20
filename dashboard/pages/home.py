"""Home page — project overview and current session status."""

import streamlit as st

st.title("📊 Correlation Engine")

st.markdown("""
A multi-series correlation analysis toolkit.  Upload or fetch time-series data,
preprocess it, and explore correlations through an interactive, multi-page
dashboard.

### Quick Start
1. **Data Loader** — upload CSVs, or fetch from FRED / Yahoo Finance
2. **Preprocessing** — align frequencies, handle missing data, test stationarity
3. **Correlation Matrix** — interactive heatmap with clustering
4. **Lead / Lag** — cross-correlation analysis to detect leading indicators
5. **Rolling Correlations** — time-varying relationships with window sensitivity
6. **Significance** — bootstrap CIs and multiple-testing correction
7. **Granger Causality** — test directional predictive relationships
8. **DCC-GARCH** — dynamic conditional correlations under volatility clustering
9. **Network Graph** — correlation network with adjustable threshold
""")

st.divider()

if "raw_data" in st.session_state and st.session_state["raw_data"] is not None:
    df = st.session_state["raw_data"]
    st.success(f"**Data loaded:** {df.shape[1]} series, {df.shape[0]} observations")
    st.dataframe(df.head(), use_container_width=True)
else:
    st.info("No data loaded yet. Head to **Data Loader** to get started.")
