"""Correlation Engine — Streamlit dashboard entry point."""

import streamlit as st

st.set_page_config(
    page_title="Correlation Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation ────────────────────────────────────────────────────────
pages = [
    st.Page("pages/home.py", title="Home", icon="🏠"),
    st.Page("pages/data_loader.py", title="Data Loader", icon="📂"),
    st.Page("pages/preprocessing.py", title="Preprocessing", icon="🔧"),
    st.Page("pages/correlation_matrix.py", title="Correlation Matrix", icon="🔥"),
    st.Page("pages/lead_lag.py", title="Lead / Lag", icon="⏱️"),
    st.Page("pages/rolling.py", title="Rolling Correlations", icon="📈"),
    st.Page("pages/significance.py", title="Significance", icon="🎯"),
    st.Page("pages/granger.py", title="Granger Causality", icon="➡️"),
    st.Page("pages/dcc_garch.py", title="DCC-GARCH", icon="📉"),
    st.Page("pages/network.py", title="Network Graph", icon="🕸️"),
]

nav = st.navigation(pages)
nav.run()
