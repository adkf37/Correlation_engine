"""Data Loader page — CSV upload, FRED API, Yahoo Finance."""

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _store_raw(df: pd.DataFrame) -> None:
    st.session_state["raw_data"] = df
    # Clear downstream results when new data is loaded
    for key in ("clean_data", "corr_matrix", "pval_matrix", "granger_matrix",
                "dcc_result", "rolling_dict"):
        st.session_state.pop(key, None)


st.title("📂 Data Loader")

tab_csv, tab_fred, tab_yahoo = st.tabs(["CSV Upload", "FRED API", "Yahoo Finance"])

# ── CSV Upload ────────────────────────────────────────────────────────
with tab_csv:
    uploaded = st.file_uploader("Upload CSV / TSV", type=["csv", "tsv"], accept_multiple_files=True)
    if uploaded:
        frames = []
        for f in uploaded:
            sep = "\t" if f.name.endswith(".tsv") else ","
            tmp = pd.read_csv(f, sep=sep, parse_dates=True, index_col=0)
            frames.append(tmp)
        if frames:
            merged = pd.concat(frames, axis=1)
            merged.index = pd.to_datetime(merged.index)
            merged = merged.apply(pd.to_numeric, errors="coerce")
            st.dataframe(merged.head(10), use_container_width=True)
            if st.button("Use this data", key="csv_use"):
                _store_raw(merged)
                st.success(f"Loaded {merged.shape[1]} series × {merged.shape[0]} rows")

# ── FRED API ──────────────────────────────────────────────────────────
with tab_fred:
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        st.warning("Set `FRED_API_KEY` in your `.env` or environment to use FRED.")
    series_ids = st.text_input("Series IDs (comma-separated)", value="GDP,CPIAUCSL,UNRATE",
                               key="fred_ids")
    col1, col2 = st.columns(2)
    fred_start = col1.date_input("Start", value=pd.Timestamp("2010-01-01"), key="fred_start")
    fred_end = col2.date_input("End", value=pd.Timestamp("2024-12-31"), key="fred_end")

    if st.button("Fetch from FRED", key="fred_fetch") and api_key:
        with st.spinner("Fetching FRED data…"):
            from correlation_engine.ingest.fred import FredLoader
            loader = FredLoader(api_key=api_key)
            ids = [s.strip() for s in series_ids.split(",") if s.strip()]
            df = loader.load(ids, start=str(fred_start), end=str(fred_end))
            st.dataframe(df.head(10), use_container_width=True)
            _store_raw(df)
            st.success(f"Loaded {df.shape[1]} FRED series")

# ── Yahoo Finance ─────────────────────────────────────────────────────
with tab_yahoo:
    tickers = st.text_input("Tickers (comma-separated)", value="SPY,QQQ,TLT,GLD",
                            key="yf_tickers")
    col1, col2 = st.columns(2)
    yf_start = col1.date_input("Start", value=pd.Timestamp("2015-01-01"), key="yf_start")
    yf_end = col2.date_input("End", value=pd.Timestamp("2024-12-31"), key="yf_end")

    if st.button("Fetch from Yahoo Finance", key="yf_fetch"):
        with st.spinner("Fetching Yahoo Finance data…"):
            from correlation_engine.ingest.yahoo import YahooLoader
            loader = YahooLoader()
            tkrs = [s.strip() for s in tickers.split(",") if s.strip()]
            df = loader.load(tkrs, start=str(yf_start), end=str(yf_end))
            if df.empty:
                st.error("No data returned. Check tickers and date range.")
            else:
                st.dataframe(df.head(10), use_container_width=True)
                _store_raw(df)
                st.success(f"Loaded {df.shape[1]} tickers")
