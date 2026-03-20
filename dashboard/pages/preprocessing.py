"""Preprocessing page — align, fill missing, test/apply stationarity."""

import streamlit as st
import pandas as pd

from correlation_engine.preprocessing import (
    PreprocessingPipeline,
    align_frequencies,
    check_stationarity_all,
    handle_missing,
    report_missing,
)

st.title("🔧 Preprocessing")

if "raw_data" not in st.session_state or st.session_state["raw_data"] is None:
    st.warning("No data loaded. Go to **Data Loader** first.")
    st.stop()

raw: pd.DataFrame = st.session_state["raw_data"]

# ── Missing data report ──────────────────────────────────────────────
st.subheader("Missing Data")
missing = report_missing(raw)
st.dataframe(missing, use_container_width=True)

# ── Pipeline configuration ───────────────────────────────────────────
st.subheader("Pipeline Settings")
col1, col2, col3 = st.columns(3)
freq = col1.selectbox("Target frequency", ["D", "W", "M", "Q", "Y"], index=2)
missing_strategy = col2.selectbox("Missing data strategy",
                                  ["ffill", "bfill", "interpolate", "drop_rows"])
stat_method = col3.selectbox("Stationarity transform", ["none", "diff", "log_diff", "detrend"])

if st.button("Run Preprocessing"):
    steps: list[tuple[str, dict]] = [
        ("align", {"target_freq": freq}),
        ("missing", {"strategy": missing_strategy}),
    ]
    if stat_method != "none":
        steps.append(("transform", {"method": stat_method}))

    pipeline = PreprocessingPipeline(steps=steps)
    with st.spinner("Running pipeline…"):
        clean = pipeline.run(raw)

    st.session_state["clean_data"] = clean

    # Before / after
    st.subheader("Before → After")
    c1, c2 = st.columns(2)
    c1.caption("Raw (head)")
    c1.dataframe(raw.head(10), use_container_width=True)
    c2.caption("Clean (head)")
    c2.dataframe(clean.head(10), use_container_width=True)

    # Stationarity report
    st.subheader("Stationarity (ADF test)")
    adf = check_stationarity_all(clean)
    st.dataframe(adf, use_container_width=True)

    # Pipeline summary
    report = pipeline.report()
    if report:
        st.subheader("Pipeline Report")
        st.json(report)

    st.success(f"Preprocessed: {clean.shape[1]} series × {clean.shape[0]} rows")
