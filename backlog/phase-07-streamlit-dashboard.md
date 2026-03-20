# Phase 07: Streamlit Dashboard

**Status:** Not Started
**Depends on:** Phases 02–06 (can scaffold structure in parallel with Phases 5–6, wiring in pages as analysis modules land)
**Estimated scope:** L

## Objective

Build a multi-page Streamlit dashboard that exposes every feature of the `correlation_engine` library through an interactive, explorable UI. The dashboard is the primary user-facing deliverable — it should be intuitive enough that someone with no Python knowledge can upload data and get results.

## Tasks

### App Shell & Navigation
- [ ] Implement `dashboard/app.py` using `st.navigation()` + `st.Page()` pattern
  - Global sidebar: date range selector, frequency picker, correlation method toggle (Pearson/Spearman/Kendall)
  - Store shared state in `st.session_state` (loaded DataFrame, preprocessed DataFrame, analysis results)
  - Navigation pages: Home, Data Loader, Preprocessing, Correlation Matrix, Lead/Lag, Rolling Correlations, Significance, Granger Causality, DCC-GARCH, Network Graph

### Page: Home (`dashboard/pages/home.py`)
- [ ] Welcome page with project overview, quick-start instructions
  - Show which data is currently loaded (if any) from session state
  - Links/buttons to jump to each analysis page

### Page: Data Loader (`dashboard/pages/data_loader.py`)
- [ ] **CSV Upload** tab: `st.file_uploader()` accepting CSV/TSV files, preview with `st.dataframe()`
- [ ] **FRED API** tab: text input for series IDs (comma-separated), date range picker, "Fetch" button
  - Show setup prompt if `FRED_API_KEY` not found in environment
  - Display fetched series names and date ranges
- [ ] **Yahoo Finance** tab: text input for tickers, date range picker, "Fetch" button
  - Validate tickers, warn on empty results
- [ ] **Combine** section: option to merge data from multiple sources into one DataFrame
- [ ] Store loaded data in `st.session_state['raw_data']`
- [ ] Use `@st.cache_data` for API fetches

### Page: Preprocessing (`dashboard/pages/preprocessing.py`)
- [ ] Show `report_missing()` summary table
- [ ] Dropdowns for: target frequency, missing data strategy, stationarity method
- [ ] "Run Preprocessing" button → executes `PreprocessingPipeline`
- [ ] Before/after comparison: side-by-side data previews, line charts
- [ ] Show ADF test results table (`check_stationarity_all()`)
- [ ] Store result in `st.session_state['clean_data']`

### Page: Correlation Matrix (`dashboard/pages/correlation_matrix.py`)
- [ ] Interactive Plotly heatmap via `plot_correlation_heatmap()`
- [ ] Toggle: clustered ordering on/off
- [ ] Toggle: Pearson / Spearman / Kendall (re-computes on change)
- [ ] Download buttons: PNG, CSV export of matrix
- [ ] Highlight: click a cell to drill into that pair's detail (link to Lead/Lag page)

### Page: Lead/Lag Analysis (`dashboard/pages/lead_lag.py`)
- [ ] Pair selector: two dropdowns to pick series X and Y
- [ ] Max lag slider (1–36)
- [ ] CCF bar chart for the selected pair via `plot_ccf()`
- [ ] Full lead/lag summary matrix via `plot_lead_lag_matrix()`
- [ ] Text summary: "Series X leads Series Y by N periods (r = 0.XX)"

### Page: Rolling Correlations (`dashboard/pages/rolling.py`)
- [ ] Multi-select for pairs to display
- [ ] Window size slider (10–360)
- [ ] Rolling correlation time-series plot via `plot_rolling_correlation()`
- [ ] Window sensitivity analysis: checkbox to show multiple window sizes overlaid

### Page: Significance Testing (`dashboard/pages/significance.py`)
- [ ] Bootstrap CI settings: number of bootstrap samples, confidence level
- [ ] "Run Bootstrap" button (with spinner — this is slow)
- [ ] Display CI matrix alongside point estimates
- [ ] P-value matrix with multiple testing correction dropdown (Bonferroni, FDR)
- [ ] Significance heatmap: non-significant cells visually dimmed

### Page: Granger Causality (`dashboard/pages/granger.py`)
- [ ] Max lag selector
- [ ] "Run Granger Tests" button
- [ ] Granger causality matrix heatmap (p-values, significant cells highlighted)
- [ ] Summary table of significant causal relationships via `granger_summary()`
- [ ] Drill-down: select a pair to see detailed F-statistics at each lag

### Page: DCC-GARCH (`dashboard/pages/dcc_garch.py`)
- [ ] Series multi-select (enforce ≤10 with `st.warning` if exceeded)
- [ ] "Fit DCC-GARCH" button (with spinner + estimated time warning)
- [ ] Conditional correlation time-series plots for selected pairs
- [ ] Conditional volatility plot
- [ ] Model parameter summary table
- [ ] Minimum data requirement check (≥250 observations)

### Page: Network Graph (`dashboard/pages/network.py`)
- [ ] Correlation threshold slider (0.0–1.0)
- [ ] Layout selector (spring, circular, kamada_kawai)
- [ ] Interactive Plotly network graph
- [ ] Node/edge count summary
- [ ] Only show this page if ≥5 series loaded (otherwise show helpful message)

### Configuration & Polish
- [ ] Create `dashboard/.streamlit/config.toml` with theme settings
- [ ] Add error handling: friendly messages when data isn't loaded yet (redirect to Data Loader)
- [ ] Add `@st.cache_data` to all expensive computations
- [ ] Add download/export buttons on every analysis page (CSV for data, PNG for charts)

### Tests
- [ ] Manual smoke test checklist:
  - [ ] Load sample CSVs → all pages render without errors
  - [ ] Fetch FRED data (if API key available) → end-to-end flow works
  - [ ] Fetch Yahoo Finance data → end-to-end flow works
  - [ ] Change correlation method → heatmap updates
  - [ ] Adjust lag slider → CCF chart updates
  - [ ] Run bootstrap → CI overlay appears
  - [ ] Run Granger → matrix renders with significant cells highlighted
  - [ ] Fit DCC-GARCH on small dataset → conditional correlations plot
  - [ ] Adjust network threshold → graph updates dynamically

## Key Files

- `dashboard/app.py` — main entry point with `st.navigation()`
- `dashboard/pages/home.py` — welcome page
- `dashboard/pages/data_loader.py` — CSV upload + FRED + Yahoo fetching
- `dashboard/pages/preprocessing.py` — preprocessing configuration & execution
- `dashboard/pages/correlation_matrix.py` — interactive heatmap
- `dashboard/pages/lead_lag.py` — CCF and lead/lag matrix
- `dashboard/pages/rolling.py` — rolling window correlations
- `dashboard/pages/significance.py` — bootstrap CI + p-value correction
- `dashboard/pages/granger.py` — Granger causality testing
- `dashboard/pages/dcc_garch.py` — DCC-GARCH fitting & visualization
- `dashboard/pages/network.py` — correlation network graph
- `dashboard/.streamlit/config.toml` — Streamlit theme configuration

## Acceptance Criteria

- `streamlit run dashboard/app.py` starts without errors
- All 10 pages render and are navigable via sidebar
- Loading sample CSV data and walking through every page produces correct visualizations
- FRED and Yahoo Finance integration works when credentials/network available
- Expensive operations (bootstrap, DCC-GARCH) show progress spinners
- Every page has a working export/download option
- Pages that require data show a friendly "load data first" message instead of crashing
- Session state persists across page navigation (don't lose data when switching pages)

## Notes

- **Scaffolding can start early:** The app shell, navigation, Data Loader, and Preprocessing pages only depend on Phases 2–3. Build these while Phases 5–6 are in progress, then wire in remaining pages as analysis modules land
- Use `st.session_state` as the single source of truth. Every analysis page checks for prerequisite data and redirects gracefully
- `@st.cache_data` is critical for UX — without it, switching pages re-computes everything
- For DCC-GARCH, consider showing estimated compute time based on dataset size before the user clicks "Fit"
- The dashboard should work fully with just CSV uploads and no API keys — FRED/Yahoo are enhancements, not requirements
