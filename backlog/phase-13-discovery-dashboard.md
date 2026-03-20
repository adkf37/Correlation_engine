# Phase 13: Discovery Dashboard

**Status:** Not Started
**Depends on:** Phase 10 (FindingsDatabase), Phase 12 (ScanRunner, CLI config)
**Estimated scope:** L

## Objective

Replace the existing analyst-oriented Streamlit dashboard with a discovery-feed interface that surfaces the engine's autonomous findings. The UI is read-first: the engine does the work, the analyst explores and drills down.

## Tasks

- [ ] Rewrite `dashboard/app.py`:
  - `st.set_page_config(page_title="Correlation Discovery Engine", layout="wide", initial_sidebar_state="expanded")`
  - `st.navigation()` with 6 pages (see below)
  - Sidebar: scan status badge (last scan time, n findings), quick `Run Scan` button, config path indicator
  - Session state keys: `findings_db_path`, `latest_findings`, `selected_finding`, `scan_running`
  - Load `FindingsDatabase` once per session via `@st.cache_resource`
- [ ] Create `dashboard/pages/feed.py` — **Main landing page**:
  - Heading: "Latest Findings — [scan timestamp]"
  - Filter bar: trigger type multiselect, min score slider, series category filter
  - Findings displayed as cards (3-column grid using `st.columns`):
    - Card: series A ↔ series B, score badge (color-coded: green >0.7, yellow 0.4-0.7, grey <0.4), trigger type pills, r value, lag, one-line template summary
  - Click on card → sets `st.session_state.selected_finding` and navigates to Finding Detail page
  - "No findings yet" empty state with link to Run Scan page
  - Export button: download all shown findings as CSV
- [ ] Create `dashboard/pages/finding_detail.py` — **Deep dive view**:
  - Reads `st.session_state.selected_finding` (Finding object)
  - Layout: left column (stats), right column (charts)
  - Stats panel: both series names, all numeric fields, trigger type pills, full template summary, LLM summary (if available) in expandable section
  - Charts (using existing `viz/` modules):
    - Rolling correlation chart (90-day) over full history — `plot_rolling_correlation`
    - CCF bar chart — `plot_ccf`
    - Granger p-value indicator (gauge or metric + traffic light color)
    - Rolling z-score timeline showing regime change threshold line
  - Breadcrumb: "← Back to Feed"
  - Historical performance: `FindingsDatabase.load_pair_history()` → sparkline of `interestingness_score` over time
- [ ] Create `dashboard/pages/history.py` — **Scan timeline**:
  - Top section: line chart of `top_score` per scan over time (from `load_all_scans()`)
  - Below: paginated table of all scans (timestamp, n_findings, top_score, HTML report link)
  - Pair search: text inputs for Series A and Series B → shows `load_pair_history()` as chart + table
  - "Download findings archive" → CSV of all findings across all scans
- [ ] Create `dashboard/pages/universe.py` — **Watchlist viewer**:
  - Loads `config/universe.yaml` and displays as a styled DataFrame: ID, name, category, source, status (loaded/failed from last scan)
  - Grouped by category with row counts
  - "Refresh Status" button: re-reads last scan's metadata to update loaded/failed markers
  - Read-only in this phase (editing the YAML file directly is the update path)
- [ ] Create `dashboard/pages/run_scan.py` — **Manual scan trigger**:
  - "Run Full Scan" button — disabled if scan already running (`st.session_state.scan_running`)
  - Uses `st.status()` context for live progress feedback (polls ScanRunner via thread or subprocess)
  - Dry run toggle: runs `ScanRunner.run(dry_run=True)` to preview without writing
  - Shows `ScanResult` summary after completion: n_series, n_pairs, n_findings, duration, top_score
  - "View Results" button → navigates to Feed page after scan completes
  - Note: `ScanRunner.run()` is synchronous; use `st.spinner` + `threading.Thread` to avoid blocking UI
- [ ] Create `dashboard/pages/settings.py` — **Config editor**:
  - Reads `config/scan_config.yaml`, displays editable form fields for all parameters
  - Sections matching YAML structure: Scan Parameters, Scoring Thresholds, Score Weights (with validation that weights sum to 1.0), Output Options, LLM Settings
  - LLM section: toggle enable/disable, model name input, "Test Connection" button calling `is_ollama_available()`
  - Email section: SMTP fields with password masked; "Send Test Email" button
  - "Save Config" button: writes updated YAML back to `config/scan_config.yaml`
  - Warning: changes take effect on next scan run
- [ ] Delete or archive old `dashboard/pages/` directory (the 10 analyst-tool pages from Phase 07):
  - Archive to `dashboard/pages_archive/` (don't delete — preserve for reference)
- [ ] Update `README.md`:
  - Update "Running the Dashboard" section to reflect new purpose
  - Add "Running the Scanner" CLI section with Windows Task Scheduler instructions
  - Add "Ollama Setup" section: `ollama pull llama3.2`, enable in `config/scan_config.yaml`

## Key Files

- `dashboard/app.py` — rewrite: new 6-page navigation
- `dashboard/pages/feed.py` — create: findings feed with filter + cards
- `dashboard/pages/finding_detail.py` — create: deep dive with charts
- `dashboard/pages/history.py` — create: scan timeline and pair history
- `dashboard/pages/universe.py` — create: watchlist viewer
- `dashboard/pages/run_scan.py` — create: manual scan trigger
- `dashboard/pages/settings.py` — create: YAML config editor
- `README.md` — modify: updated run instructions

## Acceptance Criteria

- `uv run streamlit run dashboard/app.py` launches without errors
- Feed page displays findings from `data/findings/` if any scan has run; shows empty state if not
- Finding detail page renders all 4 charts without error for any valid Finding
- Run Scan page triggers `ScanRunner.run(dry_run=True)` and shows result without crashing
- Settings page saves modified YAML and the change is reflected when `scan_config.yaml` is re-read
- Ollama "Test Connection" button shows green success / red failure based on actual connectivity
- No import of `correlation_engine.scheduler` or `correlation_engine.store` fails (packages exist from Phases 10/12)

## Notes

- **Threading for scan**: `st.session_state.scan_running = True` before thread start, `= False` in thread finally block. Use `st.rerun()` to refresh UI after completion. On Windows, avoid `multiprocessing` in Streamlit — use `threading.Thread` only.
- **Chart reuse**: import directly from `src/correlation_engine/viz/` — all chart functions return Plotly figures, use `st.plotly_chart(fig, use_container_width=True)`
- **Empty state design**: the Feed page MUST work gracefully before any scan has run (first launch). Show a prominent "No scans yet — run your first scan" card with a link to the Run Scan page.
- **Score badge colors**: `interestingness_score >= 0.7` → green; `0.4–0.69` → orange; `< 0.4` → grey. Use `st.markdown` with inline CSS for badge styling.
- **Trigger type pills**: display as colored tags. Use a simple HTML span approach via `st.markdown(unsafe_allow_html=True)` for visual differentiation: high_correlation=blue, regime_change=orange, granger_causality=purple, newly_emerging=green, anomalous_lag=yellow, rolling_divergence=red.
- Old pages (*Phase 07*) should be moved to `dashboard/pages_archive/` not deleted — they represent valid work and may be resurfaced as an "Expert Mode" in a future phase.
