# Phase 12: CLI Scheduler & Runner

**Status:** Not Started
**Depends on:** Phase 08, 09, 10, 11 (all previous discovery phases)
**Estimated scope:** M

## Objective

Provide a single, schedulable CLI entry point (`corr-scan`) that orchestrates the full discovery pipeline: load watchlist → scan all pairs → save findings → generate report → send email digest. All tuneable parameters are driven by YAML config files.

## Tasks

- [ ] Create `config/scan_config.yaml`:
  ```yaml
  scan:
    start_date: "2015-01-01"       # lookback start for all series
    target_freq: "M"               # monthly alignment
    rolling_window: 12             # months for rolling correlation
    min_r_for_granger: 0.3         # pre-filter: skip Granger if |r| < this
    min_score_threshold: 0.1       # only store findings above this score
    top_n_report: 25               # top N findings in HTML report / email
  
  scoring:
    correlation_threshold: 0.7
    zscore_threshold: 2.0
    granger_p_threshold: 0.05
    lag_correlation_threshold: 0.6
    weights:
      high_correlation: 0.25
      newly_emerging: 0.15
      regime_change: 0.20
      granger_causality: 0.20
      anomalous_lag: 0.10
      rolling_divergence: 0.10
  
  outputs:
    findings_dir: "data/findings"
    reports_dir: "data/reports"
    save_html_report: true
    send_email: false              # set to true with valid SMTP config
  
  llm:
    enabled: false                 # set to true if Ollama is running
    model: "llama3.2"
    host: "http://localhost:11434"
  ```
- [ ] Create `src/correlation_engine/scheduler/__init__.py` (empty init)
- [ ] Create `src/correlation_engine/scheduler/runner.py`:
  - `ScanRunner(config_path: str | Path = "config/scan_config.yaml")` class
  - `.run(dry_run: bool = False) -> ScanResult`
  - `ScanResult` dataclass: `scan_id`, `scanned_at`, `n_series_loaded`, `n_pairs_evaluated`, `n_findings`, `top_score`, `html_report_path`, `email_sent`, `duration_seconds`, `errors: list[str]`
  - Pipeline steps with timing:
    1. Load config from YAML
    2. Instantiate `Watchlist` → `.load(start_date, end_date=today)`
    3. Instantiate `FindingsDatabase(config.outputs.findings_dir)`
    4. Instantiate `DiscoveryScanner(config)` → `.scan(watchlist, db)` with progress logging
    5. For each finding: `generate_template_summary(finding)` + optional `enhance_with_llm(finding, ...)` if `config.llm.enabled`
    6. `db.save_findings(findings, scan_id, scanned_at)`
    7. If `config.outputs.save_html_report`: `generate_html_report(top_n, metadata, reports_dir/...)`
    8. If `config.outputs.send_email` and SMTP configured: `send_email_digest(top_n, email_config)`
    9. Return `ScanResult`
  - `dry_run=True`: runs steps 1-4 only (no writes), logs what would be saved
  - Logs progress at each step using Python `logging` (not print statements)
- [ ] Create `src/correlation_engine/scheduler/cli.py`:
  - Click-based CLI with top-level group `@click.group()`
  - `corr-scan run [--config PATH] [--dry-run]` — runs full scan, prints `ScanResult` summary
  - `corr-scan findings [--config PATH] [--top N] [--format table|json]` — prints top-N from latest scan
  - `corr-scan status [--config PATH]` — prints scan history table (scan time, n_findings, top_score)
  - `corr-scan check-ollama [--model MODEL]` — checks if Ollama is running and model is available
  - All commands: `--help` works, graceful error messages if findings DB empty
- [ ] Update `pyproject.toml`:
  - Add to `dependencies`: `pyyaml>=6.0`, `click>=8.0`, `jinja2>=3.1`
  - Add optional extras section: `[project.optional-dependencies]` with `llm = ["ollama>=0.1"]`
  - Add `[project.scripts]`: `corr-scan = "correlation_engine.scheduler.cli:main"`
  - Update `.env.example` with SMTP variable comments
- [ ] Create `tests/test_runner.py`:
  - Test `ScanRunner.run(dry_run=True)` with mocked Watchlist and Scanner completes without writing files
  - Test full run with mocked components produces `ScanResult` with correct counts
  - Test config loading from YAML populates all expected fields
  - Test LLM disabled → no Ollama calls made

## Key Files

- `config/scan_config.yaml` — create: all configurable parameters
- `src/correlation_engine/scheduler/__init__.py` — create: empty init
- `src/correlation_engine/scheduler/runner.py` — create: ScanRunner orchestrator
- `src/correlation_engine/scheduler/cli.py` — create: click CLI entry point
- `pyproject.toml` — modify: add deps, optional extras, scripts entry point
- `tests/test_runner.py` — create: runner tests

## Acceptance Criteria

- `corr-scan run --dry-run` prints series count, pair count, estimated findings; exits 0; writes nothing to disk
- `corr-scan run` completes full scan and produces a `data/findings/scan_*.parquet` file
- `corr-scan findings --top 10` prints a readable table of top-10 findings from latest scan
- `corr-scan status` shows at least the most recent scan's timestamp, n_findings, and top_score
- `corr-scan --help` and all subcommand `--help` work
- Config YAML changes (e.g. raise `correlation_threshold`) are reflected in next scan without code changes
- All runner tests pass with mocked dependencies

## Notes

- **Windows Task Scheduler**: to schedule daily runs, user creates a Task that executes: `C:\Users\aaron\.local\bin\uv.exe run corr-scan run` with working directory set to the project root. Document this in README.
- Config path resolution: use `pathlib.Path` and resolve relative to current working directory; also check project root as fallback (same `find_dotenv()` pattern used in fred.py)
- Logging: use `logging.getLogger("correlation_engine.scanner")` etc. — not `print()`. CLI commands set log level to INFO by default; add `--verbose` flag for DEBUG.
- The weights in `scoring.yaml` must sum to 1.0 — validate on load; raise `ValueError` with helpful message if not
- `ScanResult` should be JSON-serializable (all fields are primitives or ISO strings) for potential future webhook/notification use
