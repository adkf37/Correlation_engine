# Phase 10: Findings Store

**Status:** Not Started
**Depends on:** Phase 09 (Finding dataclass schema)
**Estimated scope:** S

## Objective

Persist every scan's findings to disk as a Parquet file, maintain a lightweight JSON index of all scan runs, and expose a query API for reading latest findings, pair history, and checking whether a pair is "newly emerging."

## Tasks

- [ ] Create `src/correlation_engine/store/__init__.py` (empty package init)
- [ ] Create `src/correlation_engine/store/findings_db.py`:
  - `FindingsDatabase(base_path: str | Path = "data/findings")` class
  - `.save_findings(findings: list[Finding], scan_id: str, scanned_at: datetime)`:
    - Converts findings to DataFrame via `[f.to_dict() for f in findings]`
    - Writes to `{base_path}/scan_{YYYYMMDD_HHMMSS}.parquet`
    - Updates `{base_path}/index.json` with scan entry: `{scan_id, timestamp, n_findings, top_score, parquet_file}`
    - Creates `base_path` directory if it doesn't exist
  - `.load_latest(n: int = 50) -> list[Finding]`:
    - Reads most recent scan parquet
    - Returns top-N findings sorted by `interestingness_score` descending
    - Returns `[]` if no scans exist yet
  - `.load_pair_history(series_a: str, series_b: str) -> pd.DataFrame`:
    - Reads all scan parquets (via index)
    - Filters rows where `(series_a == a AND series_b == b) OR (series_a == b AND series_b == a)`
    - Returns DataFrame with columns: `scanned_at`, `correlation`, `optimal_lag`, `interestingness_score`, `trigger_types`
    - Returns empty DataFrame if pair never seen
  - `.was_seen_before(series_a: str, series_b: str, lookback_scans: int = 3) -> bool`:
    - Checks last `lookback_scans` scan files (from index, most recent first)
    - Returns True if pair appeared in ANY of those scans
    - Used by Scanner in Phase 09 for `is_new` determination
  - `.load_all_scans() -> pd.DataFrame`:
    - Returns DataFrame from `index.json`: columns `scan_id`, `timestamp`, `n_findings`, `top_score`, `parquet_file`
    - Returns empty DataFrame if no scans yet
  - `.get_scan_count() -> int`: number of completed scans
- [ ] Create `tests/test_findings_db.py`:
  - Test `save_findings` creates parquet file and updates index
  - Test `load_latest` returns correct top-N sorted by score
  - Test `load_pair_history` finds pair in both orderings (a,b) and (b,a)
  - Test `was_seen_before` returns True after saving, False before
  - Test `load_all_scans` reflects all saved scans
  - Test empty DB (no scans yet) returns sensible defaults (`[]`, empty DataFrame)
  - Test directory auto-creation if `base_path` doesn't exist

## Key Files

- `src/correlation_engine/store/__init__.py` — create: empty init
- `src/correlation_engine/store/findings_db.py` — create: FindingsDatabase class
- `tests/test_findings_db.py` — create: unit tests (use `tmp_path` pytest fixture for isolation)

## Acceptance Criteria

- Save + load round-trip preserves all Finding fields without data loss
- `load_pair_history` finds pair regardless of which series is A vs B
- `was_seen_before` correctly returns False for a brand-new pair and True after it has been saved
- `load_latest(n=10)` returns at most 10 findings even if scan has 500
- All tests pass using a temporary directory (no side effects on `data/findings/`)
- Index file is valid JSON and human-readable

## Notes

- Use `pd.DataFrame.to_parquet()` and `pd.read_parquet()` — already available via pandas (pyarrow or fastparquet as engine; pyarrow is preferred and likely already installed transitively)
- `index.json` format: `{"scans": [...]}` array of scan metadata dicts; append new entry on each save
- Pair canonicalization: always store as `(min(a,b), max(a,b))` alphabetically to ensure consistent lookup — apply in both `save_findings` (normalize before write) and `was_seen_before` / `load_pair_history`
- Keep `FindingsDatabase` stateless between calls — reads index fresh each time (file is small, fast to read; avoids stale cache)
- This module has zero dependency on Streamlit or any UI — pure data layer
