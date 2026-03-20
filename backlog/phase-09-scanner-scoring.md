# Phase 09: Scanner & Scoring

**Status:** Not Started
**Depends on:** Phase 08 (Watchlist output shape and series dict)
**Estimated scope:** L

## Objective

Enumerate every pairwise combination of series in the watchlist, run the existing analysis modules against each pair, evaluate 6 interestingness criteria, and produce a ranked list of `Finding` records ready to be persisted.

## Tasks

- [ ] Create `src/correlation_engine/discovery/findings.py`:
  - `Finding` dataclass with all fields (see schema below)
  - `to_dict()` method for Parquet serialization
  - `from_dict()` classmethod for deserialization
  - `trigger_types` stored as pipe-separated string in Parquet (`"high_correlation|granger_causality"`), restored as `list[str]` on load
- [ ] Create `src/correlation_engine/discovery/scoring.py`:
  - `FindingScorer(config: ScoringConfig)` class
  - `ScoringConfig` dataclass loaded from `config/scoring.yaml`:
    - `correlation_threshold: float = 0.7`
    - `zscore_threshold: float = 2.0`
    - `granger_p_threshold: float = 0.05`
    - `lag_correlation_threshold: float = 0.6`
    - `weights: dict[str, float]` (one per criterion, sum to 1.0)
  - `score(pair_stats: dict, is_new: bool) -> (list[str], float)` → (trigger_types, composite_score)
  - 6 criterion methods:
    1. `_high_correlation(r)` → 0-1 score; fires if `|r| >= threshold`
    2. `_newly_emerging(is_new)` → 1.0 if new pair, 0 otherwise
    3. `_regime_change(rolling_zscore)` → `min(|z| / 3, 1.0)` if `|z| >= zscore_threshold`
    4. `_granger_causality(p_value)` → `1.0 - p_value` if `p < threshold`
    5. `_anomalous_lag(lag, lag_r)` → scaled score if `lag != 0` and `|lag_r| >= threshold`
    6. `_rolling_divergence(rolling_zscore)` → same z-score logic as regime_change (separate trigger label)
- [ ] Create `src/correlation_engine/discovery/scanner.py`:
  - `DiscoveryScanner(config: ScanConfig)` class
  - `.scan(watchlist_dict: dict[str, pd.Series], db: FindingsDatabase) -> list[Finding]`
  - Iterates all `N*(N-1)/2` unique pairs
  - **Pre-filter**: skip pair if `|pearson_r| < min_r_for_granger` (default 0.3) before running Granger (performance optimization)
  - Per pair, computes:
    - `compute_correlation_matrix` (pearson) — extract scalar r
    - `compute_cross_correlation` → extract optimal lag + lag_correlation
    - `compute_rolling_correlation` (window=90d → ~3 months after monthly resampling, use window=`rolling_window` from config)
    - Rolling z-score: `(current_rolling_r - historical_mean) / historical_std` where historical uses full series window
    - Granger only if `|r| >= min_r_for_granger`
    - `db.was_seen_before(a, b)` → bool for `is_new`
  - Calls `FindingScorer.score()` → trigger_types + score
  - Only stores findings where `len(trigger_types) > 0` (at least one criterion fired) OR `score >= min_score_threshold`
  - Returns sorted list by `interestingness_score` descending
  - Progress callback: yields `(completed_pairs, total_pairs)` tuples via optional `on_progress` callable
- [ ] Create `tests/test_scanner.py`:
  - Synthetic watchlist: 5 perfectly correlated pairs, 5 with known lag, 5 uncorrelated
  - Assert high-corr pairs appear in findings
  - Assert lagged pairs trigger `anomalous_lag`
  - Assert uncorrelated pairs are filtered out (no trigger fires)
  - Assert findings sorted by score descending
- [ ] Create `tests/test_scoring.py`:
  - Test each of 6 criteria in isolation
  - Test composite score correctly weights criteria
  - Test pair below all thresholds → empty trigger_types, score = 0
  - Test pair hitting all criteria → score near 1.0

## Key Files

- `src/correlation_engine/discovery/findings.py` — create: Finding dataclass + serialization
- `src/correlation_engine/discovery/scoring.py` — create: FindingScorer with 6 criteria
- `src/correlation_engine/discovery/scanner.py` — create: DiscoveryScanner orchestrator
- `config/scoring.yaml` — create: scoring thresholds and weights
- `tests/test_scanner.py` — create: end-to-end scanner tests
- `tests/test_scoring.py` — create: unit tests per criterion

## Finding Schema

```
scan_id: str                    # UUID for the scan run
scanned_at: str                 # ISO 8601 timestamp
series_a: str                   # series ID (e.g. "UNRATE")
series_b: str                   # series ID (e.g. "SPY")
series_a_name: str              # human name from universe config
series_b_name: str              # human name from universe config
correlation: float              # Pearson r at lag=0
correlation_method: str         # always "pearson" for scanner
optimal_lag: int                # months; positive = A leads B
lag_correlation: float          # r at optimal_lag
granger_p_value: float | None   # None if pre-filter skipped Granger
granger_direction: str | None   # "a_causes_b" | "b_causes_a" | "bidirectional" | None
rolling_zscore: float           # z-score of 90d rolling r vs. full-window mean
regime_change_detected: bool    # |rolling_zscore| > threshold
trigger_types: list[str]        # which of 6 criteria fired
interestingness_score: float    # 0-1 composite weighted score
is_new: bool                    # pair not in last 3 scans
template_summary: str           # generated in Phase 11
llm_summary: str | None         # generated in Phase 11
lookback_days: int              # scan window used
frequency: str                  # "M" (monthly)
```

## Acceptance Criteria

- Scanner produces at least 1 finding for a watchlist of 5 series containing known correlated pairs
- Pre-filter reduces Granger calls to <50% of total pairs (verified via mock counting)
- `Finding.to_dict()` / `from_dict()` round-trip without data loss
- `trigger_types` correctly identifies each criterion independently
- `interestingness_score` is always in [0.0, 1.0]
- All tests in `test_scanner.py` and `test_scoring.py` pass

## Notes

- The rolling z-score window: use a 12-month rolling window for "current" and the full available history for baseline mean/std. This needs at least 36 months of data to be meaningful; warn and set zscore=0.0 if insufficient history.
- `compute_rolling_correlation` from `rolling.py` expects two pd.Series — extract individual series from the watchlist dict
- Granger is the bottleneck: ~100ms per pair on monthly data. At 4,950 pairs with pre-filter eliminating ~60%, expect ~2 minutes per full scan on modest hardware
- `scan_id` should be a UUID4 string generated at the start of each `.scan()` call
- `ScanConfig` and `ScoringConfig` dataclasses defined here, loaded from YAML in Phase 12
