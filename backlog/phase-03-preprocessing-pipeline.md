# Phase 03: Preprocessing Pipeline

**Status:** Complete
**Depends on:** Phase 02
**Estimated scope:** M

## Objective

Build a configurable preprocessing pipeline that takes raw, messy time-series DataFrames (mixed frequencies, missing values, non-stationary) and produces clean, aligned, analysis-ready data. This is the critical bridge between raw ingestion and all analysis modules.

## Tasks

- [ ] Implement frequency alignment in `src/correlation_engine/preprocessing/align.py`
  - `align_frequencies(df, target_freq='M', method='last') -> pd.DataFrame`
  - Support target frequencies: `'D'`, `'W'`, `'M'`, `'Q'`, `'Y'`
  - Aggregation methods: `'last'`, `'mean'`, `'sum'`, `'first'`
  - For upsampling (e.g., quarterly → monthly): use interpolation or forward-fill
  - Auto-detect input frequency using `pd.infer_freq()` when possible
- [ ] Implement missing data handling in `src/correlation_engine/preprocessing/missing.py`
  - `handle_missing(df, strategy='interpolate', **kwargs) -> pd.DataFrame`
  - Strategies: `'ffill'`, `'bfill'`, `'interpolate'` (linear), `'drop_rows'`, `'drop_cols'`
  - `report_missing(df) -> pd.DataFrame` — returns percentage missing per column plus total rows
  - Configurable threshold: drop columns with >X% missing
- [ ] Implement stationarity tools in `src/correlation_engine/preprocessing/transform.py`
  - `check_stationarity(series, significance=0.05) -> dict` — runs ADF test via `statsmodels.tsa.stattools.adfuller()`, returns `{'statistic', 'pvalue', 'is_stationary', 'critical_values'}`
  - `check_stationarity_all(df, significance=0.05) -> pd.DataFrame` — run ADF on every column, return summary table
  - `make_stationary(df, method='diff') -> tuple[pd.DataFrame, dict]` — apply transformation, return (transformed_df, report of what was done per column)
  - Methods: `'diff'` (first difference), `'log_diff'` (log returns), `'detrend'` (linear detrending via `scipy.signal.detrend`)
- [ ] Build `PreprocessingPipeline` class in `src/correlation_engine/preprocessing/pipeline.py`
  - Accepts ordered list of steps: `[('align', {'target_freq': 'M'}), ('missing', {'strategy': 'interpolate'}), ('transform', {'method': 'diff'})]`
  - `pipeline.run(df) -> pd.DataFrame` executes steps in order
  - `pipeline.report() -> dict` returns summary of what each step did (rows before/after, columns dropped, stationarity results)
  - Steps are optional — user can skip any
- [ ] Write unit tests in `tests/test_preprocessing.py`
  - Test alignment: daily data → monthly produces correct number of rows
  - Test missing: DataFrame with NaNs → fully filled output
  - Test stationarity: random walk → `is_stationary=False`; differenced → `is_stationary=True`
  - Test pipeline: raw messy data → clean output in one call
  - Test edge cases: all-NaN columns, single-row DataFrames, constant series

## Key Files

- `src/correlation_engine/preprocessing/__init__.py` — re-export key functions
- `src/correlation_engine/preprocessing/align.py` — `align_frequencies()`
- `src/correlation_engine/preprocessing/missing.py` — `handle_missing()`, `report_missing()`
- `src/correlation_engine/preprocessing/transform.py` — `check_stationarity()`, `check_stationarity_all()`, `make_stationary()`
- `src/correlation_engine/preprocessing/pipeline.py` — `PreprocessingPipeline`
- `tests/test_preprocessing.py` — preprocessing test suite

## Acceptance Criteria

- A raw DataFrame with daily data, 15% missing values, and non-stationary series → `PreprocessingPipeline` produces a monthly, fully-filled, stationary DataFrame
- `check_stationarity_all()` correctly identifies which series need differencing
- `report_missing()` accurately reports NaN percentages
- `pipeline.report()` returns a human-readable summary of all transformations applied
- `pytest tests/test_preprocessing.py` passes

## Notes

- Stationarity transforms are **critical** for Phase 5 (Granger causality) and Phase 4 (CCF). Non-stationary inputs produce spurious correlations
- The pipeline should **not** silently drop data. All transformations should be reported via `pipeline.report()`
- `make_stationary()` should return the transformation metadata so it can be reversed or displayed in the dashboard later
- For differencing, the first row becomes NaN — handle this by dropping the leading NaN row(s) after differencing
