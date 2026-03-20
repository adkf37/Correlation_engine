# Phase 05: Statistical Robustness

**Status:** Complete
**Depends on:** Phase 04
**Estimated scope:** M

## Objective

Add statistical rigor to the correlation results — bootstrap confidence intervals, multiple testing corrections, and rolling window analysis to reveal time-varying relationships. This transforms raw correlation numbers into defensible, publishable findings.

## Tasks

### Bootstrap Confidence Intervals
- [ ] Implement bootstrap CI in `src/correlation_engine/analysis/significance.py`
  - `bootstrap_correlation_ci(x, y, method='pearson', n_boot=1000, alpha=0.05, seed=None) -> dict`
  - Returns `{'point_estimate': float, 'ci_lower': float, 'ci_upper': float, 'se': float}`
  - Use block bootstrap (not iid) for time-series data — resample contiguous blocks to preserve autocorrelation
  - Block length heuristic: `int(n ** (1/3))` where n = series length
- [ ] Implement matrix-wide bootstrap
  - `bootstrap_correlation_matrix_ci(df, method='pearson', n_boot=1000, alpha=0.05) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
  - Returns `(point_estimates, ci_lower, ci_upper)` — three NxN DataFrames

### Multiple Testing Correction
- [ ] Implement p-value adjustment in `src/correlation_engine/analysis/significance.py`
  - `compute_pvalue_matrix(df, method='pearson') -> pd.DataFrame` — p-value for each pairwise correlation using `scipy.stats.pearsonr` / `spearmanr` / `kendalltau`
  - `adjust_pvalues(pvalue_matrix, method='fdr_bh') -> pd.DataFrame` — adjust using `statsmodels.stats.multitest.multipletests`
  - Support methods: `'bonferroni'`, `'fdr_bh'` (Benjamini-Hochberg), `'fdr_by'`, `'holm'`
- [ ] Add significance flag matrix
  - `flag_significant(adjusted_pvalues, alpha=0.05) -> pd.DataFrame` — boolean matrix of significant correlations

### Rolling Window Correlations
- [ ] Implement rolling correlation in `src/correlation_engine/analysis/rolling.py`
  - `compute_rolling_correlation(x, y, window=60, method='pearson') -> pd.Series` — rolling pairwise correlation
  - `compute_rolling_matrix(df, window=60, method='pearson') -> dict[tuple[str,str], pd.Series]` — all pairs
  - Handle edge: return NaN for periods with insufficient data (< window size)
- [ ] Implement window sensitivity analysis
  - `window_sensitivity(x, y, windows=[30, 60, 90, 120, 180], method='pearson') -> pd.DataFrame` — compute rolling correlation at multiple window sizes, return DataFrame for comparison

### Visualizations
- [ ] Implement rolling correlation plots in `src/correlation_engine/viz/rolling_plots.py`
  - `plot_rolling_correlation(rolling_series, pair_label=None) -> Figure` — time-series line plot
  - `plot_rolling_multi(rolling_dict, pairs=None) -> Figure` — overlay multiple pairs
  - `plot_window_sensitivity(sensitivity_df) -> Figure` — one line per window size
- [ ] Implement significance overlay for heatmap
  - `plot_significance_heatmap(corr_matrix, pvalue_matrix, alpha=0.05) -> Figure` — heatmap with non-significant cells crossed out or dimmed

### Tests
- [ ] Write tests in `tests/test_significance.py`
  - Bootstrap CI on uncorrelated noise → CI should contain 0
  - Bootstrap CI on perfectly correlated data → CI should be tight around 1.0
  - P-value matrix: known correlated pair → p < 0.05; uncorrelated → p > 0.05
  - FDR correction: adjusted p-values >= raw p-values
  - Rolling correlation on stationary correlated pair → roughly constant series
  - Rolling correlation on regime-switching data → visible shift
  - Window sensitivity: larger windows → smoother rolling estimates

## Key Files

- `src/correlation_engine/analysis/significance.py` — `bootstrap_correlation_ci()`, `bootstrap_correlation_matrix_ci()`, `compute_pvalue_matrix()`, `adjust_pvalues()`, `flag_significant()`
- `src/correlation_engine/analysis/rolling.py` — `compute_rolling_correlation()`, `compute_rolling_matrix()`, `window_sensitivity()`
- `src/correlation_engine/viz/rolling_plots.py` — `plot_rolling_correlation()`, `plot_rolling_multi()`, `plot_window_sensitivity()`
- `tests/test_significance.py` — robustness test suite

## Acceptance Criteria

- Bootstrap CI on 1000 uncorrelated noise samples contains 0 in >90% of runs
- `adjust_pvalues()` with Bonferroni produces values = raw_p * n_tests (capped at 1.0)
- Rolling correlation on synthetic stationary data has standard deviation < 0.1
- Window sensitivity plot shows convergence for large windows
- Significance heatmap visually distinguishes significant from non-significant pairs
- All tests pass

## Notes

- Block bootstrap is important for time-series — iid bootstrap breaks temporal structure and produces anti-conservative (too narrow) confidence intervals
- For large datasets (>20 series), the bootstrap on the full matrix is expensive. Consider parallelizing with `concurrent.futures` or adding a progress callback
- The significance overlay on the heatmap is one of the most useful outputs — it prevents users from over-interpreting noise
- Rolling windows are the simplest way to show time-varying relationships before jumping to DCC-GARCH in Phase 6
