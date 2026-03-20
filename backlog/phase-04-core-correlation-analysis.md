 # Phase 04: Core Correlation Analysis

**Status:** Complete
**Depends on:** Phase 03
**Estimated scope:** M

## Objective

Implement the foundational correlation computations — full correlation matrix with clustering, cross-correlation at multiple lags, and a lead/lag summary matrix — plus interactive and static visualizations for each. This is the heart of the engine.

## Tasks

### Correlation Matrix
- [ ] Implement `compute_correlation_matrix()` in `src/correlation_engine/analysis/correlation.py`
  - `compute_correlation_matrix(df, method='pearson') -> pd.DataFrame`
  - Support methods: `'pearson'`, `'spearman'`, `'kendall'`
  - Compute via `df.corr(method=method)`
- [ ] Implement hierarchical clustering for matrix reordering
  - `cluster_correlation_matrix(corr_matrix) -> tuple[pd.DataFrame, linkage_matrix]`
  - Use `scipy.cluster.hierarchy.linkage(distance_matrix, method='ward')` where distance = `1 - abs(corr)`
  - Return reordered matrix + linkage for optional dendrogram

### Cross-Correlation / Lag Analysis
- [ ] Implement cross-correlation in `src/correlation_engine/analysis/lag.py`
  - `compute_cross_correlation(x, y, max_lag=24) -> pd.DataFrame` — returns DataFrame with columns `['lag', 'correlation']`
  - Use `statsmodels.tsa.stattools.ccf` under the hood
  - Support both positive and negative lags (X leads Y vs. Y leads X)
- [ ] Implement lead/lag summary matrix
  - `compute_lead_lag_matrix(df, max_lag=12) -> pd.DataFrame` — for each pair, find the lag with maximum absolute correlation
  - Return DataFrame with multi-level info: `optimal_lag` and `max_correlation` for each pair
  - This is the "at-a-glance" summary of which series lead/lag others

### Visualizations
- [ ] Implement heatmap in `src/correlation_engine/viz/heatmap.py`
  - `plot_correlation_heatmap(corr_matrix, interactive=True) -> Figure`
  - Interactive mode: Plotly heatmap with hover showing exact values, clustered ordering
  - Static mode: seaborn `clustermap` for PNG/PDF export
  - Color scale: diverging (blue = negative, red = positive), centered at 0
- [ ] Implement lag plots in `src/correlation_engine/viz/lag_plots.py`
  - `plot_ccf(ccf_df, title=None) -> Figure` — bar chart of correlation vs. lag for one pair
  - `plot_lead_lag_matrix(lead_lag_df) -> Figure` — heatmap of optimal lags (color = lag value, annotation = correlation)
  - Both in Plotly for interactivity

### Tests
- [ ] Write tests in `tests/test_correlation.py`
  - Synthetic test: create series Y = X shifted by 3 periods + noise → verify `compute_cross_correlation` peaks at lag=3
  - Correlation matrix on known data (e.g., perfectly correlated pair → 1.0, uncorrelated → ~0)
  - Clustering: verify reordered matrix groups correlated series adjacently
  - Lead/lag matrix: verify it picks the correct optimal lag for synthetic pairs
  - Visualization smoke tests: verify functions return valid Plotly/matplotlib Figure objects

## Key Files

- `src/correlation_engine/analysis/__init__.py` — re-export key functions
- `src/correlation_engine/analysis/correlation.py` — `compute_correlation_matrix()`, `cluster_correlation_matrix()`
- `src/correlation_engine/analysis/lag.py` — `compute_cross_correlation()`, `compute_lead_lag_matrix()`
- `src/correlation_engine/viz/heatmap.py` — `plot_correlation_heatmap()`
- `src/correlation_engine/viz/lag_plots.py` — `plot_ccf()`, `plot_lead_lag_matrix()`
- `tests/test_correlation.py` — correlation analysis test suite

## Acceptance Criteria

- `compute_correlation_matrix()` returns correct NxN matrix matching `pandas.DataFrame.corr()` for Pearson, Spearman, and Kendall
- `compute_cross_correlation()` on synthetic lagged data correctly identifies the lag with peak correlation
- `compute_lead_lag_matrix()` produces a summary with correct optimal lags for all pairs
- Plotly heatmap renders with hover values and clustered ordering
- CCF bar chart clearly shows correlation peaks at correct lags
- All test cases pass

## Notes

- All analysis functions take plain `pd.DataFrame` and return `pd.DataFrame` or Plotly `Figure` — **no Streamlit imports** in this layer
- Clustering is optional in the heatmap — provide a `clustered=True` parameter with sensible default
- For large numbers of series (>50), CCF computation on all pairs may be slow. Consider adding a progress callback parameter that the dashboard can hook into later
- The lead/lag matrix is one of the most valuable outputs — it tells you at a glance "CPI leads unemployment by 6 months" or "S&P leads MSCI by 1 day"
