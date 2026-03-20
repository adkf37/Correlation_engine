# Phase 06: Advanced Analysis

**Status:** Complete
**Depends on:** Phase 05
**Estimated scope:** L

## Objective

Add Granger causality testing (pairwise and matrix), DCC-GARCH time-varying conditional correlations, and a network graph visualization. These are the power-user features that elevate this from a correlation calculator to a genuine analysis platform.

## Tasks

### Granger Causality
- [ ] Implement Granger causality in `src/correlation_engine/analysis/granger.py`
  - `granger_causality_test(df, target, predictor, max_lag=12) -> dict`
    - Wraps `statsmodels.tsa.stattools.grangercausalitytests`
    - Returns `{'optimal_lag': int, 'f_stat': float, 'p_value': float, 'reject_null': bool}` at the best lag
    - Pre-check: run ADF test on both series. If non-stationary, auto-difference and emit a warning (don't silently transform)
  - `granger_causality_matrix(df, max_lag=12, significance=0.05) -> pd.DataFrame`
    - Pairwise test for all (target, predictor) combinations
    - Return DataFrame with p-values; rows = targets, columns = predictors
    - Diagonal = NaN (no self-causality)
  - `granger_summary(granger_matrix, alpha=0.05) -> list[dict]`
    - Extract significant causal pairs: `[{'predictor': 'CPI', 'target': 'unemployment', 'lag': 6, 'p_value': 0.003}, ...]`
    - Sorted by p-value

### DCC-GARCH
- [ ] Implement DCC-GARCH in `src/correlation_engine/analysis/dcc_garch.py`
  - `fit_dcc_garch(df, p=1, q=1) -> DccResult`
    - **Stage 1:** Fit `arch.univariate.ConstantMean` + `arch.univariate.GARCH(p=1, q=1)` to each series individually
    - **Stage 2:** Extract standardized residuals → fit `arch.multivariate.DCC` model
    - Return custom `DccResult` dataclass containing:
      - `conditional_correlations: dict[tuple[str,str], pd.Series]` — time-varying correlation per pair
      - `conditional_volatilities: pd.DataFrame` — per-series volatility over time
      - `model_params: dict` — estimated DCC parameters (a, b)
      - `convergence_info: dict` — solver status, log-likelihood
  - Input validation:
    - Warn if `df` has more than 10 columns (computation scales O(n²) and convergence degrades)
    - Require minimum 250 observations for reliable estimation
    - Data should be returns/log-returns (not levels); auto-detect and warn if levels
  - Use `solver='L-BFGS-B'` for better convergence (default solver can be slow)

### Network Visualization
- [ ] Implement correlation network in `src/correlation_engine/viz/network.py`
  - `build_correlation_network(corr_matrix, threshold=0.5) -> nx.Graph`
    - Nodes = series names, edges = correlations above threshold (absolute value)
    - Edge weight = correlation value; edge color = sign (positive/negative)
  - `plot_correlation_network(graph, layout='spring') -> Figure`
    - Plotly interactive network graph
    - Node size proportional to number of strong connections (degree)
    - Edge thickness proportional to correlation strength
    - Layout options: `'spring'`, `'circular'`, `'kamada_kawai'`
  - `plot_correlation_network_with_slider(corr_matrix) -> Figure`
    - Include threshold slider so user can interactively adjust which edges appear

### DCC-GARCH Visualization
- [ ] Add DCC-GARCH plots in `src/correlation_engine/viz/dcc_plots.py`
  - `plot_conditional_correlation(dcc_result, pair) -> Figure` — single pair's time-varying correlation
  - `plot_conditional_correlations_grid(dcc_result, pairs=None) -> Figure` — subplot grid of multiple pairs
  - `plot_conditional_volatility(dcc_result) -> Figure` — volatility over time per series

### Tests
- [ ] Write tests in `tests/test_granger.py`
  - Synthetic: create Y = 0.5*X_lagged_3 + noise → Granger test should reject null at lag 3
  - Non-stationary input → function warns and auto-differences
  - Granger matrix on 4 series → correct shape (4x4), diagonal is NaN
  - `granger_summary()` returns only significant pairs
- [ ] Write tests in `tests/test_dcc_garch.py`
  - Smoke test: fit DCC-GARCH on 3 synthetic volatile series → `DccResult` has expected structure
  - Conditional correlations are bounded in [-1, 1]
  - Warning emitted for >10 series
  - Warning emitted for <250 observations
- [ ] Write tests in `tests/test_network.py`
  - Network from identity matrix with threshold=0.5 → no edges (self-loops excluded)
  - Network from matrix with known structure → correct number of edges
  - Plot functions return valid Plotly Figure objects

## Key Files

- `src/correlation_engine/analysis/granger.py` — `granger_causality_test()`, `granger_causality_matrix()`, `granger_summary()`
- `src/correlation_engine/analysis/dcc_garch.py` — `fit_dcc_garch()`, `DccResult` dataclass
- `src/correlation_engine/viz/network.py` — `build_correlation_network()`, `plot_correlation_network()`
- `src/correlation_engine/viz/dcc_plots.py` — `plot_conditional_correlation()`, `plot_conditional_correlations_grid()`
- `tests/test_granger.py` — Granger test suite
- `tests/test_dcc_garch.py` — DCC-GARCH test suite
- `tests/test_network.py` — network viz test suite

## Acceptance Criteria

- Granger test on synthetic causal data correctly rejects null hypothesis
- Granger matrix is asymmetric (X→Y ≠ Y→X) when ground truth is directional
- DCC-GARCH fits successfully on 3-5 financial return series, conditional correlations vary over time
- Network graph correctly shows/hides edges as threshold changes
- All warnings fire correctly (>10 series, <250 obs, non-stationary input, level data)
- All tests pass

## Notes

- **DCC-GARCH is the heaviest computation in the entire project.** For the dashboard (Phase 7), wrap it in `@st.cache_data` and show a progress spinner
- Granger causality is **directional** — "X Granger-causes Y" ≠ "Y Granger-causes X". The matrix is intentionally asymmetric. Make this clear in docs and dashboard labels
- The `arch` library's multivariate DCC module has been evolving. If `arch.multivariate.DCC` isn't available in the installed version, fall back to manual two-stage estimation using univariate models + `numpy` DCC parameter estimation. Document this fallback
- Network graphs are most useful when there are 10+ series — for 3-4 series, the heatmap is sufficient. The dashboard should conditionally show the network page
