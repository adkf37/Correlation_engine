# Phase 08: Universe & Watchlist

**Status:** Not Started
**Depends on:** Phase 06 (ingest + preprocessing modules)
**Estimated scope:** M

## Objective

Define the full data universe (FRED macroeconomic series + Yahoo Finance ETFs/indices) in a declarative config file, and build a `Watchlist` class that fetches, preprocesses, and caches all series into a scan-ready `dict[str, pd.Series]`.

## Tasks

- [ ] Create `config/universe.yaml` with full series list:
  - FRED series (~80), organized by category: GDP, inflation (CPI, PCE, breakevens), rates (FEDFUNDS, DGS2, DGS10, DGS30, T10Y2Y), employment (UNRATE, PAYEMS, ICSA), credit (BAMLH0A0HYM2, BAMLC0A0CM), housing (CSUSHPISA, HOUST), money (M2SL), financial conditions (NFCI, STLFSI2), commodities (DCOILWTICO, PCOPPUSDM), FX (DTWEXBGS)
  - Yahoo tickers (~20): SPY, QQQ, IWM, GLD, SLV, GDX, TLT, IEF, SHY, HYG, LQD, VNQ, XLE, XLF, XLK, XLV, EEM, EFA, UUP, ^VIX
  - Each entry has: `id`, `name`, `category`, `source` (fred|yahoo), `frequency` hint
- [ ] Create `src/correlation_engine/discovery/__init__.py`
- [ ] Create `src/correlation_engine/discovery/watchlist.py`:
  - `Watchlist(config_path, cache_ttl_hours=24)` class
  - `.load(start_date, end_date)` → `dict[str, pd.Series]`
  - Internally: FredLoader + YahooLoader (existing) → DataCache (existing) → PreprocessingPipeline: align monthly (`target_freq='M'`), ffill missing, `make_stationary(method='log_diff')` on non-stationary series
  - `.metadata` property → DataFrame with series id, name, category, source, obs_count, date_range
  - Skips series that fail to load (logs warning, continues)
  - Reports how many series loaded successfully vs. skipped
- [ ] Create `tests/test_watchlist.py`:
  - Mock FredLoader.load and YahooLoader.load with synthetic DataFrames
  - Assert output keys match config entries that succeeded
  - Assert all output series have DatetimeIndex at monthly frequency
  - Assert no NaNs in output (preprocessing applied)
  - Test partial failure (one loader raises) → missing series skipped, rest returned

## Key Files

- `config/universe.yaml` — create: full series universe definition
- `src/correlation_engine/discovery/__init__.py` — create: empty package init
- `src/correlation_engine/discovery/watchlist.py` — create: Watchlist class
- `tests/test_watchlist.py` — create: unit tests with mocked loaders

## Acceptance Criteria

- `Watchlist.load()` returns a dict with at least 80% of configured series when FRED_API_KEY is valid
- All series in the output dict are `pd.Series` with `DatetimeIndex` at monthly frequency (`'ME'` or `'MS'`)
- No NaN values in output (preprocessing applied before return)
- If a series fails to load, it is skipped silently with a warning log; does not crash the whole load
- `tests/test_watchlist.py` all pass with mocked loaders
- Watchlist load is fully cached (second call hits cache, no HTTP requests)

## Notes

- Reuse `FredLoader`, `YahooLoader`, `DataCache`, and `PreprocessingPipeline` exactly as-is — no modifications to existing src
- The FRED `request_delay` on FredLoader should be set to 0.5s to avoid rate limiting when loading 80 series on first run
- Yahoo `^VIX` needs mapping: the `^` prefix is valid for yfinance but may need URL encoding in some contexts — test explicitly
- Monthly alignment is the universal frequency for the scanner; daily series are downsampled, lower-frequency series (quarterly) are upsampled with ffill
- Start date recommended default: 10 years back from scan date; configurable in `config/scan_config.yaml` (Phase 12)
