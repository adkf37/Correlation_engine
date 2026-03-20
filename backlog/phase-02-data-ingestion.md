# Phase 02: Data Ingestion

**Status:** Not Started
**Depends on:** Phase 01
**Estimated scope:** M

## Objective

Build a generic, extensible data loading layer with concrete implementations for CSV files, FRED API, and Yahoo Finance — plus a Parquet-based caching layer to avoid redundant API calls. This is the foundation for all downstream analysis.

## Tasks

- [ ] Define `BaseLoader` abstract base class in `src/correlation_engine/ingest/base.py`
  - Abstract method: `load(**kwargs) -> pd.DataFrame`
  - Contract: returns DataFrame with `DatetimeIndex` and one column per series
  - Include validation method `_validate_output(df)` that checks index type and dtype
- [ ] Implement `CsvLoader` in `src/correlation_engine/ingest/csv_loader.py`
  - Accept file path or list of paths
  - Auto-detect date column (heuristic: first column that parses as datetime)
  - Support multi-column CSVs (each column = one series)
  - Handle common formats: CSV, TSV, semicolon-delimited
- [ ] Implement `FredLoader` in `src/correlation_engine/ingest/fred.py`
  - Accept list of series IDs (e.g., `['GDP', 'CPIAUCSL', 'UNRATE']`) plus start/end dates
  - Load API key from environment variable via `python-dotenv`
  - Merge multiple series into single DataFrame (outer join on date)
  - Handle API rate limits with `time.sleep()` between requests
  - Raise clear error if API key is missing
- [ ] Implement `YahooLoader` in `src/correlation_engine/ingest/yahoo.py`
  - Accept list of tickers plus start/end dates
  - Use `yf.download()` batch API for efficiency
  - Default to Adjusted Close prices; allow selecting other columns (Open, High, Low, Volume)
  - Validate returned data shape (warn on empty tickers = delisted/invalid)
- [ ] Implement `DataCache` in `src/correlation_engine/ingest/cache.py`
  - Store fetched DataFrames as Parquet files in `data/cache/`
  - TTL-based staleness: accept `max_age` parameter (default 24 hours)
  - Cache key: hash of (loader type + parameters)
  - Methods: `get(key) -> DataFrame | None`, `put(key, df)`, `invalidate(key)`
- [ ] Create sample datasets in `data/sample/`
  - `macro_indicators.csv` — 3-4 synthetic macro-style series (monthly, 10+ years)
  - `equity_prices.csv` — 3-4 synthetic daily price series
  - Include some NaN values and mixed frequencies for testing preprocessing later
- [ ] Write unit tests in `tests/test_ingest.py`
  - Test CsvLoader on sample files
  - Test FredLoader and YahooLoader with mocked API responses (use `unittest.mock.patch`)
  - Test DataCache put/get/invalidate/TTL expiry
  - Test validation (wrong index type, missing columns → clear errors)

## Key Files

- `src/correlation_engine/ingest/__init__.py` — re-export all loaders
- `src/correlation_engine/ingest/base.py` — `BaseLoader` ABC
- `src/correlation_engine/ingest/csv_loader.py` — `CsvLoader`
- `src/correlation_engine/ingest/fred.py` — `FredLoader`
- `src/correlation_engine/ingest/yahoo.py` — `YahooLoader`
- `src/correlation_engine/ingest/cache.py` — `DataCache`
- `data/sample/macro_indicators.csv` — synthetic macro data
- `data/sample/equity_prices.csv` — synthetic price data
- `tests/test_ingest.py` — ingestion test suite

## Acceptance Criteria

- `CsvLoader` reads sample CSVs and returns valid DataFrames with DatetimeIndex
- `FredLoader` and `YahooLoader` work with mocked responses in tests
- `DataCache` correctly caches, retrieves, and expires data
- All loaders produce output conforming to the `BaseLoader` contract
- `pytest tests/test_ingest.py` passes

## Notes

- FRED API key is **not** required for tests (mocked). Real API testing is manual
- `YahooLoader` should handle HTTP 429 errors with a single retry + exponential backoff, but don't over-engineer retry logic
- Cache directory (`data/cache/`) should be added to `.gitignore`
- The `BaseLoader` interface should be simple enough that users can write custom loaders (e.g., for databases or other APIs) by subclassing
