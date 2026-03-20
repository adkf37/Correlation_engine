# Correlation Engine

Reproducible Python toolkit for computing, visualizing, and exploring correlations (including lead/lag relationships) among any number of time-series datasets.

## Features

- **Multi-source data ingestion** — CSV, FRED API, Yahoo Finance with Parquet caching
- **Preprocessing pipeline** — frequency alignment, missing data handling, stationarity transforms
- **Correlation analysis** — Pearson/Spearman/Kendall matrices with hierarchical clustering
- **Lead/lag analysis** — cross-correlation functions, optimal lag detection
- **Rolling windows** — time-varying correlation tracking
- **Statistical robustness** — bootstrap confidence intervals, multiple testing correction (FDR/Bonferroni)
- **Granger causality** — pairwise and matrix-wide causality testing
- **DCC-GARCH** — dynamic conditional correlation modeling
- **Network graphs** — interactive correlation network visualization
- **Streamlit dashboard** — interactive multi-page UI for exploration

## Quick Start

```bash
# Install uv (if you don't have it)
# https://docs.astral.sh/uv/getting-started/installation/

# Clone and install
uv sync

# Run the dashboard
uv run streamlit run dashboard/app.py
```

## Configuration

Copy `.env.example` to `.env` and add your FRED API key:

```bash
cp .env.example .env
# Edit .env and set FRED_API_KEY=your_actual_key
```

Yahoo Finance requires no API key.

## Project Structure

```
src/correlation_engine/     Core library (UI-independent)
├── ingest/                 Data loaders (CSV, FRED, Yahoo) + cache
├── preprocessing/          Alignment, missing data, stationarity
├── analysis/               Correlation, lag, rolling, significance, Granger, DCC-GARCH
└── viz/                    Plotly/seaborn visualizations

dashboard/                  Streamlit multi-page app
tests/                      pytest suite
data/sample/                Demo datasets
notebooks/                  Jupyter exploration
```

## Development

```bash
# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest tests/

# Lint
uv run ruff check .
```