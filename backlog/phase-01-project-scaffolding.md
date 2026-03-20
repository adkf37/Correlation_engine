# Phase 01: Project Scaffolding

**Status:** Not Started
**Depends on:** None
**Estimated scope:** S

## Objective

Set up the uv-managed Python project with the full package directory structure, all dependencies pinned, and a working importable package — so every subsequent phase has a stable foundation to build on.

## Tasks

- [ ] Run `uv init` to generate `pyproject.toml`
- [ ] Configure `pyproject.toml`:
  - Project name: `correlation-engine`
  - Python version: `>=3.11`
  - Core dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`, `arch`, `plotly`, `seaborn`, `matplotlib`, `streamlit`, `fredapi`, `yfinance`, `networkx`, `python-dotenv`
  - Dev dependencies: `pytest`, `ruff`, `mypy`
  - Package source: `src` layout
- [ ] Create full directory tree:
  ```
  src/correlation_engine/__init__.py
  src/correlation_engine/ingest/__init__.py
  src/correlation_engine/preprocessing/__init__.py
  src/correlation_engine/analysis/__init__.py
  src/correlation_engine/viz/__init__.py
  dashboard/app.py          (placeholder)
  dashboard/pages/.gitkeep
  tests/__init__.py
  data/sample/.gitkeep
  notebooks/.gitkeep
  ```
- [ ] Create `.env.example` with `FRED_API_KEY=your_key_here`
- [ ] Create `.gitignore` (Python defaults + `.env`, `__pycache__`, `.venv`, `*.parquet` cache files)
- [ ] Create a minimal `README.md` with project description and setup instructions
- [ ] Run `uv sync` and verify it completes without errors
- [ ] Verify `python -c "import correlation_engine"` succeeds

## Key Files

- `pyproject.toml` — project metadata and all dependencies
- `src/correlation_engine/__init__.py` — package root
- `.env.example` — FRED API key template
- `.gitignore` — ignore patterns
- `README.md` — project overview and quickstart

## Acceptance Criteria

- `uv sync` installs all dependencies without errors
- `python -c "import correlation_engine"` runs successfully
- All directories exist with proper `__init__.py` files
- `.env.example` documents required environment variables

## Notes

- Use `src` layout (package under `src/correlation_engine/`) to avoid import ambiguity
- The dashboard placeholder in `dashboard/app.py` can be a simple `st.title("Correlation Engine")` — it gets built out in Phase 7
- Pin minimum versions for critical deps: `arch>=6.0`, `statsmodels>=0.14`, `streamlit>=1.30`
