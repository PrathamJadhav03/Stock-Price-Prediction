---
phase: 1
plan: 1
wave: 1
depends_on: []
files_modified:
  - pyproject.toml
  - .python-version
  - src/__init__.py
  - src/data/__init__.py
  - src/data/ingestion.py
  - tests/__init__.py
  - tests/test_ingestion.py
autonomous: true
gap_closure: false

must_haves:
  truths:
    - "Hermetic environment resolves and locks dependencies cleanly via uv sync"
    - "Data ingestion transparently checks local disk cache before network download"
    - "Data ingestion falls back to historical Data/*.csv records if network is unavailable or ticker matches"
    - "Ingested datasets have monotonic DatetimeIndex and standard normalized OHLCV column schema"
  artifacts:
    - "pyproject.toml exists with complete core and dev dependency definitions"
    - "uv.lock exists"
    - "src/data/ingestion.py exists and exports fetch_stock_data"
    - "tests/test_ingestion.py exists and passes via uv run pytest"
---

# Plan 1.1: `uv` Environment Setup & Resilient Data Ingestion

## Objective
Establish the hermetic Python development environment using `uv` with pinned dependencies, and build a resilient data ingestion module with local disk caching and automatic offline fallback to historical `Data/*.csv` records.

## Context
Load these files for context:
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/REQUIREMENTS.md
- .gsd/DECISIONS.md
- Data/GOOGL_2006-01-01_to_2018-01-01.csv

## Tasks

<task type="auto">
  <name>Configure pyproject.toml and lock dependencies with uv</name>
  <files>
    pyproject.toml
    .python-version
    src/__init__.py
    src/data/__init__.py
    tests/__init__.py
  </files>
  <action>
    Configure a standard Python package managed strictly by `uv`.
    
    Steps:
    1. Create `.python-version` specifying `3.11` (or `>=3.10,<3.13`).
    2. Create `pyproject.toml` with:
       - `[project]` metadata: `name = "stock-price"`, `version = "0.1.0"`, `requires-python = ">=3.10,<3.13"`.
       - Core dependencies:
         - `torch>=2.0.0`
         - `numpy>=1.24.0`
         - `pandas>=2.0.0`
         - `yfinance>=0.2.36`
         - `scikit-learn>=1.3.0`
         - `joblib>=1.3.0`
         - `pydantic>=2.5.0`
         - `pyyaml>=6.0`
         - `requests>=2.31.0`
       - Dev / Test dependencies in `[project.optional-dependencies]` or `[dependency-groups]`:
         - `pytest>=8.0.0`
         - `pytest-mock>=3.12.0`
         - `ruff>=0.3.0`
       - Build system: standard `hatchling` (or `setuptools`).
       - `[tool.pytest.ini_options]` configured with `testpaths = ["tests"]` and `pythonpath = ["."]`.
    3. Initialize package directories: `src/__init__.py`, `src/data/__init__.py`, and `tests/__init__.py`.
    4. Run `uv lock` and `uv sync` to produce a deterministic `uv.lock`.
    
    AVOID: Using `pip install`, `pip-tools`, or activating virtual environments manually.
    USE: Exclusive `uv` tooling (`uv sync`, `uv run`).
  </action>
  <verify>
    uv sync && uv run python -c "import torch, pandas, numpy, yfinance, sklearn, joblib; print('Dependencies successfully verified under uv environment')"
  </verify>
  <done>
    `uv.lock` is generated and all core dependencies import cleanly via `uv run python`.
  </done>
</task>

<task type="auto">
  <name>Implement resilient data ingestion with disk caching and historical CSV fallback</name>
  <files>
    src/data/ingestion.py
    tests/test_ingestion.py
  </files>
  <action>
    Build the data ingestion module capable of fetching live market bars with local disk caching and graceful offline fallback to `Data/*.csv`.
    
    Steps:
    1. In `src/data/ingestion.py`, implement `fetch_stock_data`:
       - Signature: `fetch_stock_data(symbol: str, start_date: str = "2006-01-01", end_date: str = "2018-01-01", cache_dir: str = "data/raw", fallback_dir: str = "Data", force_refresh: bool = False) -> pd.DataFrame`
       - Resolution logic:
         a. If not `force_refresh` and `{cache_dir}/{symbol}.csv` exists: load and parse dates from cache.
         b. Try downloading via `yfinance.download(symbol, start=start_date, end=end_date, progress=False)`. If successful and non-empty, persist to `{cache_dir}/{symbol}.csv` and return.
         c. If `yfinance` download fails or returns empty (offline, network error, or symbol discrepancy), search `{fallback_dir}` for candidate files matching `{symbol}_*.csv` or `{symbol}.csv` (e.g., `Data/GOOGL_2006-01-01_to_2018-01-01.csv` for `GOOGL`).
         d. If candidate file found, load and filter by date range if applicable.
         e. If no cache, download, or fallback exists, raise descriptive `FileNotFoundError`.
       - Schema normalization:
         - Standardize columns to `["Open", "High", "Low", "Close", "Volume"]`.
         - Ensure index is a `DatetimeIndex` named `Date`, monotonically increasing, with duplicates removed.
         - Cast numerical columns to `float64` (`Volume` to `int64` or `float64`).
    2. In `tests/test_ingestion.py`, implement automated tests:
       - Fallback test: Request `GOOGL` with mocked failed network call, verify data is correctly loaded from `Data/GOOGL_2006-01-01_to_2018-01-01.csv` with expected row count and schema.
       - Cache test: Verify that once cached in `data/raw/`, subsequent requests read from cache without network invocation.
       - Schema test: Verify column names, datatypes, and monotonic DatetimeIndex.
       - Error handling test: Request invalid ticker with no fallback, assert `FileNotFoundError` is raised.
    
    AVOID: Relying on live network availability during automated test execution.
    USE: Mocking for network requests in tests and fallback to existing repository CSV files.
  </action>
  <verify>
    uv run pytest tests/test_ingestion.py -v
  </verify>
  <done>
    `tests/test_ingestion.py` passes 100% with tests verifying offline fallback, disk caching, and schema normalization.
  </done>
</task>

## Must-Haves
After all tasks complete, verify:
- [ ] `uv sync` resolves without conflicts and produces `uv.lock`.
- [ ] `src/data/ingestion.py` successfully loads historical data from `Data/*.csv` when offline.
- [ ] Ingested DataFrame contains standardized `Open`, `High`, `Low`, `Close`, `Volume` columns indexed by `Date`.
- [ ] All ingestion tests pass via `uv run pytest tests/test_ingestion.py`.

## Success Criteria
- [ ] All tasks verified passing.
- [ ] Must-haves confirmed.
- [ ] No regressions in test suite.

