---
phase: 1
plan: 2
wave: 2
depends_on:
  - 1-PLAN.md
files_modified:
  - src/features/__init__.py
  - src/features/returns.py
  - src/features/technical.py
  - tests/test_features.py
autonomous: true
gap_closure: false

must_haves:
  truths:
    - "Log returns r_t = ln(P_t / P_{t-1}) are calculated stationarily without lookahead"
    - "Price trajectory reconstruction inverts log returns with numerical error < 1e-5"
    - "Technical indicators (RSI, MACD, Bollinger Bands, ATR, SMA/EMA, Volatility) calculate strictly backward in time"
    - "Combined feature set drops initial burn-in window and contains 0 NaNs"
  artifacts:
    - "src/features/returns.py exists with compute_log_returns and reconstruct_price_trajectory"
    - "src/features/technical.py exists with comprehensive technical indicator suite"
    - "tests/test_features.py exists and passes via uv run pytest"
---

# Plan 1.2: Stationary Log Returns & Technical Indicator Engine

## Objective
Implement stationary log-return calculations with exact price trajectory reconstruction to address price drift non-stationarity, and construct a comprehensive, zero-lookahead technical indicator feature suite.

## Context
Load these files for context:
- .gsd/SPEC.md
- .gsd/ROADMAP.md
- .gsd/REQUIREMENTS.md
- .gsd/DECISIONS.md
- .gsd/phases/1/1-PLAN.md
- src/data/ingestion.py

## Tasks

<task type="auto">
  <name>Implement stationary log returns and price trajectory reconstruction</name>
  <files>
    src/features/__init__.py
    src/features/returns.py
    tests/test_features.py
  </files>
  <action>
    Implement stationary daily log returns computation and exact numerical price path reconstruction.
    
    Steps:
    1. Initialize `src/features/__init__.py`.
    2. In `src/features/returns.py`, implement:
       - `compute_log_returns(prices: pd.Series | np.ndarray) -> pd.Series | np.ndarray`:
         Computes $r_t = \ln(P_t / P_{t-1})$. Preserves index if input is pandas Series. First element is NaN (or optionally dropped depending on `drop_first=False` parameter).
       - `reconstruct_price_trajectory(initial_price: float, log_returns: np.ndarray | pd.Series) -> np.ndarray`:
         Deterministically converts returns back to price levels:
         $\hat{P}_t = P_0 \exp\left(\sum_{\tau=1}^t \hat{r}_\tau\right) = P_0 \exp(\text{cumsum}(\hat{r}))$.
       - `reconstruct_prices_from_base(base_prices: pd.Series, log_returns: pd.Series) -> pd.Series`:
         Reconstructs an aligned price Series where each point at $t$ is calculated from base price $P_0$ (the price immediately preceding the return sequence).
    3. In `tests/test_features.py`, implement return tests:
       - Invertibility test: Using historical prices from `fetch_stock_data("GOOGL")`, compute log returns, reconstruct prices starting at $P_0$, and assert `np.allclose(actual[1:], reconstructed, atol=1e-5)`.
       - Boundary test: Assert behavior with constant prices (all returns 0.0, reconstructed matches constant), single-point inputs, and null checks.
    
    AVOID: Using percentage returns $(P_t - P_{t-1})/P_{t-1}$ as model targets; log returns are additive and time-reversible.
    USE: Vectorized `np.log(prices / prices.shift(1))` and `initial_price * np.exp(np.cumsum(log_returns))`.
  </action>
  <verify>
    uv run pytest tests/test_features.py -k "returns or reconstruct" -v
  </verify>
  <done>
    `compute_log_returns` and `reconstruct_price_trajectory` pass tests with numerical reconstruction error $< 10^{-5}$.
  </done>
</task>

<task type="auto">
  <name>Implement comprehensive zero-lookahead technical indicator feature suite</name>
  <files>
    src/features/technical.py
    tests/test_features.py
  </files>
  <action>
    Implement vectorized technical indicators operating strictly on historical bar data with zero forward-looking bias.
    
    Steps:
    1. In `src/features/technical.py`, implement pure-numpy/pandas indicator functions:
       - `compute_rsi(series: pd.Series, period: int = 14) -> pd.Series`:
         Wilder's RSI calculating gain/loss exponential moving averages with `ewm(alpha=1/period, adjust=False)`.
       - `compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame`:
         Returns DataFrame with `macd_line` (fast EMA - slow EMA), `macd_signal` (EMA of MACD line), and `macd_hist` (`macd_line - macd_signal`).
       - `compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame`:
         Returns `bb_middle` (SMA), `bb_upper` (SMA + num_std * std), `bb_lower` (SMA - num_std * std), `bb_percent` ($(Close - Lower) / (Upper - Lower)$), and `bb_width` ($(Upper - Lower) / Middle$).
       - `compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series`:
         Average True Range based on true range $TR = \max(H - L, |H - C_{t-1}|, |L - C_{t-1}|)$ smoothed with rolling mean/EWM.
       - `compute_moving_averages(series: pd.Series, windows: list[int] = [10, 20, 50]) -> pd.DataFrame`:
         Calculates `sma_{w}` and `ema_{w}` for each specified window.
       - `compute_rolling_volatility(log_returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series`:
         Rolling standard deviation of log returns, annualized by default with factor $\sqrt{252}$.
       - `add_all_technical_features(df: pd.DataFrame, drop_burn_in: bool = True) -> pd.DataFrame`:
         Master pipeline function computing log returns and all above technical indicators on OHLCV DataFrame. If `drop_burn_in=True`, drops the initial rows with NaNs resulting from the longest rolling window (e.g. 50 periods).
    2. In `tests/test_features.py`, implement verification tests:
       - Bounds test: RSI strictly between 0 and 100; Bollinger upper $\ge$ middle $\ge$ lower.
       - Leakage test: Modifying future price bars (at index $t+k$) must NOT change indicator values calculated at index $t$.
       - NaN burn-in test: Assert `add_all_technical_features(df, drop_burn_in=True)` contains exactly 0 NaN values across all rows and columns.
    
    AVOID: Centered rolling windows (`center=True`) or `bfill()` which introduces lookahead leakage.
    USE: Trailing rolling calculations (`rolling(window=...)`) and trailing exponential moving averages.
  </action>
  <verify>
    uv run pytest tests/test_features.py -v
  </verify>
  <done>
    `tests/test_features.py` passes 100%, verifying indicator bounds, zero forward-leakage, and 0 NaN values after burn-in drop.
  </done>
</task>

## Must-Haves
After all tasks complete, verify:
- [ ] Log returns $r_t = \ln(P_t / P_{t-1})$ are computed cleanly with stationarity preserved.
- [ ] Price trajectory reconstruction error satisfies $< 10^{-5}$ on real stock data.
- [ ] All technical indicators (RSI, MACD, Bollinger Bands, ATR, SMA/EMA, Volatility) execute strictly without future leakage.
- [ ] `add_all_technical_features(df, drop_burn_in=True)` returns a DataFrame with 0 NaNs.
- [ ] All tests in `tests/test_features.py` pass via `uv run pytest tests/test_features.py`.

## Success Criteria
- [ ] All tasks verified passing.
- [ ] Must-haves confirmed.
- [ ] No regressions in test suite.

