"""Interactive Streamlit financial analytics dashboard for stock forecasting and backtesting."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for direct "streamlit run" invocations
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import requests
import streamlit as st

from src.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    PredictRequest,
    PredictResponse,
)
from src.api.service import ModelService
from src.dashboard.components import (
    create_candlestick_chart,
    create_equity_curve_chart,
    create_forecast_fan_chart,
    format_kpi_cards,
)

# 1. Page Configuration
st.set_page_config(
    page_title="Stock Price Forecasting & Quantitative Backtesting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


class BackendClient:
    """Dual-mode client communicating with live FastAPI REST API or local ModelService."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        """Initialize backend client and detect if FastAPI service is operational."""
        self.base_url = base_url.rstrip("/")
        self._use_remote = False
        self._local_service: Optional[ModelService] = None

        # Check if FastAPI service is running
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                self._use_remote = True
        except Exception:
            self._use_remote = False

        if not self._use_remote:
            self._local_service = ModelService()

    @property
    def is_remote(self) -> bool:
        """Return True if connected to live FastAPI server."""
        return self._use_remote

    @property
    def mode_name(self) -> str:
        """Return human-readable backend operational mode string."""
        return "FastAPI REST" if self._use_remote else "In-Process Service"

    @property
    def device_name(self) -> str:
        """Return active compute device (cuda / cpu)."""
        if self._use_remote:
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=1.0).json()
                return str(resp.get("device", "cpu"))
            except Exception:
                return "cpu"
        return str(self._local_service.device) if self._local_service else "cpu"

    def get_available_models(self) -> List[str]:
        """Return list of uppercase symbols with trained model checkpoints."""
        if self._use_remote:
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=1.0).json()
                models = resp.get("models_available", [])
                if models:
                    return [str(m).upper() for m in models]
            except Exception:
                pass
        return self._local_service.get_available_models() if self._local_service else []

    def predict(self, req: PredictRequest) -> PredictResponse:
        """Execute single-step inference via remote REST or local service fallback."""
        if self._use_remote:
            try:
                resp = requests.post(
                    f"{self.base_url}/predict",
                    json=req.model_dump(),
                    timeout=15.0,
                )
                resp.raise_for_status()
                return PredictResponse(**resp.json())
            except Exception:
                if self._local_service is None:
                    self._local_service = ModelService()
                return self._local_service.predict(req)
        return self._local_service.predict(req)

    def run_backtest(self, req: BacktestRequest) -> BacktestResponse:
        """Execute walk-forward backtest via remote REST or local service fallback."""
        if self._use_remote:
            try:
                resp = requests.post(
                    f"{self.base_url}/backtest",
                    json=req.model_dump(),
                    timeout=60.0,
                )
                resp.raise_for_status()
                return BacktestResponse(**resp.json())
            except Exception:
                if self._local_service is None:
                    self._local_service = ModelService()
                return self._local_service.run_backtest(req)
        return self._local_service.run_backtest(req)


@st.cache_resource
def get_service() -> BackendClient:
    """Provide a cached singleton instance of the backend service client."""
    return BackendClient()


@st.cache_data(show_spinner=False)
def load_ticker_data(symbol: str, data_dir: str = "data/processed") -> pd.DataFrame:
    """Load and concatenate historical processed data splits for the given ticker.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol.
    data_dir : str, default "data/processed"
        Path to processed data root directory.

    Returns
    -------
    pd.DataFrame
        Consolidated historical DataFrame indexed by Datetime.
    """
    sym_dir = Path(data_dir) / symbol.upper()
    if not sym_dir.exists():
        raise FileNotFoundError(
            f"Processed data directory for '{symbol}' not found at '{sym_dir}'"
        )

    dfs: List[pd.DataFrame] = []
    for split in ["train.csv", "val.csv", "test.csv"]:
        fpath = sym_dir / split
        if fpath.exists():
            df_split = pd.read_csv(fpath, index_col=0, parse_dates=True)
            dfs.append(df_split)

    if not dfs:
        raise FileNotFoundError(f"No processed CSV partitions found in '{sym_dir}'")

    full_df = pd.concat(dfs).sort_index()
    return full_df[~full_df.index.duplicated(keep="first")]


def main() -> None:
    """Render the main Streamlit interactive analytics dashboard."""
    client = get_service()
    available_models = client.get_available_models()

    # ---------------------------------------------------------
    # Sidebar Controls
    # ---------------------------------------------------------
    st.sidebar.title("Quantitative Controls")

    # Status badge
    mode_color = "#2ecc71" if client.is_remote else "#3498db"
    st.sidebar.markdown(
        f"""
        <div style="background-color: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 6px; margin-bottom: 16px; border-left: 4px solid {mode_color};">
            <span style="font-size: 13px;">Backend: <strong>{client.mode_name}</strong></span><br>
            <span style="font-size: 12px; color: #888;">Device: <code>{client.device_name}</code></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tickers = ["GOOGL", "AMZN", "IBM", "MSFT"]
    symbol = st.sidebar.selectbox("Stock Ticker", tickers, index=0)

    lookback = st.sidebar.slider(
        "Lookback Window (Days)",
        min_value=30,
        max_value=365,
        value=120,
        step=5,
        help="Number of historical trading days to visualize in technical analysis charts",
    )

    mc_samples = st.sidebar.slider(
        "Monte Carlo Dropout Samples",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Number of stochastic forward passes used to quantify predictive uncertainty",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Backtesting Configuration")

    strategy = st.sidebar.selectbox(
        "Trading Strategy",
        ["long_flat", "long_short"],
        index=0,
        help="long_flat: long on positive prediction, cash on non-positive. long_short: short on negative prediction.",
    )

    fee_bps = st.sidebar.slider(
        "Transaction Cost (bps)",
        min_value=0.0,
        max_value=50.0,
        value=10.0,
        step=1.0,
        help="Proportional transaction cost in basis points (10 bps = 0.10%) applied per turnover",
    )

    threshold = st.sidebar.slider(
        "Hurdle Threshold τ",
        min_value=0.000,
        max_value=0.010,
        value=0.000,
        step=0.001,
        format="%.3f",
        help="Minimum predicted return required before triggering a non-zero trading position",
    )

    split = st.sidebar.selectbox(
        "Partition Split",
        ["test", "val", "all"],
        index=0,
        help="Historical dataset split to evaluate walk-forward simulation across",
    )

    initial_capital = st.sidebar.number_input(
        "Initial Capital ($)",
        min_value=100.0,
        max_value=10_000_000.0,
        value=10000.0,
        step=500.0,
    )

    run_bt_button = st.sidebar.button(
        "Run Walk-Forward Backtest",
        type="primary",
        use_container_width=True,
    )

    # ---------------------------------------------------------
    # Main Header
    # ---------------------------------------------------------
    st.title("Enterprise Stock Forecasting & Walk-Forward Backtesting Engine")
    st.caption(
        "Stationary Log-Return Modeling • PyTorch GRU • Monte Carlo Dropout Uncertainty • Execution Cost Simulation"
    )
    st.write("")

    # Load historical market data for the selected ticker
    try:
        df_full = load_ticker_data(symbol)
    except Exception as e:
        st.error(f"Failed to load processed market data for {symbol}: {e}")
        return

    df_view = df_full.iloc[-lookback:] if len(df_full) >= lookback else df_full

    # ---------------------------------------------------------
    # Navigation Tabs
    # ---------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Market Overview & Technicals",
            "🔮 Model Forecast & Uncertainty",
            "📈 Backtesting Scorecard",
        ]
    )

    # =========================================================
    # TAB 1: Market Overview & Technical Analysis
    # =========================================================
    with tab1:
        st.subheader(f"{symbol} Market Overview & Technical Indicator Analysis")

        # Top Metric Bar
        latest_close = float(df_view["Close"].iloc[-1])
        prev_close = (
            float(df_view["Close"].iloc[-2]) if len(df_view) > 1 else latest_close
        )
        daily_ret = (
            ((latest_close - prev_close) / prev_close) * 100.0 if prev_close > 0 else 0.0
        )

        vol_20 = (
            float(df_view["volatility_20"].iloc[-1] * 100.0)
            if "volatility_20" in df_view.columns
            else float(
                df_view["Close"].pct_change().rolling(20).std().iloc[-1]
                * 100.0
                * math.sqrt(252)
            )
        )
        rsi_14 = (
            float(df_view["rsi_14"].iloc[-1]) if "rsi_14" in df_view.columns else 50.0
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest Close", f"${latest_close:,.2f}", f"{daily_ret:+.2f}%")
        m2.metric("Daily Return", f"{daily_ret:+.2f}%")
        m3.metric("20-Day Annualized Volatility", f"{vol_20:.2f}%")
        m4.metric("14-Day RSI", f"{rsi_14:.1f}")

        st.write("")

        # Interactive Candlestick Chart
        candlestick_fig = create_candlestick_chart(
            df_view,
            title=f"{symbol} Price Action, Bollinger Bands & Volume ({lookback} Days)",
        )
        st.plotly_chart(candlestick_fig, use_container_width=True)

        # Tabular Feature Expander
        with st.expander("🔍 View Technical Indicators & Feature Matrix"):
            st.dataframe(df_view.tail(100), use_container_width=True)

    # =========================================================
    # TAB 2: Model Forecast & Uncertainty Analysis
    # =========================================================
    with tab2:
        st.subheader(f"{symbol} 1-Step Price Forecast & Epistemic Uncertainty")

        # Check model availability
        if symbol not in available_models:
            st.warning(
                f"⚠️ Model checkpoint for **{symbol}** not found in `models/`. "
                f"Available checkpoints: {', '.join(available_models) if available_models else 'None'}. "
                f"To train a PyTorch GRU for {symbol}, run: `uv run python scripts/train.py --symbol {symbol}`."
            )
        else:
            with st.spinner(f"Computing Monte Carlo Dropout forecast (N={mc_samples})..."):
                try:
                    pred_req = PredictRequest(
                        symbol=symbol,
                        mc_samples=mc_samples,
                        return_ci=True,
                    )
                    pred_resp = client.predict(pred_req)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    pred_resp = None

            if pred_resp is not None:
                # Directional Signal Banner
                exp_ret_pct = pred_resp.predicted_log_return * 100.0
                if pred_resp.direction == "UP":
                    badge_color = "#2ecc71"
                    badge_bg = "rgba(46, 204, 113, 0.15)"
                    signal_symbol = "↗"
                    signal_desc = "BULLISH (UP)"
                else:
                    badge_color = "#e74c3c"
                    badge_bg = "rgba(231, 76, 60, 0.15)"
                    signal_symbol = "↘"
                    signal_desc = "BEARISH (DOWN)"

                st.markdown(
                    f"""
                    <div style="background-color: {badge_bg}; border: 2px solid {badge_color}; border-radius: 8px; padding: 14px 20px; margin-bottom: 24px; text-align: center;">
                        <span style="color: {badge_color}; font-size: 22px; font-weight: 700;">
                            {signal_symbol} SIGNAL: {signal_desc} | Expected Log Return: {exp_ret_pct:+.2f}% | Latency: {pred_resp.latency_ms:.1f}ms
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # 4 KPI Columns
                delta_price = pred_resp.predicted_price - pred_resp.current_price
                delta_pct = (delta_price / pred_resp.current_price) * 100.0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Observed Price ($P_0$)", f"${pred_resp.current_price:,.2f}")
                c2.metric(
                    "Predicted Price ($\\hat{P}_1$)",
                    f"${pred_resp.predicted_price:,.2f}",
                    f"{delta_pct:+.2f}%",
                )
                c3.metric(
                    "95% Confidence Interval",
                    f"${pred_resp.lower_95_price:,.2f} – ${pred_resp.upper_95_price:,.2f}",
                )
                c4.metric(
                    "Predictive Uncertainty ($\\sigma_{\\text{MC}}$)",
                    f"{pred_resp.uncertainty_std:.4f}",
                    help="Empirical standard deviation of Monte Carlo Dropout forward passes",
                )

                st.write("")

                # 95% Confidence Fan Chart
                hist_window = df_full["Close"].iloc[-60:]
                last_hist_date = pd.to_datetime(hist_window.index[-1])
                forecast_date = (last_hist_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

                fan_fig = create_forecast_fan_chart(
                    historical_dates=[d.strftime("%Y-%m-%d") for d in hist_window.index],
                    historical_prices=hist_window.values.tolist(),
                    forecast_date=forecast_date,
                    current_price=pred_resp.current_price,
                    pred_price=pred_resp.predicted_price,
                    lower_95_price=pred_resp.lower_95_price,
                    upper_95_price=pred_resp.upper_95_price,
                    symbol=symbol,
                )
                st.plotly_chart(fan_fig, use_container_width=True)

                # Explanatory Epistemic Uncertainty Callout Box
                st.info(
                    """
                    **Quantifying Epistemic Model Uncertainty via Monte Carlo Dropout:**
                    Financial asset prices exhibit heavy tails, regime shifts, and non-Gaussian volatility.
                    Rather than predicting an overconfident deterministic point value, the network retains active
                    dropout masks across $N$ stochastic forward passes during inference. The resulting empirical
                    distribution quantifies model parameter uncertainty (epistemic risk), constructing distribution-free
                    95% confidence intervals $[\\hat{P}_{0.025}, \\hat{P}_{0.975}]$.
                    """
                )

    # =========================================================
    # TAB 3: Walk-Forward Backtesting Scorecard
    # =========================================================
    with tab3:
        st.subheader(f"{symbol} Walk-Forward Backtesting Scorecard ({strategy.replace('_', ' ').title()})")

        if symbol not in available_models:
            st.warning(
                f"⚠️ Model checkpoint for **{symbol}** not found. Please train the model before running backtests."
            )
        else:
            session_key = (
                f"bt_{symbol}_{strategy}_{fee_bps}_{threshold}_{split}_{initial_capital}"
            )

            # Auto-run if not cached or user clicked button
            if session_key not in st.session_state or run_bt_button:
                with st.spinner(f"Simulating walk-forward backtest for {symbol} ({split} partition)..."):
                    try:
                        bt_req = BacktestRequest(
                            symbol=symbol,
                            fee_bps=fee_bps,
                            threshold=threshold,
                            strategy=strategy,
                            split=split,
                            initial_capital=initial_capital,
                        )
                        bt_resp = client.run_backtest(bt_req)
                        st.session_state[session_key] = bt_resp
                    except Exception as e:
                        st.error(f"Backtesting simulation failed: {e}")
                        bt_resp = None
            else:
                bt_resp = st.session_state[session_key]

            if bt_resp is not None:
                # Top KPI Cards
                cards = format_kpi_cards(bt_resp.summary)

                # Render KPI cards in a 4-column responsive grid
                cols_per_row = 4
                for row_idx in range(0, len(cards), cols_per_row):
                    row_cards = cards[row_idx : row_idx + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for col, card in zip(cols, row_cards):
                        with col:
                            delta_str = (
                                f"{card['delta']} vs B&H"
                                if card["benchmark"] != "N/A"
                                else None
                            )
                            st.metric(
                                label=card["label"],
                                value=card["strategy"],
                                delta=delta_str,
                            )
                    st.write("")

                # Compounding Equity Curve & Drawdown Subplot
                equity_df = pd.DataFrame(bt_resp.equity_curve)
                equity_fig = create_equity_curve_chart(equity_df, symbol=symbol)
                st.plotly_chart(equity_fig, use_container_width=True)

                # Detailed Step-by-Step Trade & Position Audit Log Expander
                with st.expander("📋 Detailed Step-by-Step Trade & Position Audit Log"):
                    st.dataframe(equity_df, use_container_width=True)

                    csv_data = equity_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Equity & Trade Audit Log (CSV)",
                        data=csv_data,
                        file_name=f"{symbol}_{strategy}_{split}_equity_audit.csv",
                        mime="text/csv",
                    )


if __name__ == "__main__":
    main()

