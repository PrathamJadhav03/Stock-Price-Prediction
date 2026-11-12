"""Interactive Streamlit financial analytics dashboard for stock forecasting and backtesting."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path for direct "streamlit run" invocations
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from datetime import datetime
import math
from typing import Dict, List, Optional
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
from src.data.ingestion import _normalize_ohlcv_dataframe
from src.dashboard.components import (
    create_candlestick_chart,
    create_equity_curve_chart,
    create_forecast_fan_chart,
    format_kpi_cards,
)
from src.features.technical import add_all_technical_features

# 1. Page Configuration
st.set_page_config(
    page_title="Stock Price Forecasting & Quantitative Backtesting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_responsive_css() -> None:
    """Inject responsive typography and grid layout rules for tablet and mobile viewports."""
    st.markdown(
        """
        <style>
        /* Prevent giant multi-line H1 takeover on mobile */
        h1 {
            font-size: clamp(1.35rem, 2.8vw, 2.15rem) !important;
            line-height: 1.2 !important;
            margin-bottom: 0.25rem !important;
        }
        /* Keep KPI values crisp and prevent premature ellipsis truncation */
        [data-testid="stMetricValue"] {
            font-size: clamp(1.2rem, 1.8vw, 1.75rem) !important;
            line-height: 1.2 !important;
        }
        /* Responsive wrapping: 2x2 grid on tablet widths (541px-1100px), 1-col below 540px */
        @media (max-width: 1100px) {
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap !important;
                gap: 0.75rem !important;
            }
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 calc(50% - 0.75rem) !important;
                min-width: calc(50% - 0.75rem) !important;
            }
        }
        @media (max-width: 540px) {
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 100% !important;
                min-width: 100% !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
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
def load_ticker_data(
    symbol: str,
    data_dir: str = "data/raw",
    processed_dir: str = "data/processed",
) -> pd.DataFrame:
    """Load unscaled OHLCV market history and compute real-unit technical indicators.

    Loading from ``data/raw/{symbol}.csv`` preserves true dollar prices, non-negative
    volumes, and true indicator domains (e.g. RSI in [0, 100], volatility >= 0) rather
    than exposing RobustScaler-transformed ML training tensors.
    """
    clean_sym = symbol.strip().upper()
    raw_path = Path(data_dir) / f"{clean_sym}.csv"
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        norm_df = _normalize_ohlcv_dataframe(raw_df)
        return add_all_technical_features(norm_df)

    # Fallback: check processed partition folder if raw cache is unavailable
    sym_dir = Path(processed_dir) / clean_sym
    if not sym_dir.exists():
        sym_dir = Path(data_dir) / clean_sym
    if not sym_dir.exists():
        raise FileNotFoundError(
            f"Market data for '{clean_sym}' not found at '{raw_path}' or '{sym_dir}'"
        )

    dfs: List[pd.DataFrame] = []
    for split in ["train.csv", "val.csv", "test.csv"]:
        fpath = sym_dir / split
        if fpath.exists():
            df_split = pd.read_csv(fpath, index_col=0, parse_dates=True)
            dfs.append(df_split)

    if not dfs:
        raise FileNotFoundError(f"No CSV partitions found in '{sym_dir}'")

    full_df = pd.concat(dfs).sort_index()
    return full_df[~full_df.index.duplicated(keep="first")]


def _load_baseline_backtest(symbol: str) -> Optional[BacktestResponse]:
    """Load precomputed baseline backtest scorecard if available for instant display."""
    clean_sym = symbol.strip().upper()
    json_path = Path("reports/backtest") / f"backtest_{clean_sym}.json"
    csv_path = Path("reports/backtest") / f"equity_curve_{clean_sym}.csv"
    if not (json_path.exists() and csv_path.exists()):
        return None
    try:
        import json

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        eq_df = pd.read_csv(csv_path)
        return BacktestResponse(
            symbol=clean_sym,
            strategy=str(payload.get("strategy", "long_flat")),
            fee_bps=float(payload.get("fee_bps", 10.0)),
            threshold=float(payload.get("threshold", 0.0)),
            split=str(payload.get("split", "test")),
            initial_capital=float(payload.get("initial_capital", 10000.0)),
            summary=payload.get("summary", {}),
            equity_curve=eq_df.to_dict(orient="records"),
        )
    except Exception:
        return None


def _format_audit_table(equity_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich equity curve series with computed daily returns and underwater drawdown."""
    df = equity_df.copy()
    if "strategy_equity" in df.columns and "buy_and_hold_equity" in df.columns:
        df["strategy_daily_return_pct"] = (
            df["strategy_equity"].pct_change().fillna(0.0) * 100.0
        )
        df["bh_daily_return_pct"] = (
            df["buy_and_hold_equity"].pct_change().fillna(0.0) * 100.0
        )
        hwm = df["strategy_equity"].cummax()
        df["drawdown_pct"] = np.where(
            hwm > 0.0, ((df["strategy_equity"] - hwm) / hwm) * 100.0, 0.0
        )

    rename_map = {
        "date": "Date",
        "strategy_equity": "Strategy Equity",
        "buy_and_hold_equity": "Buy & Hold Equity",
        "strategy_daily_return_pct": "Strategy Daily Return",
        "bh_daily_return_pct": "B&H Daily Return",
        "drawdown_pct": "Drawdown",
    }
    ordered_cols = [c for c in rename_map if c in df.columns]
    return df[ordered_cols].rename(columns=rename_map)


def _render_kpi_grid(cards: List[Dict[str, object]], cols_per_row: int = 4) -> None:
    """Render metric cards in evenly balanced 4-column rows."""
    for row_idx in range(0, len(cards), cols_per_row):
        row_cards = cards[row_idx : row_idx + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, card in zip(cols, row_cards):
            with col:
                benchmark = str(card.get("benchmark", "N/A"))
                raw_delta = str(card.get("delta", ""))
                if benchmark != "N/A" and card.get("label") != "Executed Trades":
                    delta_str: Optional[str] = f"{raw_delta} vs B&H"
                elif raw_delta:
                    delta_str = raw_delta
                else:
                    delta_str = None
                st.metric(
                    label=str(card["label"]),
                    value=str(card["strategy"]),
                    delta=delta_str,
                    delta_color=str(card.get("delta_color", "normal")),
                )


def main() -> None:
    """Render the main Streamlit interactive analytics dashboard."""
    _inject_responsive_css()

    client = get_service()
    available_models = client.get_available_models()

    # ---------------------------------------------------------
    # Minimal Universal Sidebar (Global Asset Selection Only)
    # ---------------------------------------------------------
    st.sidebar.title("Market Workspace")

    mode_color = "#16a34a" if client.is_remote else "#2563eb"
    st.sidebar.markdown(
        f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 10px 12px; border-radius: 8px; margin-bottom: 16px; border-left: 4px solid {mode_color};">
            <div style="font-size: 13px; color: #0f172a;">Backend: <strong>{client.mode_name}</strong></div>
            <div style="font-size: 12px; color: #475569; margin-top: 2px;">Compute: <code>{client.device_name}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tickers = ["GOOGL", "AMZN", "IBM", "MSFT"]
    symbol = st.sidebar.selectbox(
        "Stock Ticker",
        tickers,
        index=0,
        format_func=lambda s: f"{s}  —  Model Ready ✓"
        if s in available_models
        else f"{s}  —  Untrained",
        help="Select an equity ticker to inspect historical technicals, next-day forecast, and backtest performance.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Chart lookback, uncertainty sample counts, and backtest rules live directly inside each corresponding tab for zero-scroll editing."
    )

    # ---------------------------------------------------------
    # Main Header
    # ---------------------------------------------------------
    st.title("Stock Forecasting & Quantitative Backtesting")
    st.caption(
        "Stationary Log-Return GRU • Monte Carlo Dropout Uncertainty • Realistic Walk-Forward Execution"
    )

    # Load unscaled historical market data
    try:
        df_full = load_ticker_data(symbol)
    except Exception as e:
        st.error(
            f"Unable to load market data for **{symbol}**: {e}. "
            f"Run `uv run python scripts/prepare_data.py --symbol {symbol}` to populate datasets."
        )
        return

    # ---------------------------------------------------------
    # Compact Mobile-Friendly Navigation Tabs
    # ---------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Overview",
            "🔮 Forecast",
            "📈 Backtest",
        ]
    )

    # =========================================================
    # TAB 1: Market Overview & Technical Analysis
    # =========================================================
    with tab1:
        hdr_col, ctrl_col = st.columns([2.2, 1.0])
        with hdr_col:
            st.subheader(f"{symbol} Market Overview & Technicals")
        with ctrl_col:
            lookback = st.slider(
                "Lookback Window (Trading Days)",
                min_value=30,
                max_value=365,
                value=120,
                step=5,
                help="Number of recent trading sessions plotted in the candlestick and indicator overlays.",
            )

        df_view = df_full.iloc[-lookback:] if len(df_full) >= lookback else df_full

        latest_close = float(df_view["Close"].iloc[-1])
        prev_close = (
            float(df_view["Close"].iloc[-2]) if len(df_view) > 1 else latest_close
        )
        daily_ret = (
            ((latest_close - prev_close) / prev_close) * 100.0
            if prev_close > 0
            else 0.0
        )

        # Compute/read unscaled volatility (>= 0), RSI ([0, 100]), and ATR ($)
        vol_20 = (
            float(df_view["volatility_20"].iloc[-1] * 100.0)
            if "volatility_20" in df_view.columns
            and df_view["volatility_20"].iloc[-1] >= 0.0
            else float(
                df_view["Close"].pct_change().rolling(20).std().iloc[-1]
                * 100.0
                * math.sqrt(252)
            )
        )
        rsi_14 = (
            float(df_view["rsi_14"].iloc[-1])
            if "rsi_14" in df_view.columns
            and 0.0 <= float(df_view["rsi_14"].iloc[-1]) <= 100.0
            else 50.0
        )
        atr_14 = (
            float(df_view["atr_14"].iloc[-1])
            if "atr_14" in df_view.columns
            else float((df_view["High"] - df_view["Low"]).rolling(14).mean().iloc[-1])
        )

        rsi_state = (
            "Overbought"
            if rsi_14 >= 70
            else ("Oversold" if rsi_14 <= 30 else "Neutral Momentum")
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest Close", f"${latest_close:,.2f}", f"{daily_ret:+.2f}% today")
        m2.metric(
            "14-Day ATR",
            f"${atr_14:,.2f}",
            f"{(atr_14 / latest_close) * 100:.2f}% of price"
            if latest_close > 0
            else None,
            delta_color="off",
        )
        m3.metric(
            "20-Day Volatility (Ann.)",
            f"{max(0.0, vol_20):.2f}%",
            help="Annualized rolling 20-day return standard deviation.",
        )
        m4.metric("14-Day RSI", f"{rsi_14:.1f}", rsi_state, delta_color="off")

        st.write("")

        candlestick_fig = create_candlestick_chart(
            df_view,
            title=f"{symbol} Price Action, Bollinger Bands & Volume ({len(df_view)} Sessions)",
        )
        st.plotly_chart(candlestick_fig, use_container_width=True)

        with st.expander("🔍 Inspect Historical OHLCV & Technical Indicators"):
            display_cols = [
                c
                for c in [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                    "ema_10",
                    "ema_20",
                    "ema_50",
                    "rsi_14",
                    "atr_14",
                    "volatility_20",
                ]
                if c in df_view.columns
            ]
            st.dataframe(
                df_view[display_cols].tail(100).sort_index(ascending=False),
                use_container_width=True,
                column_config={
                    "Open": st.column_config.NumberColumn("Open", format="$%.2f"),
                    "High": st.column_config.NumberColumn("High", format="$%.2f"),
                    "Low": st.column_config.NumberColumn("Low", format="$%.2f"),
                    "Close": st.column_config.NumberColumn("Close", format="$%.2f"),
                    "Volume": st.column_config.NumberColumn("Volume", format="%d"),
                    "ema_10": st.column_config.NumberColumn("EMA 10", format="$%.2f"),
                    "ema_20": st.column_config.NumberColumn("EMA 20", format="$%.2f"),
                    "ema_50": st.column_config.NumberColumn("EMA 50", format="$%.2f"),
                    "rsi_14": st.column_config.NumberColumn("RSI (14)", format="%.1f"),
                    "atr_14": st.column_config.NumberColumn("ATR (14)", format="$%.2f"),
                    "volatility_20": st.column_config.NumberColumn(
                        "Volatility (20D)", format="%.4f"
                    ),
                },
            )

    # =========================================================
    # TAB 2: Model Forecast & Uncertainty Analysis
    # =========================================================
    with tab2:
        f_hdr_col, f_ctrl_col = st.columns([2.2, 1.0])
        with f_hdr_col:
            st.subheader(f"{symbol} Next-Day Forecast & Uncertainty")
        with f_ctrl_col:
            mc_samples = st.slider(
                "Monte Carlo Dropout Passes",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="Stochastic forward passes used to estimate the 95% confidence range.",
            )

        if symbol not in available_models:
            st.warning(
                f"⚠️ Model checkpoint for **{symbol}** not found in `models/`. "
                f"Available checkpoints: {', '.join(available_models) if available_models else 'None'}. "
                f"To train a PyTorch GRU for {symbol}, run: `uv run python scripts/train.py --symbol {symbol}`."
            )
        else:
            with st.spinner(
                f"Running {mc_samples} stochastic forecast passes for {symbol}..."
            ):
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
                exp_ret_pct = pred_resp.predicted_log_return * 100.0
                if pred_resp.direction == "UP":
                    badge_color = "#15803d"
                    badge_bg = "rgba(22, 163, 74, 0.10)"
                    signal_symbol = "↗"
                    signal_desc = "BULLISH (UP)"
                else:
                    badge_color = "#b91c1c"
                    badge_bg = "rgba(220, 38, 38, 0.10)"
                    signal_symbol = "↘"
                    signal_desc = "BEARISH (DOWN)"

                st.markdown(
                    f"""
                    <div style="background-color: {badge_bg}; border: 1.5px solid {badge_color}; border-radius: 8px; padding: 12px 18px; margin-bottom: 18px; text-align: center;">
                        <span style="color: {badge_color}; font-size: 19px; font-weight: 700;">
                            {signal_symbol} SIGNAL: {signal_desc} &nbsp;|&nbsp; Expected 1-Day Move: {exp_ret_pct:+.2f}%
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                delta_price = pred_resp.predicted_price - pred_resp.current_price
                delta_pct = (
                    (delta_price / pred_resp.current_price) * 100.0
                    if pred_resp.current_price > 0
                    else 0.0
                )
                half_width = (
                    pred_resp.upper_95_price - pred_resp.lower_95_price
                ) / 2.0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Current Price", f"${pred_resp.current_price:,.2f}")
                c2.metric(
                    "Forecast Price (Next Day)",
                    f"${pred_resp.predicted_price:,.2f}",
                    f"{delta_pct:+.2f}%",
                )
                c3.metric(
                    "95% Confidence Range",
                    f"${pred_resp.lower_95_price:,.0f} – ${pred_resp.upper_95_price:,.0f}",
                    f"±${half_width:,.0f} spread",
                    delta_color="off",
                    help=f"Exact 95% bounds: ${pred_resp.lower_95_price:,.2f} to ${pred_resp.upper_95_price:,.2f}",
                )
                c4.metric(
                    "Uncertainty Spread (σ)",
                    f"{pred_resp.uncertainty_std:.4f}",
                    help="Empirical standard deviation of predicted log returns across Monte Carlo Dropout passes.",
                )

                st.write("")

                # Display last 30 sessions matching model sequence length so trend & fan share balanced visual weight
                hist_window = df_full["Close"].iloc[-30:]
                last_hist_date = pd.to_datetime(hist_window.index[-1])
                forecast_date = (last_hist_date + pd.Timedelta(days=1)).strftime(
                    "%Y-%m-%d"
                )

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

                st.caption(
                    f"Inference completed in {pred_resp.latency_ms:.1f} ms on `{client.device_name}` • Shaded region represents empirical 95% confidence bounds over {mc_samples} MC Dropout passes."
                )

    # =========================================================
    # TAB 3: Walk-Forward Backtesting Scorecard
    # =========================================================
    with tab3:
        st.subheader(f"{symbol} Walk-Forward Backtesting Scorecard")

        if symbol not in available_models:
            st.warning(
                f"⚠️ Model checkpoint for **{symbol}** not found. Train a model before running simulations: "
                f"`uv run python scripts/train.py --symbol {symbol}`."
            )
        else:
            # In-page parameter controls (adjacent to results per Fitts' Law)
            with st.container(border=True):
                r1c1, r1c2, r1c3 = st.columns(3)
                with r1c1:
                    strategy = st.selectbox(
                        "Trading Strategy",
                        ["long_flat", "long_short"],
                        index=0,
                        format_func=lambda v: "Long / Cash"
                        if v == "long_flat"
                        else "Long / Short",
                        help="Long / Cash holds cash on negative signals; Long / Short actively shorts negative signals.",
                    )
                with r1c2:
                    split = st.selectbox(
                        "Evaluation Period",
                        ["test", "val", "all"],
                        index=0,
                        format_func=lambda v: {
                            "test": "Out-of-Sample Test Split",
                            "val": "Validation Split",
                            "all": "Full Historical Dataset",
                        }[v],
                        help="Dataset partition evaluated during walk-forward simulation.",
                    )
                with r1c3:
                    initial_capital = st.number_input(
                        "Initial Capital ($)",
                        min_value=100.0,
                        max_value=10_000_000.0,
                        value=10000.0,
                        step=500.0,
                    )

                r2c1, r2c2, r2c3 = st.columns([1.2, 1.2, 1.0])
                with r2c1:
                    fee_bps = st.slider(
                        "Transaction Cost (bps)",
                        min_value=0.0,
                        max_value=50.0,
                        value=10.0,
                        step=1.0,
                        help="Proportional trading fee in basis points (10 bps = 0.10%) applied per portfolio turnover.",
                    )
                with r2c2:
                    threshold = st.slider(
                        "Hurdle Threshold (%)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.05,
                        format="%.2f%%",
                        help="Minimum predicted return magnitude required before entering a position.",
                    ) / 100.0
                with r2c3:
                    st.write("")
                    run_bt_button = st.button(
                        "🔄 Re-run Simulation",
                        type="primary",
                        use_container_width=True,
                    )

            session_key = (
                f"bt_{symbol}_{strategy}_{fee_bps}_{threshold:.4f}_{split}_{initial_capital}"
            )
            ts_key = f"{session_key}_ts"

            is_default_cfg = (
                strategy == "long_flat"
                and abs(fee_bps - 10.0) < 1e-6
                and abs(threshold) < 1e-6
                and split == "test"
                and abs(initial_capital - 10000.0) < 1e-6
            )

            if run_bt_button:
                with st.spinner(f"Simulating walk-forward backtest for {symbol}..."):
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
                        st.session_state[ts_key] = (
                            f"Simulated at {datetime.now().strftime('%H:%M:%S')}"
                        )
                    except Exception as e:
                        st.error(f"Backtesting simulation failed: {e}")
                        bt_resp = None
            elif session_key in st.session_state:
                bt_resp = st.session_state[session_key]
            elif is_default_cfg:
                baseline_resp = _load_baseline_backtest(symbol)
                if baseline_resp is not None:
                    st.session_state[session_key] = baseline_resp
                    st.session_state[ts_key] = "Precomputed Baseline"
                bt_resp = baseline_resp
            else:
                bt_resp = None

            if bt_resp is None and not run_bt_button:
                st.info(
                    "Configure your strategy parameters above and click **Re-run Simulation** to compute walk-forward performance."
                )

            if bt_resp is not None:
                equity_df = pd.DataFrame(bt_resp.equity_curve)
                audit_table = _format_audit_table(equity_df)
                csv_data = audit_table.to_csv(index=False)

                # Freshness timestamp + directly discoverable CSV download action bar
                status_col, export_col = st.columns([2.5, 1.0])
                with status_col:
                    sim_time = st.session_state.get(ts_key, "Ready")
                    strat_label = (
                        "Long / Cash" if strategy == "long_flat" else "Long / Short"
                    )
                    st.caption(
                        f"Status: **{sim_time}** • Strategy: **{strat_label}** • Split: **{split.title()}** • Cost: **{fee_bps:.0f} bps**"
                    )
                with export_col:
                    st.download_button(
                        label="📥 Export Audit CSV",
                        data=csv_data,
                        file_name=f"{symbol}_{strategy}_{split}_equity_audit.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                cards = format_kpi_cards(bt_resp.summary)

                perf_cards = [c for c in cards if c.get("category") == "performance"]
                risk_cards = [c for c in cards if c.get("category") == "risk"]
                exec_cards = [c for c in cards if c.get("category") == "execution"]

                if perf_cards:
                    st.markdown("##### Return & Alpha")
                    _render_kpi_grid(perf_cards, cols_per_row=4)

                if risk_cards:
                    st.markdown("##### Risk & Drawdown")
                    _render_kpi_grid(risk_cards, cols_per_row=4)

                if exec_cards:
                    st.markdown("##### Execution & Costs")
                    _render_kpi_grid(exec_cards, cols_per_row=4)

                st.write("")

                equity_fig = create_equity_curve_chart(equity_df, symbol=symbol)
                st.plotly_chart(equity_fig, use_container_width=True)

                with st.expander("📋 Detailed Daily Equity, Returns & Drawdown Audit Log"):
                    st.dataframe(
                        audit_table,
                        use_container_width=True,
                        column_config={
                            "Strategy Equity": st.column_config.NumberColumn(
                                "Strategy Equity", format="$%.2f"
                            ),
                            "Buy & Hold Equity": st.column_config.NumberColumn(
                                "Buy & Hold Equity", format="$%.2f"
                            ),
                            "Strategy Daily Return": st.column_config.NumberColumn(
                                "Strategy Daily Return", format="%.2f%%"
                            ),
                            "B&H Daily Return": st.column_config.NumberColumn(
                                "B&H Daily Return", format="%.2f%%"
                            ),
                            "Drawdown": st.column_config.NumberColumn(
                                "Drawdown", format="%.2f%%"
                            ),
                        },
                    )


if __name__ == "__main__":
    main()
