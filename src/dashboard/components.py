"""Plotly visualization engine and KPI card formatting components for Streamlit dashboard."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "Market Overview & Technical Overlays",
) -> go.Figure:
    """Create an interactive dual-panel Plotly chart with OHLC candlestick and volume subplots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by Datetime containing 'Open', 'High', 'Low', 'Close',
        and optionally 'Volume', Bollinger Bands ('bb_upper', 'bb_middle', 'bb_lower'),
        and EMAs ('ema_10', 'ema_20', 'ema_50').
    title : str, default "Market Overview & Technical Overlays"
        Chart title displayed above the subplots.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with synchronized x-axes.
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title=f"{title} (No Data)",
        )
        return fig

    dates = df.index

    # Create 2-row subplot: 75% for Candlestick + Overlays, 25% for Volume
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.74, 0.26],
    )

    # 1. Candlestick Trace (Row 1)
    candlestick = go.Candlestick(
        x=dates,
        open=df["Open"] if "Open" in df.columns else df["Close"],
        high=df["High"] if "High" in df.columns else df["Close"],
        low=df["Low"] if "Low" in df.columns else df["Close"],
        close=df["Close"],
        name="OHLC",
        increasing_line_color="#16a34a",
        decreasing_line_color="#dc2626",
    )
    fig.add_trace(candlestick, row=1, col=1)

    # Technical Overlay: Bollinger Bands (Row 1)
    if "bb_upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["bb_upper"],
                mode="lines",
                line=dict(color="rgba(59, 130, 246, 0.55)", width=1),
                name="BB Upper",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    if "bb_lower" in df.columns:
        fill_mode = "tonexty" if "bb_upper" in df.columns else None
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["bb_lower"],
                mode="lines",
                line=dict(color="rgba(59, 130, 246, 0.55)", width=1),
                fill=fill_mode,
                fillcolor="rgba(59, 130, 246, 0.08)",
                name="BB Lower",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    if "bb_middle" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["bb_middle"],
                mode="lines",
                line=dict(color="rgba(59, 130, 246, 0.4)", width=1, dash="dash"),
                name="BB Middle",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Technical Overlay: Exponential Moving Averages (Row 1)
    if "ema_10" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["ema_10"],
                mode="lines",
                line=dict(color="#d97706", width=1.5),
                name="EMA 10",
            ),
            row=1,
            col=1,
        )

    if "ema_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["ema_20"],
                mode="lines",
                line=dict(color="#0d9488", width=1.5),
                name="EMA 20",
            ),
            row=1,
            col=1,
        )

    if "ema_50" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["ema_50"],
                mode="lines",
                line=dict(color="#9333ea", width=1.5),
                name="EMA 50",
            ),
            row=1,
            col=1,
        )

    # 2. Volume Bar Trace (Row 2)
    if "Volume" in df.columns:
        if "Close" in df.columns and "Open" in df.columns:
            vol_colors = [
                "rgba(22, 163, 74, 0.6)" if c >= o else "rgba(220, 38, 38, 0.6)"
                for c, o in zip(df["Close"], df["Open"])
            ]
        else:
            vol_colors = ["rgba(59, 130, 246, 0.6)"] * len(df)

        fig.add_trace(
            go.Bar(
                x=dates,
                y=df["Volume"],
                name="Volume",
                marker_color=vol_colors,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Clean light-theme layout with bottom legend to avoid title collisions on narrow screens
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a"),
        title=dict(text=title, x=0.01, y=0.97, font=dict(size=15)),
        hovermode="x unified",
        height=540,
        margin=dict(l=48, r=24, t=52, b=72),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(rangeslider_visible=False, gridcolor="#f1f5f9")
    fig.update_yaxes(title_text="Price ($)", gridcolor="#f1f5f9", row=1, col=1)
    fig.update_yaxes(title_text="Volume", gridcolor="#f1f5f9", row=2, col=1)

    return fig


def create_forecast_fan_chart(
    historical_dates: Any,
    historical_prices: Any,
    forecast_date: Any,
    current_price: float,
    pred_price: float,
    lower_95_price: float,
    upper_95_price: float,
    symbol: str = "GOOGL",
) -> go.Figure:
    """Create an interactive Plotly fan chart displaying historical prices and 95% MC Dropout bounds.

    Parameters
    ----------
    historical_dates : Any
        Sequence of historical date objects or date strings.
    historical_prices : Any
        Sequence of historical close prices matching historical_dates.
    forecast_date : Any
        Forward date corresponding to the single-step forecast.
    current_price : float
        Last observed close price P_0.
    pred_price : float
        Model reconstructed expected price P_hat_1.
    lower_95_price : float
        Lower bound of 95% predictive confidence interval.
    upper_95_price : float
        Upper bound of 95% predictive confidence interval.
    symbol : str, default "GOOGL"
        Stock ticker symbol.

    Returns
    -------
    go.Figure
        Interactive Plotly figure with shaded confidence interval envelope.
    """
    hist_d = list(historical_dates)
    hist_p = [float(p) for p in historical_prices]

    if not hist_d:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title=f"{symbol.upper()} 1-Step Price Forecast (No Data)",
        )
        return fig

    last_date = hist_d[-1]
    fig = go.Figure()

    # 1. Historical Close Line
    fig.add_trace(
        go.Scatter(
            x=hist_d,
            y=hist_p,
            mode="lines",
            name="Historical Close",
            line=dict(color="#2563eb", width=2.2),
        )
    )

    # 2. Upper Confidence Fan Bound
    fig.add_trace(
        go.Scatter(
            x=[last_date, forecast_date],
            y=[current_price, upper_95_price],
            mode="lines",
            line=dict(color="rgba(13, 148, 136, 0.35)", width=1, dash="dot"),
            showlegend=False,
            name="Upper 95% CI",
        )
    )

    # 3. Lower Confidence Fan Bound with shaded envelope filling to upper
    fig.add_trace(
        go.Scatter(
            x=[last_date, forecast_date],
            y=[current_price, lower_95_price],
            mode="lines",
            line=dict(color="rgba(13, 148, 136, 0.35)", width=1, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(13, 148, 136, 0.18)",
            name="95% MC Dropout Fan",
        )
    )

    # 4. Expected Trajectory Line & Predicted Marker
    fig.add_trace(
        go.Scatter(
            x=[last_date, forecast_date],
            y=[current_price, pred_price],
            mode="lines+markers",
            line=dict(color="#0d9488", width=2.5, dash="dash"),
            marker=dict(size=9, color="#0d9488", line=dict(color="#ffffff", width=1.5)),
            name="Predicted Price",
        )
    )

    # Annotate predicted price and expected delta percentage safely inside the plot area
    delta_pct = (
        ((pred_price - current_price) / current_price) * 100.0
        if current_price > 0.0
        else 0.0
    )
    sign = "+" if delta_pct >= 0.0 else ""
    annotation_text = f"Pred: ${pred_price:,.2f} ({sign}{delta_pct:.2f}%)"

    fig.add_annotation(
        x=forecast_date,
        y=pred_price,
        text=annotation_text,
        showarrow=True,
        arrowhead=2,
        arrowcolor="#0d9488",
        font=dict(color="#0f172a", size=12),
        bgcolor="#ffffff",
        bordercolor="#0d9488",
        borderwidth=1.5,
        borderpad=5,
        ax=-80,
        ay=-35,
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a"),
        title=dict(
            text=f"{symbol.upper()} 1-Step Price Forecast & 95% Uncertainty Envelope",
            x=0.01,
            y=0.97,
            font=dict(size=15),
        ),
        hovermode="x unified",
        height=440,
        margin=dict(l=48, r=28, t=52, b=68),
        xaxis=dict(title="Date", gridcolor="#f1f5f9"),
        yaxis=dict(title="Price ($)", gridcolor="#f1f5f9"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
    )

    return fig


def create_equity_curve_chart(
    equity_df: pd.DataFrame,
    symbol: str = "GOOGL",
) -> go.Figure:
    """Create a dual-panel Plotly chart of compounding portfolio equity and underwater drawdowns.

    Parameters
    ----------
    equity_df : pd.DataFrame
        DataFrame with columns 'strategy_equity' and 'buy_and_hold_equity', and either
        a 'date' column or a DatetimeIndex.
    symbol : str, default "GOOGL"
        Stock ticker symbol.

    Returns
    -------
    go.Figure
        Dual-panel interactive Plotly figure.
    """
    if equity_df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title=f"{symbol.upper()} Walk-Forward Compounding Equity (No Data)",
        )
        return fig

    dates = equity_df["date"] if "date" in equity_df.columns else equity_df.index
    strat_eq = equity_df["strategy_equity"]
    bh_eq = equity_df["buy_and_hold_equity"]

    # Compute high-water marks and percentage underwater drawdowns (<= 0%)
    strat_hwm = strat_eq.cummax()
    strat_dd = np.where(strat_hwm > 0.0, ((strat_eq - strat_hwm) / strat_hwm) * 100.0, 0.0)

    bh_hwm = bh_eq.cummax()
    bh_dd = np.where(bh_hwm > 0.0, ((bh_eq - bh_hwm) / bh_hwm) * 100.0, 0.0)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.68, 0.32],
    )

    # Row 1: Compounding Equity Lines
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=strat_eq,
            mode="lines",
            name="Strategy",
            line=dict(color="#16a34a", width=2.4),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=bh_eq,
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#7c3aed", width=2, dash="dot"),
        ),
        row=1,
        col=1,
    )

    # Row 2: Underwater Drawdown Areas
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=strat_dd,
            mode="lines",
            name="Strategy Drawdown",
            fill="tozeroy",
            fillcolor="rgba(220, 38, 38, 0.18)",
            line=dict(color="#dc2626", width=1.5),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=bh_dd,
            mode="lines",
            name="B&H Drawdown",
            line=dict(color="#64748b", width=1.2, dash="dot"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(color="#0f172a"),
        title=dict(
            text=f"{symbol.upper()} Walk-Forward Compounding Equity & Underwater Drawdown",
            x=0.01,
            y=0.97,
            font=dict(size=15),
        ),
        hovermode="x unified",
        height=520,
        margin=dict(l=48, r=24, t=52, b=72),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(gridcolor="#f1f5f9")
    fig.update_yaxes(title_text="Portfolio Equity ($)", gridcolor="#f1f5f9", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", gridcolor="#f1f5f9", row=2, col=1)

    return fig


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert a value to float, handling None, NaN, inf, or strings."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


def format_kpi_cards(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and format backtest financial metrics into structured cards for UI display.

    Note on unit conventions from ``src.evaluation.metrics``:
    - ``cumulative_return``, ``annualized_return``, ``annualized_volatility`` are ratios
      (e.g. ``0.285`` for ``28.50%``).
    - ``max_drawdown``, ``win_rate``, and ``directional_accuracy`` are already expressed
      in percentage points ``[0.0, 100.0]`` (e.g. ``8.5`` for ``8.50%``).

    Returns
    -------
    List[Dict[str, Any]]
        List of card dictionaries with keys:
        'label', 'strategy', 'benchmark', 'delta', 'is_positive', 'delta_color', 'category'.
    """
    cards: List[Dict[str, Any]] = []

    # 1. Cumulative Return (fraction -> %)
    s_cum = _safe_float(
        summary.get("strategy_cumulative_return", summary.get("cumulative_return"))
    )
    b_cum = _safe_float(summary.get("buy_and_hold_cumulative_return"))
    if s_cum is not None and b_cum is not None:
        delta = s_cum - b_cum
        cards.append(
            {
                "label": "Cumulative Return",
                "strategy": f"{s_cum * 100:+.2f}%",
                "benchmark": f"{b_cum * 100:+.2f}%",
                "delta": f"{delta * 100:+.2f}%",
                "is_positive": delta >= 0.0,
                "delta_color": "normal",
                "category": "performance",
            }
        )
    elif s_cum is not None:
        cards.append(
            {
                "label": "Cumulative Return",
                "strategy": f"{s_cum * 100:+.2f}%",
                "benchmark": "N/A",
                "delta": f"{s_cum * 100:+.2f}%",
                "is_positive": s_cum >= 0.0,
                "delta_color": "normal",
                "category": "performance",
            }
        )

    # 2. Annualized Return (CAGR, fraction -> %)
    s_cagr = _safe_float(
        summary.get("strategy_annualized_return", summary.get("annualized_return"))
    )
    b_cagr = _safe_float(summary.get("buy_and_hold_annualized_return"))
    if s_cagr is not None and b_cagr is not None:
        delta = s_cagr - b_cagr
        cards.append(
            {
                "label": "Annualized Return (CAGR)",
                "strategy": f"{s_cagr * 100:+.2f}%",
                "benchmark": f"{b_cagr * 100:+.2f}%",
                "delta": f"{delta * 100:+.2f}%",
                "is_positive": delta >= 0.0,
                "delta_color": "normal",
                "category": "performance",
            }
        )

    # 3. Win Rate (already 0-100%)
    s_win = _safe_float(summary.get("strategy_win_rate", summary.get("win_rate")))
    b_win = _safe_float(summary.get("buy_and_hold_win_rate"))
    if s_win is not None:
        b_win_pct = b_win if b_win is not None else 50.0
        delta = s_win - b_win_pct
        cards.append(
            {
                "label": "Win Rate",
                "strategy": f"{s_win:.2f}%",
                "benchmark": f"{b_win_pct:.2f}%" if b_win is not None else "50.00%",
                "delta": f"{delta:+.2f}%",
                "is_positive": delta >= 0.0,
                "delta_color": "normal",
                "category": "performance",
            }
        )

    # 4. Directional Accuracy (already 0-100%)
    dir_acc = _safe_float(summary.get("directional_accuracy"))
    if dir_acc is not None:
        delta = dir_acc - 50.0
        cards.append(
            {
                "label": "Directional Accuracy",
                "strategy": f"{dir_acc:.2f}%",
                "benchmark": "50.00%",
                "delta": f"{delta:+.2f}%",
                "is_positive": delta >= 0.0,
                "delta_color": "normal",
                "category": "performance",
            }
        )

    # 5. Annualized Volatility (fraction -> %, lower is better -> inverse)
    s_vol = _safe_float(
        summary.get("strategy_annualized_volatility", summary.get("annualized_volatility"))
    )
    b_vol = _safe_float(summary.get("buy_and_hold_annualized_volatility"))
    if s_vol is not None and b_vol is not None:
        delta = s_vol - b_vol
        cards.append(
            {
                "label": "Annualized Volatility",
                "strategy": f"{s_vol * 100:.2f}%",
                "benchmark": f"{b_vol * 100:.2f}%",
                "delta": f"{delta * 100:+.2f}%",
                "is_positive": s_vol <= b_vol,
                "delta_color": "inverse",
                "category": "risk",
            }
        )

    # 6. Maximum Drawdown (already 0-100%, lower is better -> inverse)
    s_dd = _safe_float(summary.get("strategy_max_drawdown", summary.get("max_drawdown")))
    b_dd = _safe_float(summary.get("buy_and_hold_max_drawdown"))
    if s_dd is not None and b_dd is not None:
        s_pct = abs(s_dd)
        b_pct = abs(b_dd)
        delta = s_pct - b_pct
        cards.append(
            {
                "label": "Maximum Drawdown",
                "strategy": f"{s_pct:.2f}%",
                "benchmark": f"{b_pct:.2f}%",
                "delta": f"{delta:+.2f}%",
                "is_positive": s_pct <= b_pct,
                "delta_color": "inverse",
                "category": "risk",
            }
        )

    # 7. Sharpe Ratio
    s_sharpe = _safe_float(
        summary.get("strategy_sharpe_ratio", summary.get("sharpe_ratio"))
    )
    b_sharpe = _safe_float(summary.get("buy_and_hold_sharpe_ratio"))
    if s_sharpe is not None and b_sharpe is not None:
        delta = s_sharpe - b_sharpe
        cards.append(
            {
                "label": "Sharpe Ratio",
                "strategy": f"{s_sharpe:.2f}",
                "benchmark": f"{b_sharpe:.2f}",
                "delta": f"{delta:+.2f}",
                "is_positive": delta >= 0.0,
                "delta_color": "normal",
                "category": "risk",
            }
        )

    # 8. Sortino Ratio
    s_sortino = _safe_float(
        summary.get("strategy_sortino_ratio", summary.get("sortino_ratio"))
    )
    b_sortino = _safe_float(summary.get("buy_and_hold_sortino_ratio"))
    if s_sortino is not None and b_sortino is not None:
        delta = s_sortino - b_sortino
        cards.append(
            {
                "label": "Sortino Ratio",
                "strategy": f"{s_sortino:.2f}",
                "benchmark": f"{b_sortino:.2f}",
                "delta": f"{delta:+.2f}",
                "is_positive": delta >= 0.0,
                "delta_color": "normal",
                "category": "risk",
            }
        )

    # 9. Final Portfolio Equity
    final_strat = _safe_float(summary.get("final_strategy_equity"))
    final_bh = _safe_float(summary.get("final_buy_and_hold_equity"))
    if final_strat is not None and final_bh is not None:
        delta_eq = final_strat - final_bh
        sign_prefix = "+" if delta_eq >= 0 else "-"
        cards.append(
            {
                "label": "Final Portfolio Equity",
                "strategy": f"${final_strat:,.2f}",
                "benchmark": f"${final_bh:,.2f}",
                "delta": f"{sign_prefix}${abs(delta_eq):,.2f}",
                "is_positive": delta_eq >= 0.0,
                "delta_color": "normal",
                "category": "execution",
            }
        )

    # 10. Profit Factor
    s_pf = _safe_float(summary.get("strategy_profit_factor", summary.get("profit_factor")))
    b_pf = _safe_float(summary.get("buy_and_hold_profit_factor"))
    if s_pf is not None:
        delta = (s_pf - b_pf) if b_pf is not None else (s_pf - 1.0)
        cards.append(
            {
                "label": "Profit Factor",
                "strategy": f"{s_pf:.2f}",
                "benchmark": f"{b_pf:.2f}" if b_pf is not None else "1.00",
                "delta": f"{delta:+.2f}",
                "is_positive": s_pf >= 1.0,
                "delta_color": "normal",
                "category": "execution",
            }
        )

    # 11. Executed Trades
    total_trades = summary.get("total_trades")
    if total_trades is not None:
        cards.append(
            {
                "label": "Executed Trades",
                "strategy": f"{int(total_trades):,}",
                "benchmark": "1",
                "delta": "Active Rebalancing",
                "is_positive": True,
                "delta_color": "off",
                "category": "execution",
            }
        )

    # 12. Total Transaction Fees (cost drag -> inverse so positive fee delta is red/warning)
    fees_dollars = _safe_float(
        summary.get("total_transaction_costs", summary.get("total_transaction_fees"))
    )
    if fees_dollars is not None:
        cards.append(
            {
                "label": "Total Transaction Fees",
                "strategy": f"${fees_dollars:,.2f}",
                "benchmark": "$0.00",
                "delta": f"+${fees_dollars:,.2f}",
                "is_positive": fees_dollars == 0.0,
                "delta_color": "inverse",
                "category": "execution",
            }
        )

    return cards
