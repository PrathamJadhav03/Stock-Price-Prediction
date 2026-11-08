"""Plotly visualization engine and KPI card formatting components for Streamlit dashboard."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Union
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
            template="plotly_dark",
            title=f"{title} (No Data)",
        )
        return fig

    dates = df.index

    # Create 2-row subplot: 75% for Candlestick + Overlays, 25% for Volume
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.75, 0.25],
    )

    # 1. Candlestick Trace (Row 1)
    candlestick = go.Candlestick(
        x=dates,
        open=df["Open"] if "Open" in df.columns else df["Close"],
        high=df["High"] if "High" in df.columns else df["Close"],
        low=df["Low"] if "Low" in df.columns else df["Close"],
        close=df["Close"],
        name="OHLC",
        increasing_line_color="#2ecc71",
        decreasing_line_color="#e74c3c",
    )
    fig.add_trace(candlestick, row=1, col=1)

    # Technical Overlay: Bollinger Bands (Row 1)
    if "bb_upper" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=df["bb_upper"],
                mode="lines",
                line=dict(color="rgba(100, 149, 237, 0.6)", width=1),
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
                line=dict(color="rgba(100, 149, 237, 0.6)", width=1),
                fill=fill_mode,
                fillcolor="rgba(100, 149, 237, 0.1)",
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
                line=dict(color="rgba(100, 149, 237, 0.4)", width=1, dash="dash"),
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
                line=dict(color="#f39c12", width=1.5),
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
                line=dict(color="#00f5d4", width=1.5),
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
                line=dict(color="#d63384", width=1.5),
                name="EMA 50",
            ),
            row=1,
            col=1,
        )

    # 2. Volume Bar Trace (Row 2)
    if "Volume" in df.columns:
        if "Close" in df.columns and "Open" in df.columns:
            vol_colors = [
                "rgba(46, 204, 113, 0.7)" if c >= o else "rgba(231, 76, 60, 0.7)"
                for c, o in zip(df["Close"], df["Open"])
            ]
        else:
            vol_colors = ["rgba(100, 149, 237, 0.7)"] * len(df)

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

    # Clean layout styling
    fig.update_layout(
        template="plotly_dark",
        title=dict(text=title, x=0.02, y=0.98),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=30),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

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
        Last observed close price $P_0$.
    pred_price : float
        Model reconstructed expected price $\\hat{P}_1$.
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
            template="plotly_dark",
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
            line=dict(color="#3498db", width=2),
        )
    )

    # 2. Upper Confidence Fan Bound
    fig.add_trace(
        go.Scatter(
            x=[last_date, forecast_date],
            y=[current_price, upper_95_price],
            mode="lines",
            line=dict(width=0),
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
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(0, 245, 212, 0.25)",
            name="95% MC Dropout Fan",
        )
    )

    # 4. Expected Trajectory Line & Predicted Marker
    fig.add_trace(
        go.Scatter(
            x=[last_date, forecast_date],
            y=[current_price, pred_price],
            mode="lines+markers",
            line=dict(color="#00f5d4", width=2, dash="dash"),
            marker=dict(size=8, color="#00f5d4"),
            name="Predicted Price",
        )
    )

    # Annotate predicted price and expected delta percentage
    delta_pct = (
        ((pred_price - current_price) / current_price) * 100.0
        if current_price > 0.0
        else 0.0
    )
    sign = "+" if delta_pct >= 0.0 else ""
    annotation_text = f"Pred: ${pred_price:.2f} ({sign}{delta_pct:.2f}%)"

    fig.add_annotation(
        x=forecast_date,
        y=pred_price,
        text=annotation_text,
        showarrow=True,
        arrowhead=2,
        arrowcolor="#00f5d4",
        font=dict(color="#ffffff", size=12),
        bgcolor="#1e222d",
        bordercolor="#00f5d4",
        borderwidth=1,
        ax=40,
        ay=-30,
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"{symbol.upper()} 1-Step Price Forecast & 95% Uncertainty Fan",
            x=0.02,
            y=0.98,
        ),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=30),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Price ($)"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
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
            template="plotly_dark",
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
        vertical_spacing=0.05,
        row_heights=[0.70, 0.30],
    )

    # Row 1: Compounding Equity Lines
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=strat_eq,
            mode="lines",
            name="Strategy",
            line=dict(color="#2ecc71", width=2.5),
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
            line=dict(color="#9b59b6", width=2, dash="dot"),
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
            fillcolor="rgba(231, 76, 60, 0.3)",
            line=dict(color="#e74c3c", width=1.5),
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
            line=dict(color="#7f8c8d", width=1, dash="dot"),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"{symbol.upper()} Walk-Forward Compounding Equity & Underwater Drawdown",
            x=0.02,
            y=0.98,
        ),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_yaxes(title_text="Portfolio Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

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

    Parameters
    ----------
    summary : Dict[str, Any]
        Dictionary returned by backtesting engine containing strategy, buy-and-hold,
        and execution statistics.

    Returns
    -------
    List[Dict[str, Any]]
        List of card dictionaries with keys:
        'label', 'strategy', 'benchmark', 'delta', 'is_positive'.
    """
    cards: List[Dict[str, Any]] = []

    # 1. Cumulative Return
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
            }
        )

    # 2. Annualized Return (CAGR)
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
            }
        )

    # 3. Annualized Volatility
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
                "is_positive": s_vol <= b_vol,  # lower volatility is favorable
            }
        )

    # 4. Sharpe Ratio
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
            }
        )

    # 5. Sortino Ratio
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
            }
        )

    # 6. Maximum Drawdown
    s_dd = _safe_float(summary.get("strategy_max_drawdown", summary.get("max_drawdown")))
    b_dd = _safe_float(summary.get("buy_and_hold_max_drawdown"))
    if s_dd is not None and b_dd is not None:
        # Lower drawdown magnitude is better
        s_pct = abs(s_dd) * 100.0
        b_pct = abs(b_dd) * 100.0
        delta = s_pct - b_pct
        cards.append(
            {
                "label": "Maximum Drawdown",
                "strategy": f"{s_pct:.2f}%",
                "benchmark": f"{b_pct:.2f}%",
                "delta": f"{delta:+.2f}%",
                "is_positive": s_pct <= b_pct,
            }
        )

    # 7. Win Rate
    s_win = _safe_float(summary.get("strategy_win_rate", summary.get("win_rate")))
    b_win = _safe_float(summary.get("buy_and_hold_win_rate"))
    if s_win is not None:
        s_win_pct = s_win * 100.0
        b_win_pct = (b_win * 100.0) if b_win is not None else 50.0
        delta = s_win_pct - b_win_pct
        cards.append(
            {
                "label": "Win Rate",
                "strategy": f"{s_win_pct:.2f}%",
                "benchmark": f"{b_win_pct:.2f}%" if b_win is not None else "50.00%",
                "delta": f"{delta:+.2f}%",
                "is_positive": delta >= 0.0,
            }
        )

    # 8. Profit Factor
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
            }
        )

    # 9. Directional Accuracy
    dir_acc = _safe_float(summary.get("directional_accuracy"))
    if dir_acc is not None:
        acc_pct = dir_acc * 100.0
        delta = acc_pct - 50.0
        cards.append(
            {
                "label": "Directional Accuracy",
                "strategy": f"{acc_pct:.2f}%",
                "benchmark": "50.00%",
                "delta": f"{delta:+.2f}%",
                "is_positive": delta >= 0.0,
            }
        )

    # 10. Total Transaction Fees
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
                "is_positive": True,  # Informational metric
            }
        )

    return cards

