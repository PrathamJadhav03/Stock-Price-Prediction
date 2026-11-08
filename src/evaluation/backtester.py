"""Event-driven walk-forward financial backtester with transaction cost modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    _to_1d_numpy,
    calculate_directional_accuracy,
    evaluate_financial_metrics,
)


@dataclass
class BacktestResult:
    """Container for walk-forward backtesting outputs, audit records, and KPIs."""

    summary: Dict[str, Any]
    audit_df: pd.DataFrame
    equity_curve: pd.DataFrame

    def to_dict(self) -> Dict[str, Any]:
        """Convert result summary and equity curves into a JSON-serializable dictionary."""
        def _clean_val(v: Any) -> Any:
            if isinstance(v, (np.floating, float)):
                if np.isnan(v) or np.isinf(v):
                    return None
                return float(v)
            if isinstance(v, (np.integer, int)):
                return int(v)
            return v

        clean_summary = {k: _clean_val(v) for k, v in self.summary.items()}
        return {
            "summary": clean_summary,
            "equity_curve": self.equity_curve.to_dict(orient="records"),
        }


class WalkForwardBacktester:
    """Event-driven, step-by-step walk-forward financial backtesting simulator.

    Simulates discrete trading execution (long/flat or long/short) based on forecasted
    returns, models realistic transaction fees on turnover, tracks compounding equity curves,
    and benchmarks strategy performance against a Buy-and-Hold baseline.
    """

    def __init__(
        self,
        fee_bps: float = 10.0,
        threshold: float = 0.0,
        strategy: str = "long_flat",
        initial_capital: float = 10000.0,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
        include_benchmark_fee: bool = False,
    ) -> None:
        """Initialize WalkForwardBacktester.

        Parameters
        ----------
        fee_bps : float, default=10.0
            Transaction fee per unit turnover in basis points (10 bps = 0.0010 = 0.1%).
        threshold : float, default=0.0
            Prediction hurdle threshold |\\hat{r}_t| > \\tau required to take a position.
        strategy : str, default="long_flat"
            Trading strategy logic: "long_flat" or "long_short".
        initial_capital : float, default=10000.0
            Starting portfolio cash in currency units.
        risk_free_rate : float, default=0.0
            Annualized risk-free rate used for Sharpe and Sortino ratios.
        periods_per_year : int, default=252
            Trading periods per calendar year for annualization.
        include_benchmark_fee : bool, default=False
            Whether Buy-and-Hold benchmark accounts for an initial entry fee.
        """
        if strategy not in ("long_flat", "long_short"):
            raise ValueError(
                f"Invalid strategy '{strategy}'. Must be either 'long_flat' or 'long_short'."
            )
        if fee_bps < 0.0:
            raise ValueError(f"fee_bps must be non-negative, got {fee_bps}")
        if threshold < 0.0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        if initial_capital <= 0.0:
            raise ValueError(f"initial_capital must be positive, got {initial_capital}")
        if periods_per_year <= 0:
            raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")

        self.fee_bps = float(fee_bps)
        self.fee_rate = float(fee_bps * 1e-4)
        self.threshold = float(threshold)
        self.strategy = strategy
        self.initial_capital = float(initial_capital)
        self.risk_free_rate = float(risk_free_rate)
        self.periods_per_year = int(periods_per_year)
        self.include_benchmark_fee = bool(include_benchmark_fee)

    def generate_signals(
        self, predicted_returns: Union[np.ndarray, pd.Series, list]
    ) -> np.ndarray:
        """Vectorized signal generation applying hurdle threshold and strategy mode.

        Parameters
        ----------
        predicted_returns : array-like
            Predicted return values (e.g. 1-day ahead log returns).

        Returns
        -------
        np.ndarray
            Discrete signal array containing target positions (1.0, 0.0, or -1.0).
        """
        preds = _to_1d_numpy(predicted_returns, name="predicted_returns")

        if self.strategy == "long_flat":
            signals = np.where(preds > self.threshold, 1.0, 0.0)
        elif self.strategy == "long_short":
            signals = np.where(
                preds > self.threshold,
                1.0,
                np.where(preds < -self.threshold, -1.0, 0.0),
            )
        else:
            raise ValueError(f"Unsupported strategy '{self.strategy}'.")

        return signals.astype(np.float64)

    def simulate(
        self,
        actual_returns: Union[np.ndarray, pd.Series, list],
        predicted_returns: Union[np.ndarray, pd.Series, list],
        dates: Optional[Union[pd.DatetimeIndex, pd.Series, list, np.ndarray]] = None,
        base_prices: Optional[Union[np.ndarray, pd.Series, list]] = None,
        target_prices: Optional[Union[np.ndarray, pd.Series, list]] = None,
        is_log_returns: bool = True,
    ) -> BacktestResult:
        """Execute step-by-step walk-forward backtest simulation.

        Parameters
        ----------
        actual_returns : array-like
            Realized return series over the test horizon.
        predicted_returns : array-like
            Model forecasted return series.
        dates : array-like, optional
            Timestamp index or date labels for step alignment.
        base_prices : array-like, optional
            Asset price at step start (P_{t-1}). If None, reconstructed nominally.
        target_prices : array-like, optional
            Asset price at step end (P_t). If None, derived from base_prices and actual returns.
        is_log_returns : bool, default=True
            Whether actual_returns are log returns r_t = ln(P_t/P_{t-1}) or simple returns.

        Returns
        -------
        BacktestResult
            Populated result object with summary KPIs, 14-column audit DataFrame,
            and equity curve time-series.
        """
        act_r = _to_1d_numpy(actual_returns, name="actual_returns")
        pred_r = _to_1d_numpy(predicted_returns, name="predicted_returns")

        n_steps = len(act_r)
        if len(pred_r) != n_steps:
            raise ValueError(
                f"Length mismatch: actual_returns ({n_steps}) and predicted_returns ({len(pred_r)}) must match."
            )

        # Reconstruct or validate prices
        if base_prices is not None:
            base_p = _to_1d_numpy(base_prices, name="base_prices")
            if len(base_p) != n_steps:
                raise ValueError(
                    f"Length mismatch: base_prices ({len(base_p)}) must match actual_returns ({n_steps})."
                )
            if target_prices is not None:
                target_p = _to_1d_numpy(target_prices, name="target_prices")
                if len(target_p) != n_steps:
                    raise ValueError(
                        f"Length mismatch: target_prices ({len(target_p)}) must match actual_returns ({n_steps})."
                    )
            else:
                mult = np.exp(act_r) if is_log_returns else (1.0 + act_r)
                target_p = base_p * mult
        else:
            base_p = np.zeros(n_steps, dtype=np.float64)
            target_p = np.zeros(n_steps, dtype=np.float64)
            current_p = 100.0
            for t in range(n_steps):
                base_p[t] = current_p
                mult = np.exp(act_r[t]) if is_log_returns else (1.0 + act_r[t])
                current_p = current_p * mult
                target_p[t] = current_p

        # Align dates
        if dates is not None:
            if len(dates) != n_steps:
                raise ValueError(
                    f"Length mismatch: dates ({len(dates)}) must match actual_returns ({n_steps})."
                )
            date_col = pd.Index(dates)
        else:
            date_col = pd.RangeIndex(start=0, stop=n_steps, step=1, name="step")

        # Convert actual returns to simple returns for compounding
        act_simple_r = np.exp(act_r) - 1.0 if is_log_returns else act_r.copy()

        # Generate signals
        signals = self.generate_signals(pred_r)

        # Simulation loop arrays
        positions = np.zeros(n_steps, dtype=np.float64)
        turnovers = np.zeros(n_steps, dtype=np.float64)
        fees = np.zeros(n_steps, dtype=np.float64)
        strat_daily_r = np.zeros(n_steps, dtype=np.float64)
        strat_equity = np.zeros(n_steps, dtype=np.float64)
        bnh_daily_r = np.zeros(n_steps, dtype=np.float64)
        bnh_equity = np.zeros(n_steps, dtype=np.float64)
        strat_drawdown = np.zeros(n_steps, dtype=np.float64)
        dollar_fees = np.zeros(n_steps, dtype=np.float64)

        prev_position = 0.0
        current_strat_equity = self.initial_capital
        current_bnh_equity = self.initial_capital
        running_hwm = self.initial_capital

        for t in range(n_steps):
            pos_t = signals[t]
            positions[t] = pos_t

            # Turnover and transaction fee
            turnover_t = abs(pos_t - prev_position)
            turnovers[t] = turnover_t
            fee_t = turnover_t * self.fee_rate
            fees[t] = fee_t
            dollar_fees[t] = current_strat_equity * fee_t

            # Strategy daily return & equity update
            # R_strat = position * R_actual - fee
            r_strat_t = (pos_t * act_simple_r[t]) - fee_t
            strat_daily_r[t] = r_strat_t
            current_strat_equity = current_strat_equity * (1.0 + r_strat_t)
            strat_equity[t] = current_strat_equity

            # Buy-and-Hold benchmark daily return & equity update
            bnh_fee = (self.fee_rate if (t == 0 and self.include_benchmark_fee) else 0.0)
            r_bnh_t = act_simple_r[t] - bnh_fee
            bnh_daily_r[t] = r_bnh_t
            current_bnh_equity = current_bnh_equity * (1.0 + r_bnh_t)
            bnh_equity[t] = current_bnh_equity

            # High-water mark & drawdown tracking
            if current_strat_equity > running_hwm:
                running_hwm = current_strat_equity
            dd_t = (running_hwm - current_strat_equity) / running_hwm if running_hwm > 0.0 else 0.0
            strat_drawdown[t] = dd_t

            prev_position = pos_t

        # Build 14-column audit DataFrame
        audit_df = pd.DataFrame(
            {
                "date": date_col,
                "base_price": base_p,
                "target_price": target_p,
                "actual_return": act_r,
                "predicted_return": pred_r,
                "signal": signals,
                "position": positions,
                "turnover": turnovers,
                "fee": fees,
                "strategy_daily_return": strat_daily_r,
                "strategy_equity": strat_equity,
                "buy_and_hold_daily_return": bnh_daily_r,
                "buy_and_hold_equity": bnh_equity,
                "strategy_drawdown": strat_drawdown,
            }
        )

        # Build equity curve time series
        equity_curve = pd.DataFrame(
            {
                "date": date_col,
                "strategy_equity": strat_equity,
                "buy_and_hold_equity": bnh_equity,
            }
        )

        # Aggregate financial performance metrics
        strat_metrics = evaluate_financial_metrics(
            strat_daily_r,
            equity_curve=strat_equity,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
            is_log_returns=False,
        )

        bh_metrics = evaluate_financial_metrics(
            bnh_daily_r,
            equity_curve=bnh_equity,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
            is_log_returns=False,
        )

        # Directional sign hit rate
        directional_acc = calculate_directional_accuracy(act_r, pred_r)

        total_trades = int(np.sum(turnovers > 0.0))
        total_fees_dollars = float(np.sum(dollar_fees))
        total_fees_pct = float(np.sum(fees) * 100.0)

        summary: Dict[str, Any] = {
            # Strategy KPIs
            "strategy_cumulative_return": strat_metrics["cumulative_return"],
            "strategy_annualized_return": strat_metrics["annualized_return"],
            "strategy_annualized_volatility": strat_metrics["annualized_volatility"],
            "strategy_sharpe_ratio": strat_metrics["sharpe_ratio"],
            "strategy_sortino_ratio": strat_metrics["sortino_ratio"],
            "strategy_max_drawdown": strat_metrics["max_drawdown"],
            "strategy_win_rate": strat_metrics["win_rate"],
            "strategy_profit_factor": strat_metrics["profit_factor"],
            # Base aliases
            "cumulative_return": strat_metrics["cumulative_return"],
            "annualized_return": strat_metrics["annualized_return"],
            "annualized_volatility": strat_metrics["annualized_volatility"],
            "sharpe_ratio": strat_metrics["sharpe_ratio"],
            "sortino_ratio": strat_metrics["sortino_ratio"],
            "max_drawdown": strat_metrics["max_drawdown"],
            "win_rate": strat_metrics["win_rate"],
            "profit_factor": strat_metrics["profit_factor"],
            # Buy & Hold KPIs
            "buy_and_hold_cumulative_return": bh_metrics["cumulative_return"],
            "buy_and_hold_annualized_return": bh_metrics["annualized_return"],
            "buy_and_hold_annualized_volatility": bh_metrics["annualized_volatility"],
            "buy_and_hold_sharpe_ratio": bh_metrics["sharpe_ratio"],
            "buy_and_hold_sortino_ratio": bh_metrics["sortino_ratio"],
            "buy_and_hold_max_drawdown": bh_metrics["max_drawdown"],
            "buy_and_hold_win_rate": bh_metrics["win_rate"],
            "buy_and_hold_profit_factor": bh_metrics["profit_factor"],
            # Alpha / Outperformance
            "outperformance": float(
                strat_metrics["cumulative_return"] - bh_metrics["cumulative_return"]
            ),
            "alpha_cumulative_return": float(
                strat_metrics["cumulative_return"] - bh_metrics["cumulative_return"]
            ),
            "alpha_annualized_return": float(
                strat_metrics["annualized_return"] - bh_metrics["annualized_return"]
            ),
            "alpha_sharpe_ratio": float(
                strat_metrics["sharpe_ratio"] - bh_metrics["sharpe_ratio"]
            ),
            # Trading & Execution Stats
            "total_trades": total_trades,
            "total_turnover": float(np.sum(turnovers)),
            "total_transaction_costs": total_fees_dollars,
            "total_transaction_fees": total_fees_pct,
            "total_fees_pct": total_fees_pct,
            "initial_capital": self.initial_capital,
            "final_strategy_equity": float(strat_equity[-1]),
            "final_buy_and_hold_equity": float(bnh_equity[-1]),
            "directional_accuracy": directional_acc,
        }

        return BacktestResult(
            summary=summary,
            audit_df=audit_df,
            equity_curve=equity_curve,
        )
