"""
Analytics -- computes all performance metrics from the trade log + equity curve.

Metric categories (matching Section 5 of the spec doc):
    1. Returns metrics
    2. Risk metrics
    3. Quality ratios
    4. Trade-level metrics

Edge case handling:
    - Zero trades: all trade-level metrics return None.
    - Zero downside deviation (all-winning): Sortino returns None.
    - Division by zero in any ratio: returns None (not 0, not inf).
    - Backtest shorter than rolling window: rolling metrics return empty.

BRT thresholds (Section 10):
    Sharpe >= 1.5
    Calmar >= 1.0
    Max drawdown < 20%
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from .execution import Trade


class Analytics:
    """
    Computes institutional-grade performance metrics.

    Parameters
    ----------
    trades : list[Trade]
    equity_curve : pd.Series
    initial_capital : float
    risk_free_rate : float
        Annual risk-free rate as decimal. Default 0.065 (6.5% Indian T-bill).
    benchmark : pd.Series or None
        Buy-and-hold benchmark price series for alpha calculation.
    bars_per_year : int
        Number of bars in a year for annualisation. Caller should set this
        based on timeframe. Default 252 (daily). For 15-min NSE data use
        252 * 25 = 6300 (approx 6.5hr session / 25 fifteen-min bars).
    """

    BRT_SHARPE_MIN = 1.5
    BRT_CALMAR_MIN = 1.0
    BRT_MAX_DD_MAX = 0.20

    def __init__(
        self,
        trades: list[Trade],
        equity_curve: pd.Series,
        initial_capital: float,
        risk_free_rate: float = 0.065,
        benchmark: Optional[pd.Series] = None,
        bars_per_year: int = 252,
    ) -> None:
        self.trades = trades
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.benchmark = benchmark
        self.bars_per_year = bars_per_year

        self._returns: pd.Series = equity_curve.pct_change().dropna()
        self._pnl_series: list[float] = [t.net_pnl for t in trades]

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def compute(self) -> dict:
        """Return the full metrics dictionary."""
        m: dict = {}
        m.update(self._returns_metrics())
        m.update(self._risk_metrics())
        m.update(self._quality_ratios())
        m.update(self._trade_metrics())
        m.update(self._brt_thresholds(m))
        return m

    # ------------------------------------------------------------------
    # 1. Returns metrics
    # ------------------------------------------------------------------

    def _returns_metrics(self) -> dict:
        ec = self.equity_curve
        if len(ec) < 2:
            return {"total_return_pct": None, "cagr_pct": None}

        total_return = (ec.iloc[-1] - self.initial_capital) / self.initial_capital
        n_bars = len(ec)
        years = n_bars / self.bars_per_year
        cagr = (ec.iloc[-1] / self.initial_capital) ** (1 / years) - 1 if years > 0 else None

        # Monthly P&L summary
        monthly = ec.resample("ME").last().pct_change().dropna()
        monthly_pnl = {
            str(ts.to_period("M")): round(v * 100, 2)
            for ts, v in monthly.items()
        }

        # Benchmark alpha
        alpha = None
        if self.benchmark is not None:
            bench_return = (
                self.benchmark.iloc[-1] / self.benchmark.iloc[0] - 1
            )
            alpha = total_return - bench_return

        return {
            "total_return_pct": round(total_return * 100, 4),
            "cagr_pct": round(cagr * 100, 4) if cagr is not None else None,
            "monthly_pnl": monthly_pnl,
            "benchmark_alpha_pct": round(alpha * 100, 4) if alpha is not None else None,
            "final_equity": round(self.equity_curve.iloc[-1], 2),
        }

    # ------------------------------------------------------------------
    # 2. Risk metrics
    # ------------------------------------------------------------------

    def _risk_metrics(self) -> dict:
        returns = self._returns
        ec = self.equity_curve

        # Volatility (annualised)
        vol = returns.std() * math.sqrt(self.bars_per_year) if len(returns) > 1 else None

        # Max drawdown
        running_max = ec.cummax()
        drawdown = (ec - running_max) / running_max
        max_dd = drawdown.min()  # most negative value

        # VaR (historical, not parametric)
        var_95 = float(returns.quantile(0.05)) if len(returns) >= 20 else None
        var_99 = float(returns.quantile(0.01)) if len(returns) >= 100 else None

        # CVaR / Expected Shortfall
        cvar_95 = (
            float(returns[returns <= (var_95 or 0)].mean())
            if var_95 is not None and len(returns[returns <= var_95]) > 0
            else None
        )

        return {
            "max_drawdown_pct": round(max_dd * 100, 4),
            "volatility_annualised_pct": round(vol * 100, 4) if vol is not None else None,
            "var_95_pct": round(var_95 * 100, 4) if var_95 is not None else None,
            "var_99_pct": round(var_99 * 100, 4) if var_99 is not None else None,
            "cvar_95_pct": round(cvar_95 * 100, 4) if cvar_95 is not None else None,
        }

    # ------------------------------------------------------------------
    # 3. Quality ratios
    # ------------------------------------------------------------------

    def _quality_ratios(self) -> dict:
        returns = self._returns
        n = len(returns)

        rf_per_bar = (1 + self.risk_free_rate) ** (1 / self.bars_per_year) - 1
        excess = returns - rf_per_bar

        # Sharpe
        if n > 1 and returns.std() > 0:
            sharpe = float(excess.mean() / returns.std() * math.sqrt(self.bars_per_year))
        else:
            sharpe = None

        # Sortino (penalise only downside deviation)
        downside = returns[returns < rf_per_bar]
        if len(downside) > 1 and downside.std() > 0:
            sortino = float(excess.mean() / downside.std() * math.sqrt(self.bars_per_year))
        else:
            sortino = None

        # Calmar
        ec = self.equity_curve
        running_max = ec.cummax()
        max_dd = float((((ec - running_max) / running_max).min()))
        n_bars = len(ec)
        years = n_bars / self.bars_per_year
        cagr = (ec.iloc[-1] / self.initial_capital) ** (1 / years) - 1 if years > 0 else None
        calmar = float(cagr / abs(max_dd)) if (cagr is not None and max_dd < 0) else None

        # Omega ratio (threshold = risk-free per bar)
        gains = returns[returns > rf_per_bar] - rf_per_bar
        losses = rf_per_bar - returns[returns <= rf_per_bar]
        omega = float(gains.sum() / losses.sum()) if losses.sum() > 0 else None

        # Trade-based ratios
        if self._pnl_series:
            wins = [p for p in self._pnl_series if p > 0]
            losses_t = [p for p in self._pnl_series if p < 0]

            win_rate = len(wins) / len(self._pnl_series) if self._pnl_series else None
            avg_win = float(np.mean(wins)) if wins else None
            avg_loss = float(np.mean(losses_t)) if losses_t else None
            payoff = abs(avg_win / avg_loss) if (avg_win and avg_loss) else None

            gross_profit = sum(wins)
            gross_loss = abs(sum(losses_t))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
        else:
            win_rate = avg_win = avg_loss = payoff = profit_factor = None

        return {
            "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
            "sortino_ratio": round(sortino, 4) if sortino is not None else None,
            "calmar_ratio": round(calmar, 4) if calmar is not None else None,
            "omega_ratio": round(omega, 4) if omega is not None else None,
            "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
            "win_rate_pct": round(win_rate * 100, 2) if win_rate is not None else None,
            "avg_win": round(avg_win, 2) if avg_win is not None else None,
            "avg_loss": round(avg_loss, 2) if avg_loss is not None else None,
            "payoff_ratio": round(payoff, 4) if payoff is not None else None,
        }

    # ------------------------------------------------------------------
    # 4. Trade-level metrics
    # ------------------------------------------------------------------

    def _trade_metrics(self) -> dict:
        if not self.trades:
            return {
                "total_trades": 0,
                "avg_trade_duration_bars": None,
                "recovery_factor": None,
                "max_consecutive_wins": None,
                "max_consecutive_losses": None,
            }

        total = len(self.trades)

        # Duration in bars (approximate using timestamp delta -- not bar count)
        durations = []
        for t in self.trades:
            delta = t.exit_time - t.entry_time
            durations.append(delta.total_seconds() / 60)  # in minutes
        avg_duration_min = float(np.mean(durations)) if durations else None

        # Recovery factor
        ec = self.equity_curve
        running_max = ec.cummax()
        max_dd_abs = float((running_max - ec).max())
        total_return_abs = ec.iloc[-1] - self.initial_capital
        recovery = (total_return_abs / max_dd_abs) if max_dd_abs > 0 else None

        # Consecutive wins/losses
        outcomes = [1 if p > 0 else -1 for p in self._pnl_series]
        max_wins = self._max_streak(outcomes, 1)
        max_losses = self._max_streak(outcomes, -1)

        return {
            "total_trades": total,
            "avg_trade_duration_minutes": round(avg_duration_min, 1) if avg_duration_min else None,
            "recovery_factor": round(recovery, 4) if recovery is not None else None,
            "max_consecutive_wins": max_wins,
            "max_consecutive_losses": max_losses,
        }

    # ------------------------------------------------------------------
    # BRT threshold check
    # ------------------------------------------------------------------

    def _brt_thresholds(self, metrics: dict) -> dict:
        sharpe = metrics.get("sharpe_ratio")
        calmar = metrics.get("calmar_ratio")
        max_dd = metrics.get("max_drawdown_pct")

        results = {
            "sharpe_pass": bool(sharpe >= self.BRT_SHARPE_MIN) if sharpe is not None else None,
            "calmar_pass": bool(calmar >= self.BRT_CALMAR_MIN) if calmar is not None else None,
            "max_dd_pass": bool(abs(max_dd) < self.BRT_MAX_DD_MAX * 100) if max_dd is not None else None,
        }
        passes = [v for v in results.values() if v is not None]
        results["brt_pass"] = all(passes) if passes else None
        return {"brt_thresholds": results}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _max_streak(outcomes: list[int], value: int) -> int:
        max_s = cur = 0
        for v in outcomes:
            if v == value:
                cur += 1
                max_s = max(max_s, cur)
            else:
                cur = 0
        return max_s
