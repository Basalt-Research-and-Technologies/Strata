"""
Backtest -- orchestrates the full pipeline and returns a BacktestResult.

Pipeline:
    1. DataFeed (validated externally, passed in)
    2. Strategy.get_signals(feed)          -> pd.Series
    3. Execution.run()                     -> trades, equity_curve, early_stop
    4. Analytics.compute()                 -> metrics dict
    5. BacktestResult wraps everything     -> .to_json() / .to_dict() / .summary()

Usage:
    result = Backtest(feed, strategy, initial_capital=1_000_000).run()
    result.to_json("results.json")
    print(result.summary())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Union

import pandas as pd

from .datafeed import DataFeed
from .strategy import Strategy
from .execution import Execution, Trade
from .analytics import Analytics


class BacktestError(Exception):
    pass


@dataclass
class BacktestResult:
    """
    Holds the full output of a completed backtest.

    Attributes
    ----------
    metrics : dict
        All computed performance metrics.
    trades : list[Trade]
        Every simulated trade with entry/exit/costs/P&L.
    equity_curve : pd.Series
        Equity value at each bar.
    config : dict
        Snapshot of the backtest parameters used.
    early_stop : bool
        True if the backtest was halted by max_drawdown_limit.
    """
    metrics: dict
    trades: list[Trade]
    equity_curve: pd.Series
    config: dict
    early_stop: bool

    # ------------------------------------------------------------------
    # Output formats
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return the full result as a plain Python dictionary."""
        return {
            "config": self.config,
            "early_stop": self.early_stop,
            "metrics": self.metrics,
            "trade_log": [t.to_dict() for t in self.trades],
            "equity_curve": {
                ts.isoformat(): round(v, 2)
                for ts, v in self.equity_curve.items()
            },
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """
        Serialise the full result to JSON.

        Parameters
        ----------
        path : str or Path or None
            If provided, writes to file and returns the JSON string.
            If None, returns the JSON string only.
        indent : int
            JSON indentation level.

        Returns
        -------
        str : JSON string
        """
        data = self.to_dict()
        json_str = json.dumps(data, indent=indent, default=str)
        if path is not None:
            Path(path).write_text(json_str, encoding="utf-8")
        return json_str

    def summary(self) -> str:
        """Print a concise human-readable summary of key metrics."""
        m = self.metrics
        thr = m.get("brt_thresholds", {})

        lines = [
            "=" * 52,
            "  Basalt Strata -- Backtest Summary",
            "=" * 52,
        ]

        if self.early_stop:
            lines.append("  !! EARLY STOP: max drawdown limit hit")

        def _fmt(val, suffix=""):
            return f"{val}{suffix}" if val is not None else "n/a"

        rows = [
            ("Total return",       _fmt(m.get("total_return_pct"), "%")),
            ("CAGR",               _fmt(m.get("cagr_pct"), "%")),
            ("Max drawdown",       _fmt(m.get("max_drawdown_pct"), "%")),
            ("Volatility (ann.)",  _fmt(m.get("volatility_annualised_pct"), "%")),
            ("Sharpe ratio",       _fmt(m.get("sharpe_ratio"))),
            ("Sortino ratio",      _fmt(m.get("sortino_ratio"))),
            ("Calmar ratio",       _fmt(m.get("calmar_ratio"))),
            ("Profit factor",      _fmt(m.get("profit_factor"))),
            ("Win rate",           _fmt(m.get("win_rate_pct"), "%")),
            ("Total trades",       _fmt(m.get("total_trades"))),
            ("Final equity",       _fmt(m.get("final_equity"), " INR")),
        ]
        for label, val in rows:
            lines.append(f"  {label:<26} {val}")

        lines.append("-" * 52)
        lines.append(
            f"  BRT pass: "
            f"Sharpe={'PASS' if thr.get('sharpe_pass') else 'FAIL'}  "
            f"Calmar={'PASS' if thr.get('calmar_pass') else 'FAIL'}  "
            f"MaxDD={'PASS' if thr.get('max_dd_pass') else 'FAIL'}"
        )
        lines.append("=" * 52)
        return "\n".join(lines)


class Backtest:
    """
    Orchestrates the full strategy evaluation pipeline.

    Parameters
    ----------
    feed : DataFeed
    strategy : Strategy
    initial_capital : float
        Starting capital in INR. Default 1_000_000 (₹10 lakh).
    lot_size : int
        Units per lot. Default 50 (Nifty standard).
    lots_per_trade : int
        Number of lots per trade signal. Default 1. Ignored when
        position_sizing_fn is provided.
    position_sizing_fn : Callable or None
        Custom function for dynamic position sizing. Signature::

            def my_sizer(equity: float, price: float, lot_size: int) -> int:
                ...

        When provided it is called before every entry and its return value
        (number of lots) overrides ``lots_per_trade``. If None, the static
        ``lots_per_trade`` value is used.
    slippage_pct : float
        Slippage as decimal fraction of price. Default 0.0005 (0.05%).
    commission : float
        Flat INR per order (entry and exit each count separately). Default 20.
    max_drawdown_limit : float or None
        Stop the backtest if drawdown from peak exceeds this fraction.
        e.g. 0.20 = stop at 20% drawdown. Default None (no limit).
    risk_free_rate : float
        Annual risk-free rate for Sharpe/Sortino. Default 0.065.
    benchmark : pd.Series or None
        Price series for benchmark alpha calculation.
    bars_per_year : int
        Used for annualising metrics. Default 252 (daily).
        For 15-min NSE bars use 252 * 25 = 6300.

    Example
    -------
    >>> result = Backtest(
    ...     feed=feed,
    ...     strategy=strategy,
    ...     initial_capital=1_000_000,
    ...     lot_size=50,
    ...     lots_per_trade=2,
    ...     slippage_pct=0.0005,
    ...     commission=20,
    ...     max_drawdown_limit=0.20,
    ...     risk_free_rate=0.065,
    ... ).run()
    >>> result.to_json("tearsheet.json")
    >>> print(result.summary())
    """

    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        initial_capital: float = 1_000_000,
        lot_size: int = 50,
        lots_per_trade: int = 1,
        position_sizing_fn: Optional[Callable[[float, float, int], int]] = None,
        slippage_pct: float = 0.0005,
        commission: float = 20.0,
        max_drawdown_limit: Optional[float] = None,
        risk_free_rate: float = 0.065,
        benchmark: Optional[pd.Series] = None,
        bars_per_year: int = 252,
    ) -> None:
        self.feed = feed
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.lot_size = lot_size
        self.lots_per_trade = lots_per_trade
        self.position_sizing_fn = position_sizing_fn
        self.slippage_pct = slippage_pct
        self.commission = commission
        self.max_drawdown_limit = max_drawdown_limit
        self.risk_free_rate = risk_free_rate
        self.benchmark = benchmark
        self.bars_per_year = bars_per_year

    def run(self) -> BacktestResult:
        """Execute the full pipeline and return a BacktestResult."""

        # Step 1: generate and validate signals
        signals = self.strategy.get_signals(self.feed)

        # Step 2: simulate execution
        executor = Execution(
            data=self.feed.data,
            signals=signals,
            lot_size=self.lot_size,
            lots_per_trade=self.lots_per_trade,
            position_sizing_fn=self.position_sizing_fn,
            slippage_pct=self.slippage_pct,
            commission=self.commission,
            initial_capital=self.initial_capital,
            max_drawdown_limit=self.max_drawdown_limit,
        )
        trades, equity_curve, early_stop = executor.run()

        # Step 3: compute analytics
        analytics = Analytics(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=self.initial_capital,
            risk_free_rate=self.risk_free_rate,
            benchmark=self.benchmark,
            bars_per_year=self.bars_per_year,
        )
        metrics = analytics.compute()

        # Step 4: bundle config snapshot
        config = {
            "symbol": self.feed.symbol,
            "timeframe": self.feed.timeframe,
            "bars": len(self.feed),
            "from": self.feed.data.index[0].isoformat(),
            "to": self.feed.data.index[-1].isoformat(),
            "initial_capital": self.initial_capital,
            "lot_size": self.lot_size,
            "lots_per_trade": self.lots_per_trade,
            "position_sizing_fn": (
                getattr(self.position_sizing_fn, "__name__", repr(self.position_sizing_fn))
                if self.position_sizing_fn is not None
                else None
            ),
            "slippage_pct": self.slippage_pct,
            "commission": self.commission,
            "max_drawdown_limit": self.max_drawdown_limit,
            "risk_free_rate": self.risk_free_rate,
            "bars_per_year": self.bars_per_year,
            "strategy": type(self.strategy).__name__,
        }

        return BacktestResult(
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            config=config,
            early_stop=early_stop,
        )
