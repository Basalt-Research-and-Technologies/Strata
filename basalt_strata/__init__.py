"""
Basalt Strata
=============
Strategy testing and evaluation library for Indian equity and derivatives markets.

Quick start
-----------
>>> from basalt_strata import DataFeed, RuleBasedStrategy, Backtest
>>> import pandas as pd
>>>
>>> feed = DataFeed.from_csv("nifty_15min.csv", symbol="NIFTY", timeframe="15min")
>>>
>>> def my_ema_crossover(df):
...     signal = pd.Series(0, index=df.index)
...     signal[df["ema10"] > df["ema50"]] = 1
...     signal[df["ema10"] < df["ema50"]] = -1
...     return signal
>>>
>>> strategy = RuleBasedStrategy(rule_fn=my_ema_crossover)
>>>
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
>>>
>>> print(result.summary())
>>> result.to_json("tearsheet.json")
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------
__version__ = "1.0.0"
__author__ = "Basalt Research & Technologies"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------
from .datafeed import DataFeed, DataFeedError

# ---------------------------------------------------------------------------
# Strategy layer
# ---------------------------------------------------------------------------
from .strategy import Strategy, RuleBasedStrategy, StrategyError

# ---------------------------------------------------------------------------
# Execution layer
# ---------------------------------------------------------------------------
from .execution import Execution, Trade

# ---------------------------------------------------------------------------
# Analytics layer
# ---------------------------------------------------------------------------
from .analytics import Analytics

# ---------------------------------------------------------------------------
# Backtest orchestrator
# ---------------------------------------------------------------------------
from .backtest import Backtest, BacktestResult, BacktestError

# ---------------------------------------------------------------------------
# Timeframe helpers
# ---------------------------------------------------------------------------
from .timeframes import Timeframe, TF

# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------
__all__ = [
    # data
    "DataFeed",
    "DataFeedError",
    # strategy
    "Strategy",
    "RuleBasedStrategy",
    "StrategyError",
    # execution
    "Execution",
    "Trade",
    # analytics
    "Analytics",
    # backtest
    "Backtest",
    "BacktestResult",
    "BacktestError",
    # timeframes
    "Timeframe",
    "TF",
    # meta
    "__version__",
]
