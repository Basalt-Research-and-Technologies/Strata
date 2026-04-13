# Basalt Strata

**Strategy testing and evaluation library for Indian equity and derivatives markets.**

[![PyPI](https://img.shields.io/pypi/v/basalt-strata)](https://pypi.org/project/basalt-strata/)
[![Python](https://img.shields.io/pypi/pyversions/basalt-strata)](https://pypi.org/project/basalt-strata/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What this library does

Basalt Strata runs a bar-by-bar backtest on OHLCV price data.  
You bring the strategy logic. The library handles everything else:

- Realistic execution (slippage, commission, lot-based position sizing)
- Drawdown monitoring with auto-stop
- 15+ performance metrics (Sharpe, Sortino, Calmar, VaR, CVaR, win rate, etc.)
- JSON tearsheet output

---

## Installation

```bash
pip install basalt-strata
```

---

## Quick demo (no setup needed)

```bash
pip install basalt-strata
python -m basalt_strata --demo
```

---

## Using basalt-strata as a pip package

> **You only need `pip install basalt-strata`. You do NOT need this codebase.**

If you installed via pip and are writing your own trading system,  
create two files anywhere on your machine:

### File 1 — `my_strategies.py`  ← your strategy logic lives here

```python
# my_strategies.py   (your file, anywhere on your machine)
import pandas as pd

def ema_crossover(df: pd.DataFrame) -> pd.Series:
    """Buy when EMA(10) > EMA(50), sell when below."""
    fast = df["close"].ewm(span=10, adjust=False).mean()
    slow = df["close"].ewm(span=50, adjust=False).mean()
    sig  = pd.Series(0, index=df.index, dtype=int)
    sig[fast > slow] = 1
    sig[fast < slow] = -1
    return sig

def my_rsi_strategy(df: pd.DataFrame) -> pd.Series:
    """RSI(14) mean-reversion."""
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))
    sig   = pd.Series(0, index=df.index, dtype=int)
    sig[rsi < 30] = 1
    sig[rsi > 70] = -1
    return sig.fillna(0).astype(int)

# Rules for any strategy function:
#   - Receives df: pd.DataFrame (columns: open, high, low, close, volume)
#   - Returns pd.Series of int with same index as df
#   - Values must be in {-1, 0, 1}  — no NaN allowed
#   - Never put capital, lots, or commission inside the strategy
```

### File 2 — `run_backtest.py`  ← your runner

```python
# run_backtest.py   (your file, anywhere on your machine)
from basalt_strata import DataFeed, RuleBasedStrategy, Backtest, Timeframe
from my_strategies import ema_crossover   # import from YOUR file

# 1. Load your data
feed = DataFeed.from_csv(
    "nifty_15min.csv",       # your CSV
    symbol="NIFTY",
    timeframe="15min",
    date_col="timestamp",
)

# 2. Wrap your strategy
strategy = RuleBasedStrategy(rule_fn=ema_crossover)

# 3. Run backtest
result = Backtest(
    feed               = feed,
    strategy           = strategy,
    initial_capital    = 1_000_000,
    lot_size           = 50,
    lots_per_trade     = 1,
    slippage_pct       = 0.0005,
    commission         = 20,
    max_drawdown_limit = 0.20,
    risk_free_rate     = 0.065,
    bars_per_year      = Timeframe.MIN15,
).run()

# 4. View results
print(result.summary())
result.to_json("tearsheet.json")
```

**That's the complete workflow.** No codebase, no examples folder — just your  
two files and `pip install basalt-strata`.

---


Your strategy answers **one question only**: buy, sell, or flat?

```
Strategy      -->  WHEN to trade   (your logic, your indicators)
DataFeed      -->  WHAT to trade   (the price data)
Backtest()    -->  HOW to trade    (capital, lots, costs, risk)
```

Capital, lot size, commission, slippage — none of that goes inside your strategy.  
It all lives in the `Backtest()` call. This lets you test the same strategy under  
completely different conditions without touching the strategy code.

---

## Supported timeframes

The library supports **any timeframe** — 1-min, 5-min, 15-min, 30-min, 1-hour, daily, weekly.

The only requirement is that you tell the backtest how many bars are in one year  
so it can annualise metrics correctly. Use the built-in `Timeframe` helper:

```python
from basalt_strata import Timeframe

Timeframe.MIN1    # 94,500  — 1-minute bars  (NSE session = 375 bars/day)
Timeframe.MIN5    # 18,900  — 5-minute bars
Timeframe.MIN15   #  6,300  — 15-minute bars
Timeframe.MIN30   #  3,024  — 30-minute bars
Timeframe.HOUR1   #  1,512  — 1-hour bars
Timeframe.DAILY   #    252  — daily bars
Timeframe.WEEKLY  #     52  — weekly bars

# Or compute for any custom bar size:
Timeframe.custom(minutes_per_bar=3)   # 3-min bars --> 31,500
```

---

## Step-by-step usage

### Step 1 — Load your data

```python
from basalt_strata import DataFeed

# From a CSV file
feed = DataFeed.from_csv(
    "nifty_15min.csv",
    symbol="NIFTY",
    timeframe="15min",
    date_col="timestamp",    # name of your timestamp column
)

# From a pandas DataFrame you already have
feed = DataFeed.from_dataframe(df, symbol="NIFTY", timeframe="1min")

# Resample to a lower frequency
daily_feed = feed.resample("1D")
```

**CSV requirements:**
- Must have columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`
- Timestamps can be tz-aware (IST) or tz-naive (assumed IST)
- Duplicate timestamps → error. NaN in OHLC → error. Zero volume → warning only.

---

### Step 2 — Write your strategy

Open **`examples/my_strategies.py`** and add your function there.  
A strategy function receives the OHLCV DataFrame and returns a signal Series.

```python
import pandas as pd

def my_strategy(df: pd.DataFrame) -> pd.Series:
    """
    df has columns: open, high, low, close, volume, symbol, timeframe
    Return a Series of: 1 (buy), -1 (sell), 0 (flat/hold)
    """
    signal = pd.Series(0, index=df.index, dtype=int)

    # --- your logic here ---
    ema10 = df["close"].ewm(span=10, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()

    signal[ema10 > ema50] = 1    # buy when fast EMA above slow
    signal[ema10 < ema50] = -1   # sell when fast EMA below slow

    return signal

# Rules:
# - Return a pd.Series with the same index as df
# - Values must be strictly in {-1, 0, 1}
# - No NaN values allowed
# - Never put capital, lots, or commission inside here
```

**Ready-made strategies in `examples/my_strategies.py`:**

| Name | Description | Best timeframe |
|---|---|---|
| `ema_crossover` | EMA(fast) / EMA(slow) trend following | Daily, 15-min |
| `rsi_mean_reversion` | RSI oversold/overbought | 15-min, hourly |
| `bollinger_breakout` | Bollinger Band breakout | Any |
| `vwap_reversion` | VWAP mean-reversion (resets daily) | Intraday |
| `dual_momentum` | EMA trend filter + RSI entry | Any |
| `intraday_ema_crossover` | Session-aware EMA, exits at 15:20 | 1-min, 5-min |

---

### Step 3 — Run the backtest

```python
from basalt_strata import DataFeed, RuleBasedStrategy, Backtest, Timeframe

feed     = DataFeed.from_csv("data.csv", symbol="NIFTY", timeframe="15min")
strategy = RuleBasedStrategy(rule_fn=my_strategy)

result = Backtest(
    feed               = feed,
    strategy           = strategy,
    initial_capital    = 1_000_000,        # starting capital in INR
    lot_size           = 50,               # units per lot
    lots_per_trade     = 1,                # static lots per signal
    slippage_pct       = 0.0005,           # 0.05% slippage on fill price
    commission         = 20,               # flat INR per order (entry + exit separately)
    max_drawdown_limit = 0.20,             # auto-stop if drawdown > 20%
    risk_free_rate     = 0.065,            # 6.5% Indian T-bill for Sharpe/Sortino
    bars_per_year      = Timeframe.MIN15,  # use Timeframe constant for your data
).run()
```

**All Backtest parameters:**

| Parameter | What it controls | Example |
|---|---|---|
| `initial_capital` | Starting capital (INR) | `1_000_000` |
| `lot_size` | Units per lot | `50` (Nifty standard) |
| `lots_per_trade` | Static lots per signal | `2` |
| `position_sizing_fn` | Dynamic sizing function (overrides lots_per_trade) | see below |
| `slippage_pct` | Slippage as fraction of fill price | `0.0005` (0.05%) |
| `commission` | Flat INR per order (entry + exit each count) | `20` |
| `max_drawdown_limit` | Auto-stop if drawdown exceeds this | `0.20` (20%) |
| `risk_free_rate` | Annual rate for Sharpe/Sortino | `0.065` |
| `benchmark` | Optional price series for alpha calculation | Nifty 50 series |
| `bars_per_year` | Bars per year for annualisation | `Timeframe.MIN15` |

---

### Step 4 — Read the results

```python
# Print tearsheet
print(result.summary())

# Save to file (goes to outputs/ by default in examples)
result.to_json("outputs/my_run.json")

# Access individual metrics
m = result.metrics
print(m["total_return_pct"])       # e.g. 18.5
print(m["sharpe_ratio"])
print(m["max_drawdown_pct"])
print(m["win_rate_pct"])
print(m["brt_thresholds"])         # {"sharpe_pass": True, "calmar_pass": False, ...}

# Access trades
for trade in result.trades:
    print(trade.entry_time, trade.direction, trade.net_pnl)

# Access equity curve
print(result.equity_curve)         # pd.Series indexed by timestamp
```

**All metrics returned:**

| Category | Metrics |
|---|---|
| Returns | `total_return_pct`, `cagr_pct`, `final_equity`, `monthly_pnl`, `benchmark_alpha_pct` |
| Risk | `max_drawdown_pct`, `volatility_annualised_pct`, `var_95_pct`, `var_99_pct`, `cvar_95_pct` |
| Ratios | `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `omega_ratio`, `profit_factor` |
| Trades | `total_trades`, `win_rate_pct`, `avg_win`, `avg_loss`, `payoff_ratio`, `max_consecutive_wins/losses` |
| BRT check | `brt_thresholds` → `{sharpe_pass, calmar_pass, max_dd_pass, brt_pass}` |

---

## Dynamic position sizing

Replace static `lots_per_trade` with a function that sizes positions based on current equity:

```python
def risk_1pct(equity: float, price: float, lot_size: int) -> int:
    """Risk 1% of current equity per trade."""
    return max(1, int((equity * 0.01) / (price * lot_size)))

result = Backtest(
    feed=feed,
    strategy=strategy,
    initial_capital=1_000_000,
    lot_size=50,
    position_sizing_fn=risk_1pct,   # overrides lots_per_trade
    commission=20,
    bars_per_year=Timeframe.MIN15,
).run()
```

---

## Sensitivity analysis — same strategy, different configs

```python
# Two capital sizes
result_small = Backtest(feed, strategy, initial_capital=500_000,   lots_per_trade=1, bars_per_year=Timeframe.MIN15).run()
result_large = Backtest(feed, strategy, initial_capital=5_000_000, lots_per_trade=5, bars_per_year=Timeframe.MIN15).run()

# Two brokers
result_zerodha = Backtest(feed, strategy, commission=20, slippage_pct=0.0005, bars_per_year=Timeframe.MIN15).run()
result_upstox  = Backtest(feed, strategy, commission=0,  slippage_pct=0.0003, bars_per_year=Timeframe.MIN15).run()
```

---

## Running from the command line

Install first:
```bash
pip install basalt-strata
```

```bash
# Show version
basalt-strata --version

# Run a synthetic demo (no data file needed)
basalt-strata --demo

# Custom demo
basalt-strata --demo --capital 2000000 --lots 2 --bars 500 --no-file

# All options
basalt-strata --help
```

---

## Testing the library

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all 51 tests
pytest test_e2e.py -v

# Run with custom parameters (all values are overrideable)
pytest test_e2e.py -v --capital 500000 --lots 1 --lot-size 25
pytest test_e2e.py -v --bars 1000 --seed 99 --slippage 0.001
pytest test_e2e.py -v --max-dd 0.15 --bars-per-year 252
```

---

## Testing on real data (examples/)

```bash
cd "d:\BRT\Tech\files (1)"

# Default: EMA crossover on 2024 data, output saved to outputs/
py examples/run_real_data.py

# Custom date range
py examples/run_real_data.py --from 2023-01-01 --to 2023-12-31

# Different strategy
py examples/run_real_data.py --strategy rsi
py examples/run_real_data.py --strategy intraday    # best for 1-min data

# Full custom config
py examples/run_real_data.py \
    --from 2022-01-01 --to 2022-12-31 \
    --strategy ema \
    --capital 2000000 --lot-size 50 --lots 2 \
    --slippage 0.0005 --commission 20 \
    --max-dd 0.15 --rfr 0.065 \
    --bars-per-year 94500

# Output is auto-saved to: outputs/<strategy>_<from>_<to>.json
```

**Available strategies in `examples/run_real_data.py`:**
`ema`, `rsi`, `bollinger`, `vwap`, `dual`, `intraday`, `ema_5_20`, `ema_20_50`, `rsi_tight`

**To add your own strategy:**
1. Write your function in `examples/my_strategies.py`
2. Add it to the `STRATEGIES` dict in `examples/run_real_data.py`
3. Run with `--strategy your_name`

---

## Project structure

```
basalt_strata/           <- the library (pip install basalt-strata)
    __init__.py          <- public API
    __main__.py          <- python -m basalt_strata CLI
    datafeed.py          <- data loading & validation
    strategy.py          <- Strategy base class + RuleBasedStrategy
    execution.py         <- bar-by-bar trade simulation
    analytics.py         <- performance metrics
    backtest.py          <- orchestrator + BacktestResult
    timeframes.py        <- Timeframe constants (MIN1, MIN15, DAILY...)
    py.typed             <- PEP 561 type marker

examples/                <- your workspace (not part of the pip package)
    my_strategies.py     <- WHERE YOU WRITE YOUR STRATEGIES
    run_backtest.py      <- runner for synthetic data
    run_real_data.py     <- runner for real Nifty CSV data

outputs/                 <- all tearsheet JSON files saved here

test_e2e.py              <- full pytest test suite (51 tests)
conftest.py              <- pytest CLI options (--capital, --lots, etc.)
pyproject.toml           <- build + metadata config
README.md                <- this file
```

---

## Publishing

```bash
pip install build twine
python -m build
twine upload dist/*
```

---

## License

MIT — Basalt Research & Technologies.
