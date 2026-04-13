"""
demo/run_demo.py
================
Self-contained end-to-end demo for basalt-strata.

Just run:
    py demo/run_demo.py

No data file needed. No configuration needed.
Everything runs, all outputs are saved inside demo/outputs/.

What this demo does:
  1. Generates synthetic Nifty-like OHLCV data
  2. Runs three strategies (EMA crossover, RSI, Dual Momentum)
  3. Compares results side by side
  4. Saves individual JSON tearsheets to demo/outputs/
  5. Saves a combined comparison summary to demo/outputs/comparison.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# The library is the ONLY external dependency — pip install basalt-strata
# ---------------------------------------------------------------------------
try:
    from basalt_strata import (
        DataFeed,
        RuleBasedStrategy,
        Backtest,
        Timeframe,
        __version__,
    )
except ImportError:
    print("\n  [ERROR] basalt-strata is not installed.")
    print("  Run:  pip install basalt-strata\n")
    sys.exit(1)

# ---------------------------------------------------------------------------
# All outputs stay inside demo/outputs/
# ---------------------------------------------------------------------------
DEMO_DIR    = Path(__file__).resolve().parent
OUTPUTS_DIR = DEMO_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# STEP 1 — Synthetic data generator (no CSV needed)
# ===========================================================================

def generate_data(n_bars: int = 756, seed: int = 42) -> DataFeed:
    """
    Generate 3 years of synthetic Nifty-like daily OHLCV data.
    Uses a random walk with slight upward drift.
    """
    rng    = np.random.default_rng(seed)
    close  = 18_000.0
    closes = [close]
    for _ in range(n_bars - 1):
        close *= 1 + rng.normal(0.0004, 0.010)   # slight upward drift
        closes.append(close)

    closes = np.array(closes)
    noise  = rng.uniform(0.002, 0.008, n_bars)
    opens  = closes * (1 + rng.uniform(-0.005, 0.005, n_bars))
    highs  = np.maximum(closes, opens) * (1 + noise)
    lows   = np.minimum(closes, opens) * (1 - noise)
    vols   = rng.integers(100_000, 2_000_000, n_bars).astype(float)

    idx = pd.bdate_range(
        end=pd.Timestamp.today().normalize(),
        periods=n_bars,
        freq="B",
    )
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )
    return DataFeed.from_dataframe(df, symbol="NIFTY-DEMO", timeframe="1D")


# ===========================================================================
# STEP 2 — Strategy definitions (all self-contained in this file)
# ===========================================================================

def ema_crossover(df: pd.DataFrame) -> pd.Series:
    """EMA(10) above/below EMA(50) trend following."""
    fast = df["close"].ewm(span=10, adjust=False).mean()
    slow = df["close"].ewm(span=50, adjust=False).mean()
    sig  = pd.Series(0, index=df.index, dtype=int)
    sig[fast > slow] = 1
    sig[fast < slow] = -1
    return sig


def rsi_reversion(df: pd.DataFrame) -> pd.Series:
    """RSI(14) oversold (<30) buy, overbought (>70) sell."""
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))
    sig   = pd.Series(0, index=df.index, dtype=int)
    sig[rsi < 30] = 1
    sig[rsi > 70] = -1
    return sig.fillna(0).astype(int)


def dual_momentum(df: pd.DataFrame) -> pd.Series:
    """EMA(200) trend filter + RSI(14) entry trigger."""
    trend = df["close"].ewm(span=200, adjust=False).mean()
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rsi   = 100 - (100 / (1 + gain / loss.replace(0, float("nan"))))
    sig   = pd.Series(0, index=df.index, dtype=int)
    sig[(df["close"] > trend) & (rsi < 40)]  = 1
    sig[(df["close"] < trend) & (rsi > 60)]  = -1
    return sig.fillna(0).astype(int)


# ===========================================================================
# STEP 3 — Run a single backtest and return result
# ===========================================================================

BACKTEST_CONFIG = dict(
    initial_capital    = 1_000_000,   # INR 10 lakh
    lot_size           = 50,          # Nifty standard
    lots_per_trade     = 1,
    slippage_pct       = 0.0005,      # 0.05%
    commission         = 20,          # INR per order
    max_drawdown_limit = 0.25,        # stop at 25% drawdown
    risk_free_rate     = 0.065,       # 6.5% Indian T-bill
    bars_per_year      = Timeframe.DAILY,
)


def run_strategy(name: str, fn, feed: DataFeed):
    strategy = RuleBasedStrategy(rule_fn=fn)
    result   = Backtest(feed=feed, strategy=strategy, **BACKTEST_CONFIG).run()

    # Save individual tearsheet
    out_path = OUTPUTS_DIR / f"{name}.json"
    result.to_json(str(out_path))

    return result, out_path


# ===========================================================================
# STEP 4 — Print and compare
# ===========================================================================

def _row(label, value):
    return f"  {label:<28} {value}"


def print_comparison(results: dict):
    names = list(results.keys())
    SEP   = "=" * (30 + 22 * len(names))

    print("\n" + SEP)
    print(f"  {'Metric':<28}" + "".join(f"  {n:<20}" for n in names))
    print(SEP)

    metrics_to_show = [
        ("Total return %",      "total_return_pct"),
        ("CAGR %",              "cagr_pct"),
        ("Max drawdown %",      "max_drawdown_pct"),
        ("Sharpe ratio",        "sharpe_ratio"),
        ("Sortino ratio",       "sortino_ratio"),
        ("Calmar ratio",        "calmar_ratio"),
        ("Profit factor",       "profit_factor"),
        ("Win rate %",          "win_rate_pct"),
        ("Total trades",        "total_trades"),
        ("Final equity INR",    "final_equity"),
    ]

    for label, key in metrics_to_show:
        row = f"  {label:<28}"
        for name in names:
            val = results[name].metrics.get(key)
            row += f"  {str(val):<20}"
        print(row)

    print("-" * (30 + 22 * len(names)))
    for name in names:
        thr = results[name].metrics.get("brt_thresholds", {})
        es  = "EARLY-STOP" if results[name].early_stop else "completed"
        print(f"  {name:<28}  BRT={thr.get('brt_pass')}  Run={es}")
    print(SEP + "\n")


# ===========================================================================
# MAIN — runs everything automatically
# ===========================================================================

def main():
    print("\n" + "=" * 60)
    print(f"  Basalt Strata Demo  v{__version__}")
    print("=" * 60)

    # 1. Data
    print("\n[1/4] Generating synthetic Nifty data (3 years daily)...")
    feed = generate_data(n_bars=756, seed=42)
    print(f"      {feed}")

    # 2. Run all three strategies
    strategies = {
        "ema_crossover": ema_crossover,
        "rsi_reversion": rsi_reversion,
        "dual_momentum": dual_momentum,
    }

    print("\n[2/4] Running backtests...")
    all_results = {}
    for name, fn in strategies.items():
        result, path = run_strategy(name, fn, feed)
        all_results[name] = result
        status = "EARLY-STOP" if result.early_stop else "completed"
        trades = result.metrics.get("total_trades")
        ret    = result.metrics.get("total_return_pct")
        print(f"      {name:<22}  trades={trades:<5}  return={ret}%  [{status}]")
        print(f"         --> {path}")

    # 3. Comparison table
    print("\n[3/4] Comparison")
    print_comparison(all_results)

    # 4. Save combined summary
    print("[4/4] Saving combined summary...")
    summary = {}
    for name, result in all_results.items():
        m = result.metrics
        summary[name] = {
            "total_return_pct":   m.get("total_return_pct"),
            "cagr_pct":           m.get("cagr_pct"),
            "max_drawdown_pct":   m.get("max_drawdown_pct"),
            "sharpe_ratio":       m.get("sharpe_ratio"),
            "calmar_ratio":       m.get("calmar_ratio"),
            "profit_factor":      m.get("profit_factor"),
            "win_rate_pct":       m.get("win_rate_pct"),
            "total_trades":       m.get("total_trades"),
            "final_equity":       m.get("final_equity"),
            "early_stop":         result.early_stop,
            "brt_pass":           m.get("brt_thresholds", {}).get("brt_pass"),
        }

    comparison_path = OUTPUTS_DIR / "comparison.json"
    comparison_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"      --> {comparison_path}")

    print("\n  All demo outputs are inside demo/outputs/")
    print("  Nothing was written outside the demo/ folder.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
