"""
examples/run_backtest.py
========================
Complete example showing how to run any strategy from my_strategies.py.

Run:
    python examples/run_backtest.py
    python examples/run_backtest.py --strategy rsi
    python examples/run_backtest.py --strategy bollinger --capital 2000000
    python examples/run_backtest.py --csv path/to/data.csv --symbol BANKNIFTY
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the library is importable when running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basalt_strata import DataFeed, RuleBasedStrategy, Backtest

# Import all strategies from your strategy file — add new ones here
from my_strategies import (
    ema_crossover,
    rsi_mean_reversion,
    bollinger_breakout,
    vwap_reversion,
    dual_momentum,
)


# ---------------------------------------------------------------------------
# Strategy registry — ADD YOUR OWN STRATEGIES HERE
# ---------------------------------------------------------------------------
#
# Key   = name you pass to --strategy
# Value = a function (df) -> pd.Series of signals {-1, 0, 1}
#
# You can use functools.partial to bake in parameters:
#   "ema_fast": partial(ema_crossover, fast=5, slow=20),
#

STRATEGIES = {
    "ema":        ema_crossover,
    "rsi":        rsi_mean_reversion,
    "bollinger":  bollinger_breakout,
    "vwap":       vwap_reversion,
    "dual":       dual_momentum,

    # Examples of parameterised variants:
    "ema_fast":   partial(ema_crossover, fast=5, slow=20),
    "rsi_tight":  partial(rsi_mean_reversion, oversold=25, overbought=75),
}


# ---------------------------------------------------------------------------
# Synthetic data (used when no --csv is provided)
# ---------------------------------------------------------------------------

def _synthetic_feed(n_bars: int, seed: int | None, symbol: str, timeframe: str) -> DataFeed:
    rng    = np.random.default_rng(seed)
    close  = 18_000.0
    closes = [close]
    for _ in range(n_bars - 1):
        close *= 1 + rng.normal(0.0003, 0.009)
        closes.append(close)

    closes = np.array(closes)
    noise  = rng.uniform(0.002, 0.006, n_bars)
    opens  = closes * (1 + rng.uniform(-0.004, 0.004, n_bars))
    highs  = np.maximum(closes, opens) * (1 + noise)
    lows   = np.minimum(closes, opens) * (1 - noise)
    vols   = rng.integers(50_000, 500_000, n_bars).astype(float)

    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n_bars, freq="B")
    df  = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )
    return DataFeed.from_dataframe(df, symbol=symbol, timeframe=timeframe)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run any strategy from my_strategies.py against a backtest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Data source --
    src = p.add_argument_group("Data source")
    src.add_argument("--csv",       type=str,   default=None,    help="Path to OHLCV CSV file.")
    src.add_argument("--date-col",  type=str,   default="timestamp", help="Timestamp column name in CSV.")
    src.add_argument("--bars",      type=int,   default=504,     help="Bars to generate when no --csv given.")
    src.add_argument("--seed",      type=int,   default=None,    help="Random seed for synthetic data.")
    src.add_argument("--symbol",    type=str,   default="NIFTY", help="Instrument symbol label.")
    src.add_argument("--timeframe", type=str,   default="1D",    help="Timeframe label e.g. '15min', '1D'.")

    # -- Strategy selection --
    strat = p.add_argument_group("Strategy")
    strat.add_argument(
        "--strategy", type=str, default="ema",
        choices=list(STRATEGIES.keys()),
        help="Strategy to run. Add new ones to STRATEGIES dict in this file.",
    )

    # -- Backtest config (all user-controlled) --
    bt = p.add_argument_group("Backtest configuration")
    bt.add_argument("--capital",      type=float, default=1_000_000, help="Starting capital in ₹.")
    bt.add_argument("--lot-size",     type=int,   default=50,        help="Units per lot.")
    bt.add_argument("--lots",         type=int,   default=1,         help="Lots per trade signal.")
    bt.add_argument("--slippage",     type=float, default=0.0005,    help="Slippage fraction of price.")
    bt.add_argument("--commission",   type=float, default=20.0,      help="Flat ₹ per order.")
    bt.add_argument("--max-dd",       type=float, default=None,      help="Max drawdown limit (e.g. 0.20).")
    bt.add_argument("--rfr",          type=float, default=0.065,     help="Annual risk-free rate.")
    bt.add_argument("--bars-per-year",type=int,   default=252,       help="Bars per year for annualisation.")

    # -- Output --
    out = p.add_argument_group("Output")
    out.add_argument("--output",  type=str,  default=None,  help="Write JSON tearsheet to this path.")
    out.add_argument("--no-summary", action="store_true",   help="Skip printing the summary table.")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _build_parser().parse_args()

    # 1. Load data
    if args.csv:
        print(f"[+] Loading data from {args.csv} …")
        feed = DataFeed.from_csv(
            args.csv,
            symbol=args.symbol,
            timeframe=args.timeframe,
            date_col=args.date_col,
        )
    else:
        print(f"[+] Generating {args.bars} synthetic bars (seed={args.seed}) …")
        feed = _synthetic_feed(
            n_bars=args.bars,
            seed=args.seed,
            symbol=args.symbol,
            timeframe=args.timeframe,
        )

    if feed.validation_warnings:
        for w in feed.validation_warnings:
            print(f"    [WARN] {w}")

    print(f"[+] Feed : {feed}")

    # 2. Select strategy
    rule_fn  = STRATEGIES[args.strategy]
    strategy = RuleBasedStrategy(rule_fn=rule_fn)
    print(f"[+] Strategy : {args.strategy}  ({rule_fn.__name__ if hasattr(rule_fn, '__name__') else rule_fn.func.__name__})")

    # 3. Run backtest
    print("[+] Running backtest …\n")
    result = Backtest(
        feed              = feed,
        strategy          = strategy,
        initial_capital   = args.capital,
        lot_size          = args.lot_size,
        lots_per_trade    = args.lots,
        slippage_pct      = args.slippage,
        commission        = args.commission,
        max_drawdown_limit= args.max_dd,
        risk_free_rate    = args.rfr,
        bars_per_year     = args.bars_per_year,
    ).run()

    # 4. Print results
    if not args.no_summary:
        print(result.summary())

    if result.early_stop:
        print("\n[!] Early stop: max drawdown limit was hit.")

    m = result.metrics
    print(
        f"\n  Trades      : {m.get('total_trades')}\n"
        f"  Final equity: ₹{m.get('final_equity'):>15,.2f}\n"
        f"  BRT pass    : {m.get('brt_thresholds', {}).get('brt_pass')}\n"
    )

    # 5. Save tearsheet
    if args.output:
        result.to_json(args.output)
        print(f"[+] Tearsheet written → {Path(args.output).resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
