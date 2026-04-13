"""
examples/run_real_data.py
=========================
Run a backtest on the real Nifty 1-min CSV.

Usage
-----
    # Default: Jan 2024 – Dec 2024, EMA crossover
    python examples/run_real_data.py

    # Custom date range
    python examples/run_real_data.py --from 2023-01-01 --to 2023-12-31

    # Different strategy
    python examples/run_real_data.py --strategy rsi
    python examples/run_real_data.py --strategy bollinger
    python examples/run_real_data.py --strategy vwap
    python examples/run_real_data.py --strategy dual

    # Full custom config
    python examples/run_real_data.py \\
        --from 2022-01-01 --to 2022-12-31 \\
        --strategy ema \\
        --capital 2000000 --lot-size 50 --lots 1 \\
        --slippage 0.0005 --commission 20 \\
        --max-dd 0.15 --rfr 0.065 \\
        --output my_result.json
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basalt_strata import DataFeed, RuleBasedStrategy, Backtest, Timeframe
from my_strategies import (
    ema_crossover,
    rsi_mean_reversion,
    bollinger_breakout,
    vwap_reversion,
    dual_momentum,
    intraday_ema_crossover,
)

# ---------------------------------------------------------------------------
# Strategy registry — add your own strategies here
# ---------------------------------------------------------------------------
STRATEGIES = {
    "ema":        ema_crossover,
    "rsi":        rsi_mean_reversion,
    "bollinger":  bollinger_breakout,
    "vwap":       vwap_reversion,
    "dual":       dual_momentum,
    "intraday":   intraday_ema_crossover,   # best for 1-min data

    # Parameterised variants using partial()
    "ema_5_20":   partial(ema_crossover, fast=5, slow=20),
    "ema_20_50":  partial(ema_crossover, fast=20, slow=50),
    "rsi_tight":  partial(rsi_mean_reversion, oversold=25, overbought=75),
}

CSV_PATH   = r"D:\BRT\Nifty_Equity_1min.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backtest any strategy on real Nifty 1-min data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Date range
    p.add_argument("--from", dest="date_from", type=str, default="2024-01-01",
                   help="Start date (YYYY-MM-DD inclusive).")
    p.add_argument("--to",   dest="date_to",   type=str, default="2024-12-31",
                   help="End date   (YYYY-MM-DD inclusive).")

    # Strategy
    p.add_argument("--strategy", type=str, default="ema",
                   choices=list(STRATEGIES.keys()),
                   help="Strategy to run.")

    # Backtest config — all user-controlled, nothing hardcoded
    p.add_argument("--capital",       type=float, default=1_000_000,
                   help="Starting capital in INR.")
    p.add_argument("--lot-size",      type=int,   default=50,
                   help="Units per lot (Nifty standard = 50).")
    p.add_argument("--lots",          type=int,   default=1,
                   help="Lots per trade signal.")
    p.add_argument("--slippage",      type=float, default=0.0005,
                   help="Slippage fraction of price.")
    p.add_argument("--commission",    type=float, default=20.0,
                   help="Flat INR per order.")
    p.add_argument("--max-dd",        type=float, default=None,
                   help="Auto-stop drawdown limit, e.g. 0.20 = 20%%.")
    p.add_argument("--rfr",           type=float, default=0.065,
                   help="Annual risk-free rate.")
    p.add_argument("--bars-per-year", type=int, default=Timeframe.MIN1,
                   help=(
                       "Bars per year for annualisation. "
                       "Timeframe constants: MIN1=94500 MIN5=18900 MIN15=6300 "
                       "MIN30=3024 HOUR1=1512 DAILY=252. "
                       f"Default: {Timeframe.MIN1} (1-min NSE)."
                   ))

    # Output auto-named per run; default resolved after parse in main()
    p.add_argument("--output", type=str, default=None,
                   help="Write JSON tearsheet to this path. "
                        "Default: outputs/<strategy>_<from>_<to>.json")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _build_parser().parse_args()

    # Auto-name output if not specified by user
    OUTPUT_DIR.mkdir(exist_ok=True)
    if args.output is None:
        args.output = str(
            OUTPUT_DIR / f"{args.strategy}_{args.date_from}_{args.date_to}.json"
        )

    print(f"\n  CSV          : {CSV_PATH}")
    print(f"  Date range   : {args.date_from}  -->  {args.date_to}")
    print(f"  Strategy     : {args.strategy}")
    print(f"  Capital      : INR {args.capital:,.0f}")
    print(f"  Lot size     : {args.lot_size} x {args.lots} lot(s)")
    print(f"  Slippage     : {args.slippage * 100:.3f}%")
    print(f"  Commission   : INR {args.commission} per order")
    print(f"  Max DD limit : {f'{args.max_dd*100:.1f}%' if args.max_dd else 'disabled'}")
    print(f"  Bars/year    : {args.bars_per_year}")
    print(f"  Output       : {args.output}\n")

    # Step 1: Load & slice the CSV
    print("[1/4] Loading CSV ...")
    df = pd.read_csv(
        CSV_PATH,
        parse_dates=["timestamp"],
    )

    # Filter to requested date window
    mask = (
        (df["timestamp"].dt.date >= pd.Timestamp(args.date_from).date()) &
        (df["timestamp"].dt.date <= pd.Timestamp(args.date_to).date())
    )
    df = df[mask].copy()

    if df.empty:
        print(f"  [ERROR] No data found between {args.date_from} and {args.date_to}.")
        return 1

    print(f"      Rows in window : {len(df):,}")
    print(f"      First bar      : {df['timestamp'].iloc[0]}")
    print(f"      Last bar       : {df['timestamp'].iloc[-1]}")

    # Step 2: Build DataFeed
    print("[2/4] Validating data ...")
    feed = DataFeed.from_dataframe(
        df,
        symbol="NIFTY",
        timeframe="1min",
        timestamp_col="timestamp",
    )
    if feed.validation_warnings:
        for w in feed.validation_warnings:
            print(f"      [WARN] {w}")
    print(f"      DataFeed ready : {feed}")

    # Step 3: Run backtest
    print("[3/4] Running backtest ...")
    rule_fn  = STRATEGIES[args.strategy]
    strategy = RuleBasedStrategy(rule_fn=rule_fn)

    result = Backtest(
        feed               = feed,
        strategy           = strategy,
        initial_capital    = args.capital,
        lot_size           = args.lot_size,
        lots_per_trade     = args.lots,
        slippage_pct       = args.slippage,
        commission         = args.commission,
        max_drawdown_limit = args.max_dd,
        risk_free_rate     = args.rfr,
        bars_per_year      = args.bars_per_year,
    ).run()

    # Step 4: Results
    print("[4/4] Results\n")
    print(result.summary())

    if result.early_stop:
        print("\n  [!] Backtest stopped early -- max drawdown limit triggered.")

    m = result.metrics
    print(
        f"\n  Trades         : {m.get('total_trades')}\n"
        f"  Win rate       : {m.get('win_rate_pct')} %\n"
        f"  Profit factor  : {m.get('profit_factor')}\n"
        f"  Final equity   : INR {m.get('final_equity'):>15,.2f}\n"
        f"  BRT pass       : {m.get('brt_thresholds', {}).get('brt_pass')}\n"
    )

    if args.output:
        result.to_json(args.output)
        print(f"  Tearsheet  -->  {Path(args.output).resolve()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
