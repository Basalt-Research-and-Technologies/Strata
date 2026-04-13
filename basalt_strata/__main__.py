"""
python -m basalt_strata
-----------------------
Command-line entry point for the Basalt Strata library.

Usage
-----
    python -m basalt_strata --help
    python -m basalt_strata --version
    python -m basalt_strata --demo
    python -m basalt_strata --demo --capital 500000 --lot-size 25 --lots 1
    python -m basalt_strata --demo --slippage 0.001 --commission 15 --max-dd 0.15
    python -m basalt_strata --demo --bars 1000 --seed 7 --output my_result.json
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    print(
        "\n"
        "╔══════════════════════════════════════════════════╗\n"
        "║        Basalt Strata  ·  Backtesting Library     ║\n"
        "║        Indian Equity & Derivatives · NSE/BSE      ║\n"
        "╚══════════════════════════════════════════════════╝"
    )


# ---------------------------------------------------------------------------
# Argument parser — every value is a user choice, nothing hardcoded
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m basalt_strata",
        description=(
            "Basalt Strata — strategy testing and evaluation library "
            "for Indian equity and derivatives markets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Basic demo with defaults
  python -m basalt_strata --demo

  # Custom capital and lots
  python -m basalt_strata --demo --capital 2000000 --lot-size 25 --lots 4

  # Aggressive slippage / low commission broker
  python -m basalt_strata --demo --slippage 0.001 --commission 0

  # Tight drawdown guard, 15-min bars
  python -m basalt_strata --demo --max-dd 0.10 --bars-per-year 6300

  # Save tearsheet to a custom file
  python -m basalt_strata --demo --output tearsheet.json

  # Reproducible run with fixed seed
  python -m basalt_strata --demo --seed 123 --bars 504
""",
    )

    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Print the installed version and exit.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run a synthetic EMA(10/50) crossover demo on generated price data. "
            "No data file required. Use the flags below to customise every parameter."
        ),
    )

    # ── Data generation ──────────────────────────────────────────────────
    gen = parser.add_argument_group("Data generation (--demo only)")
    gen.add_argument(
        "--bars", type=int, default=504, metavar="N",
        help="Number of synthetic price bars to generate. (default: 504 ≈ 2 years daily)",
    )
    gen.add_argument(
        "--seed", type=int, default=None, metavar="INT",
        help="Random seed for reproducibility. Omit for a random run each time.",
    )

    # ── Backtest config ───────────────────────────────────────────────────
    bt = parser.add_argument_group("Backtest configuration")
    bt.add_argument(
        "--capital", type=float, default=1_000_000, metavar="INR",
        help="Starting capital in ₹. (default: 1,000,000)",
    )
    bt.add_argument(
        "--lot-size", type=int, default=50, metavar="UNITS",
        help="Units per lot. (default: 50, Nifty standard)",
    )
    bt.add_argument(
        "--lots", type=int, default=1, metavar="N",
        help="Number of lots per trade signal. (default: 1)",
    )
    bt.add_argument(
        "--slippage", type=float, default=0.0005, metavar="FRAC",
        help="Slippage as a fraction of price, e.g. 0.0005 = 0.05%%. (default: 0.0005)",
    )
    bt.add_argument(
        "--commission", type=float, default=20.0, metavar="INR",
        help="Flat ₹ per order (entry and exit each counted). (default: 20)",
    )
    bt.add_argument(
        "--max-dd", type=float, default=None, metavar="FRAC",
        help=(
            "Auto-stop the backtest if drawdown from peak exceeds this fraction. "
            "e.g. 0.20 = stop at 20%% drawdown. Omit to disable. (default: disabled)"
        ),
    )
    bt.add_argument(
        "--rfr", type=float, default=0.065, metavar="RATE",
        help="Annual risk-free rate (decimal) for Sharpe/Sortino. (default: 0.065 = 6.5%%)",
    )
    bt.add_argument(
        "--bars-per-year", type=int, default=252, metavar="N",
        help=(
            "Bars in one trading year for annualisation. "
            "252 for daily, 6300 for 15-min NSE. (default: 252)"
        ),
    )

    # ── Output ────────────────────────────────────────────────────────────
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--output", type=str, default="demo_tearsheet.json", metavar="PATH",
        help="File path for the JSON tearsheet. (default: demo_tearsheet.json)",
    )
    out.add_argument(
        "--no-file",
        action="store_true",
        help="Skip writing the tearsheet file; only print to console.",
    )

    return parser


# ---------------------------------------------------------------------------
# Synthetic data generator — fully parameterised
# ---------------------------------------------------------------------------

def _generate_ohlcv(n_bars: int, seed: int | None) -> pd.DataFrame:
    """Generate NIFTY-like random-walk OHLCV bars."""
    rng = np.random.default_rng(seed)
    close = 18_000.0
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

    idx = pd.bdate_range(
        start=datetime.today() - timedelta(days=int(n_bars * 1.5)),
        periods=n_bars,
        freq="B",
    )
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Demo strategy (EMA crossover)
# ---------------------------------------------------------------------------

def _ema_crossover(df: pd.DataFrame) -> pd.Series:
    """EMA(10) / EMA(50) crossover — buy when fast > slow, sell otherwise."""
    ema10 = df["close"].ewm(span=10, adjust=False).mean()
    ema50 = df["close"].ewm(span=50, adjust=False).mean()
    sig   = pd.Series(0, index=df.index, dtype=int)
    sig[ema10 > ema50] = 1
    sig[ema10 < ema50] = -1
    return sig


# ---------------------------------------------------------------------------
# Demo runner — every parameter driven by CLI args
# ---------------------------------------------------------------------------

def run_demo(args: argparse.Namespace) -> int:
    from basalt_strata import DataFeed, RuleBasedStrategy, Backtest

    seed_label = str(args.seed) if args.seed is not None else "random"
    print(
        f"\n  Generating {args.bars} synthetic bars  (seed={seed_label}) …"
    )
    raw  = _generate_ohlcv(n_bars=args.bars, seed=args.seed)
    feed = DataFeed.from_dataframe(raw, symbol="NIFTY-SYNTHETIC", timeframe="1D")

    print("  Strategy     : EMA(10) / EMA(50) crossover")
    print(f"  Capital      : ₹{args.capital:,.0f}")
    print(f"  Lot size     : {args.lot_size} units × {args.lots} lot(s)")
    print(f"  Slippage     : {args.slippage * 100:.3f}%")
    print(f"  Commission   : ₹{args.commission} per order")
    print(f"  Max DD limit : {f'{args.max_dd*100:.1f}%' if args.max_dd else 'disabled'}")
    print(f"  Risk-free    : {args.rfr * 100:.2f}%  |  bars/year: {args.bars_per_year}\n")

    strategy = RuleBasedStrategy(rule_fn=_ema_crossover)

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

    print(result.summary())

    if result.early_stop:
        print("\n  [!] Backtest stopped early — max drawdown limit hit.")

    m = result.metrics
    print(
        f"\n  Trades logged : {m.get('total_trades')}\n"
        f"  Final equity  : ₹{m.get('final_equity'):>15,.2f}\n"
    )

    if not args.no_file:
        result.to_json(args.output)
        print(f"  Tearsheet     → {Path(args.output).resolve()}\n")

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    _print_banner()

    if args.version:
        from basalt_strata import __version__
        print(f"\n  basalt-strata  {__version__}\n")
        return 0

    if args.demo:
        try:
            return run_demo(args)
        except Exception as exc:
            print(f"\n  [ERROR] {exc}", file=sys.stderr)
            return 1

    # No flag: show quick-start hint
    from basalt_strata import __version__
    print(
        f"\n  version      : {__version__}\n"
        "  quick start  : python -m basalt_strata --demo\n"
        "  full options : python -m basalt_strata --help\n\n"
        "  from basalt_strata import DataFeed, RuleBasedStrategy, Backtest\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
