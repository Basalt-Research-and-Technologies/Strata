"""
examples/run_base_strategy.py
==============================
Runs the ORB + H2 + Bias intraday strategy (BaseStrategyEngine) on
real 1-minute Nifty data via basalt-strata.

Reference benchmark (BRT-v2, TradingView image):
  Period         : Mar 1 2025 - Feb 28 2026
  Total P&L      : +38,223 INR  (+38.22%)
  Max DD         : 58,221 INR   (49.19%)
  Total trades   : 384
  Win rate       : 42.45% (163/384)
  Profit factor  : 1.171

Run:
    py examples/run_base_strategy.py
    py examples/run_base_strategy.py --from 2025-03-01 --to 2026-02-28
    py examples/run_base_strategy.py --capital 100000 --lot-size 50 --lots 1
"""

from __future__ import annotations

import argparse
import sys
from datetime import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basalt_strata import DataFeed, RuleBasedStrategy, Backtest, Timeframe

CSV_1MIN   = r"D:\BRT\Nifty_Equity_1min.csv"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


# ===========================================================================
# Per-day state machine — mirrors BaseStrategyEngine exactly, no DB needed
# ===========================================================================

def _process_day(day_df: pd.DataFrame) -> pd.Series:
    """
    Run ORB/H2/Bias state machine on one trading day's 1-min bars.
    Returns signal Series {-1, 0, 1} indexed to day_df.
    """
    sig = pd.Series(0, index=day_df.index, dtype=int)

    state = dict(
        open_price=None, bias=None,
        orb_high=None,   orb_low=None,
        crossed_above_open=False, crossed_below_open=False,
        broke_orb_high=False,    broke_orb_low=False,
        locked=False,
        f15_direction=None, conviction=None,
        decision_taken=False,
        position=None,    # "LONG" or "SHORT"
        exited=False,
        h13_buy_done=False,
        h13_sell_done=False,
    )

    for row in day_df.itertuples():
        ts       = row.Index
        bar_time = ts.time()
        end_time = (ts + pd.Timedelta(minutes=1)).time()
        o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)

        # 9:15 — initialise open price + ORB seed
        if state["open_price"] is None and bar_time == time(9, 15):
            state["open_price"] = o
            state["orb_high"]   = h
            state["orb_low"]    = l

        op = state["open_price"]
        if op is None:
            continue

        # 9:15-9:28 — update ORB, track crosses
        if not state["locked"] and time(9, 15) <= bar_time <= time(9, 28):
            state["orb_high"] = max(state["orb_high"], h)
            state["orb_low"]  = min(state["orb_low"],  l)
            if h > op: state["crossed_above_open"] = True
            if l < op: state["crossed_below_open"] = True

        # BIAS at end_time 9:16 (9:15 candle close)
        if state["bias"] is None and end_time == time(9, 16):
            state["bias"] = "LONG" if c >= op else "SHORT"

        # LOCK at end_time 9:29 (9:28 candle close)
        if not state["locked"] and end_time == time(9, 29):
            state["locked"]        = True
            state["f15_direction"] = "UP" if c > op else "DOWN"
            agree = (
                (state["bias"] == "LONG"  and state["f15_direction"] == "UP") or
                (state["bias"] == "SHORT" and state["f15_direction"] == "DOWN")
            )
            state["conviction"] = "HIGH" if agree else "LOW"

        # ─── Post-lock ORB break tracking ─────────────────────────────────
        # IMPORTANT: update BEFORE breakout check (mirrors original engine)
        if state["locked"]:
            if h > (state["orb_high"] or 0): state["broke_orb_high"] = True
            if l < (state["orb_low"]  or 0): state["broke_orb_low"]  = True

        # H2 decision at end_time 9:30 (9:29 candle)
        if state["locked"] and not state["decision_taken"] and end_time == time(9, 30):
            above = state["crossed_above_open"]
            below = state["crossed_below_open"]
            if not above:                        # H2_SHORT_DEVELOPING
                sig.loc[ts] = -1
                state["position"] = "SHORT"
                state["decision_taken"] = True
            elif not below:                      # H2_LONG_DEVELOPING
                sig.loc[ts] = 1
                state["position"] = "LONG"
                state["decision_taken"] = True

        # ORB breakout at end_time 9:31 (9:30 candle close)
        # broke_orb already updated above with this bar's data
        if state["locked"] and not state["decision_taken"] and end_time == time(9, 31):
            orb_h, orb_l = state["orb_high"], state["orb_low"]
            if c > orb_h and not state["broke_orb_low"]:
                sig.loc[ts] = 1;  state["position"] = "LONG"
            elif c < orb_l and not state["broke_orb_high"]:
                sig.loc[ts] = -1; state["position"] = "SHORT"
            state["decision_taken"] = True

        # Exit management
        if state["position"] and not state["exited"]:
            orb_h = state["orb_high"] or 0
            orb_l = state["orb_low"]  or 0

            if end_time == time(15, 15):           # Time exit
                sig.loc[ts] = 0
                state["position"] = None
                state["exited"]   = True
            elif state["position"] == "LONG"  and l < orb_h:   # Re-enter range
                sig.loc[ts] = 0; state["position"] = None; state["exited"] = True
            elif state["position"] == "SHORT" and h > orb_l:
                sig.loc[ts] = 0; state["position"] = None; state["exited"] = True

        # H13 micro-trade signals — fire every day near close
        # BUY at end_time 15:29 (bar 15:28), SELL at end_time 15:30 (bar 15:29)
        if end_time == time(15, 29) and not state["h13_buy_done"]:
            sig.loc[ts] = 1
            state["h13_buy_done"] = True

        elif end_time == time(15, 30) and not state["h13_sell_done"]:
            sig.loc[ts] = -1
            state["h13_sell_done"] = True

    return sig


def orb_h2_signal_series(df: pd.DataFrame) -> pd.Series:
    """Called by RuleBasedStrategy on the full multi-day DataFrame."""
    parts = [_process_day(grp) for _, grp in df.groupby(df.index.date)]
    if not parts:
        return pd.Series(0, index=df.index, dtype=int)
    return pd.concat(parts).sort_index().reindex(df.index, fill_value=0).astype(int)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ORB + H2 + Bias strategy on Nifty 1-min data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--from", dest="date_from", default="2025-03-01")
    p.add_argument("--to",   dest="date_to",   default="2026-02-28")
    p.add_argument("--capital",    type=float, default=100_000)
    p.add_argument("--lot-size",   type=int,   default=50)
    p.add_argument("--lots",       type=int,   default=1)
    p.add_argument("--slippage",   type=float, default=0.0002)
    p.add_argument("--commission", type=float, default=0.0,
                   help="INR per order (0 matches TradingView default)")
    p.add_argument("--output",     default=None)
    return p


def main() -> int:
    args = _build_parser().parse_args()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = args.output or str(
        OUTPUT_DIR / f"orb_h2_{args.date_from}_{args.date_to}.json"
    )

    print(f"\n  Strategy     : ORB + H2 + Bias (BaseStrategyEngine)")
    print(f"  Data         : {CSV_1MIN}")
    print(f"  Date range   : {args.date_from}  -->  {args.date_to}")
    print(f"  Capital      : INR {args.capital:,.0f}")
    print(f"  Lot size     : {args.lot_size} x {args.lots} lot(s)")
    print(f"  Slippage     : {args.slippage * 100:.3f}%")
    print(f"  Commission   : INR {args.commission} per order")
    print(f"  Output       : {out_path}\n")
    print("  Benchmark (BRT-v2 TradingView):")
    print("    Return: +38.22%  |  Trades: 384  |  Win: 42.45%  |  PF: 1.171\n")

    print("[1/4] Loading 1-min data ...")
    df_raw = pd.read_csv(CSV_1MIN, parse_dates=["timestamp"])
    mask = (
        (df_raw["timestamp"].dt.date >= pd.Timestamp(args.date_from).date()) &
        (df_raw["timestamp"].dt.date <= pd.Timestamp(args.date_to).date())
    )
    df_raw = df_raw[mask].copy()
    print(f"      Rows in window : {len(df_raw):,}")
    print(f"      First bar      : {df_raw['timestamp'].iloc[0]}")
    print(f"      Last  bar      : {df_raw['timestamp'].iloc[-1]}")

    print("[2/4] Building DataFeed ...")
    before = len(df_raw)
    df_raw = df_raw.drop_duplicates(subset="timestamp", keep="last")
    if len(df_raw) < before:
        print(f"      Removed {before - len(df_raw):,} duplicate timestamp(s).")
    feed = DataFeed.from_dataframe(
        df_raw, symbol="NIFTY", timeframe="1min", timestamp_col="timestamp"
    )
    print(f"      {feed}")

    print("[3/4] Running backtest (generating signals day-by-day, ~30s) ...")
    strategy = RuleBasedStrategy(rule_fn=orb_h2_signal_series)

    result = Backtest(
        feed               = feed,
        strategy           = strategy,
        initial_capital    = args.capital,
        lot_size           = args.lot_size,
        lots_per_trade     = args.lots,
        slippage_pct       = args.slippage,
        commission         = args.commission,
        max_drawdown_limit = 0.65,
        risk_free_rate     = 0.065,
        bars_per_year      = Timeframe.MIN1,
    ).run()

    print("[4/4] Results\n")
    print(result.summary())

    m = result.metrics
    print(f"\n  ---- Basalt Strata vs BRT-v2 Benchmark ----")
    print(f"  {'Metric':<24} {'Basalt':>14} {'Benchmark':>14}")
    print(f"  {'-'*54}")
    for label, val, bmark in [
        ("Total return %",  m.get("total_return_pct"),    "+38.22"),
        ("Total trades",    m.get("total_trades"),        "384"),
        ("Win rate %",      m.get("win_rate_pct"),        "42.45"),
        ("Profit factor",   m.get("profit_factor"),       "1.171"),
        ("Max drawdown %",  m.get("max_drawdown_pct"),    "-49.19"),
        ("Final equity",    m.get("final_equity"),        "138,223"),
    ]:
        print(f"  {label:<24} {str(val):>14} {bmark:>14}")

    result.to_json(out_path)
    print(f"\n  Tearsheet  -->  {Path(out_path).resolve()}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
