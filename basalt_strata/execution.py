"""
Execution -- bar-by-bar trade simulation.

Fill price rules:
    - Entry: bar N+1 open * (1 + slippage_pct) for longs
             bar N+1 open * (1 - slippage_pct) for shorts
    - Exit:  bar N+1 open * (1 - slippage_pct) for longs
             bar N+1 open * (1 + slippage_pct) for shorts

Slippage_pct is expressed as a decimal (e.g. 0.0005 = 0.05%).

Commission:
    Flat ₹ per order (not per lot). Applied on both entry and exit independently.
    e.g. commission=20 means ₹20 on entry + ₹20 on exit = ₹40 round-trip.

Position model:
    - Single-position model: only one trade open at a time.
    - Signal 1  -> enter long (if flat) or ignore (if already long)
    - Signal -1 -> enter short (if flat) or ignore (if already short)
    - Signal 0  -> exit current position (if any)
    - Opposing signal (long when short, or short when long) -> exit then re-enter

P&L calculation:
    Long:  (exit_price - entry_price) * lots * lot_size - total_commission
    Short: (entry_price - exit_price) * lots * lot_size - total_commission

max_drawdown_limit:
    Checked after each trade close. If the running drawdown from the equity
    peak exceeds this limit, the backtest stops early. No exception raised --
    the partial trade log is returned with early_stop=True in the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import pandas as pd
import numpy as np


@dataclass
class Trade:
    """Single completed trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: int          # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    lots: int
    lot_size: int
    slippage_cost: float    # total slippage cost in ₹ (entry + exit combined)
    commission_cost: float  # total commission in ₹ (entry + exit combined)
    gross_pnl: float        # P&L before costs
    net_pnl: float          # P&L after all costs

    @property
    def duration_bars(self) -> int:
        return 0  # filled by Execution after the fact

    def to_dict(self) -> dict:
        return {
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "direction": "long" if self.direction == 1 else "short",
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "lots": self.lots,
            "lot_size": self.lot_size,
            "units": self.lots * self.lot_size,
            "slippage_cost": round(self.slippage_cost, 2),
            "commission_cost": round(self.commission_cost, 2),
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
        }


class Execution:
    """
    Simulates trade execution bar-by-bar given a validated signal Series.

    Parameters
    ----------
    data : pd.DataFrame
        DataFeed.data
    signals : pd.Series
        Validated signal Series from Strategy.get_signals()
    lot_size : int
    lots_per_trade : int
    slippage_pct : float
        e.g. 0.0005 for 0.05%
    commission : float
        Flat ₹ per order (entry and exit each count as one order).
    initial_capital : float
    max_drawdown_limit : float or None
        Fraction e.g. 0.20 = stop if drawdown exceeds 20% from peak.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        lot_size: int,
        lots_per_trade: int,
        position_sizing_fn: Optional[Callable[[float, float, int], int]],
        slippage_pct: float,
        commission: float,
        initial_capital: float,
        max_drawdown_limit: Optional[float],
    ) -> None:
        self.data = data
        self.signals = signals
        self.lot_size = lot_size
        self.lots_per_trade = lots_per_trade
        self.position_sizing_fn = position_sizing_fn
        self.slippage_pct = slippage_pct
        self.commission = commission
        self.initial_capital = initial_capital
        self.max_drawdown_limit = max_drawdown_limit

    def run(self) -> tuple[list[Trade], pd.Series, bool]:
        """
        Execute the simulation.

        Returns
        -------
        trades : list[Trade]
        equity_curve : pd.Series (DatetimeIndex -> float, one value per bar)
        early_stop : bool -- True if max_drawdown_limit was hit
        """
        data = self.data
        signals = self.signals

        trades: list[Trade] = []
        equity = self.initial_capital
        peak_equity = equity
        equity_curve: dict[pd.Timestamp, float] = {}
        early_stop = False

        # State
        position = 0          # 0 = flat, 1 = long, -1 = short
        entry_price = 0.0
        entry_time = None
        entry_fill_cost = 0.0  # commission + slippage on entry
        entry_lots = self.lots_per_trade  # overridden per-trade when position_sizing_fn is set

        bars = list(data.itertuples())

        for i, bar in enumerate(bars):
            ts = bar.Index

            # Signal was generated at bar i-1, so we act on bar i's open.
            # Bar 0 has no prior signal -- always flat.
            if i == 0:
                equity_curve[ts] = equity
                continue

            sig = int(signals.iloc[i - 1])
            open_price = bar.open

            # --- determine what action to take ---
            # Opposing signal = exit current + enter opposite
            # Same direction signal while already in position = ignore
            # Zero signal = exit if in position

            action = self._resolve_action(position, sig)

            if action in ("exit_long", "exit_short", "reverse_to_long", "reverse_to_short"):
                # Close existing position
                direction = position
                exit_raw = open_price
                slip_exit = exit_raw * self.slippage_pct
                if direction == 1:
                    exit_fill = exit_raw - slip_exit
                else:
                    exit_fill = exit_raw + slip_exit

                units = entry_lots * self.lot_size
                if direction == 1:
                    gross = (exit_fill - entry_price) * units
                else:
                    gross = (entry_price - exit_fill) * units

                total_commission = self.commission * 2  # entry + exit
                entry_slip_cost = abs(entry_price - (
                    entry_price / (1 + self.slippage_pct)
                    if direction == 1
                    else entry_price / (1 - self.slippage_pct)
                )) * units
                exit_slip_cost = slip_exit * units
                total_slippage = entry_slip_cost + exit_slip_cost
                net = gross - total_commission

                trade = Trade(
                    entry_time=entry_time,
                    exit_time=ts,
                    direction=direction,
                    entry_price=round(entry_price, 4),
                    exit_price=round(exit_fill, 4),
                    lots=entry_lots,
                    lot_size=self.lot_size,
                    slippage_cost=round(total_slippage, 2),
                    commission_cost=round(total_commission, 2),
                    gross_pnl=round(gross, 2),
                    net_pnl=round(net, 2),
                )
                trades.append(trade)
                equity += net
                position = 0

                # Check drawdown limit after closing a trade
                if equity > peak_equity:
                    peak_equity = equity
                if self.max_drawdown_limit is not None:
                    current_dd = (peak_equity - equity) / peak_equity
                    if current_dd >= self.max_drawdown_limit:
                        equity_curve[ts] = equity
                        early_stop = True
                        break

            if action in ("enter_long", "reverse_to_long"):
                slip = open_price * self.slippage_pct
                entry_price = open_price + slip
                entry_time = ts
                position = 1
                # Resolve lots: dynamic sizer takes priority over static lots_per_trade
                if self.position_sizing_fn is not None:
                    entry_lots = max(1, int(self.position_sizing_fn(equity, open_price, self.lot_size)))
                else:
                    entry_lots = self.lots_per_trade

            elif action in ("enter_short", "reverse_to_short"):
                slip = open_price * self.slippage_pct
                entry_price = open_price - slip
                entry_time = ts
                position = -1
                # Resolve lots: dynamic sizer takes priority over static lots_per_trade
                if self.position_sizing_fn is not None:
                    entry_lots = max(1, int(self.position_sizing_fn(equity, open_price, self.lot_size)))
                else:
                    entry_lots = self.lots_per_trade

            equity_curve[ts] = equity

        # If still in a position at end of data, close at last bar's close
        if position != 0 and not early_stop:
            last_bar = bars[-1]
            last_close = last_bar.close
            direction = position
            slip = last_close * self.slippage_pct
            if direction == 1:
                exit_fill = last_close - slip
                gross = (exit_fill - entry_price) * entry_lots * self.lot_size
            else:
                exit_fill = last_close + slip
                gross = (entry_price - exit_fill) * entry_lots * self.lot_size
            net = gross - self.commission * 2
            trade = Trade(
                entry_time=entry_time,
                exit_time=last_bar.Index,
                direction=direction,
                entry_price=round(entry_price, 4),
                exit_price=round(exit_fill, 4),
                lots=entry_lots,
                lot_size=self.lot_size,
                slippage_cost=round(slip * entry_lots * self.lot_size * 2, 2),
                commission_cost=round(self.commission * 2, 2),
                gross_pnl=round(gross, 2),
                net_pnl=round(net, 2),
            )
            trades.append(trade)
            equity += net
            equity_curve[last_bar.Index] = equity

        equity_series = pd.Series(equity_curve, name="equity")
        equity_series.index.name = "timestamp"
        return trades, equity_series, early_stop

    @staticmethod
    def _resolve_action(position: int, signal: int) -> str:
        """Map current position + incoming signal to an action string."""
        if position == 0:
            if signal == 1:
                return "enter_long"
            elif signal == -1:
                return "enter_short"
            else:
                return "hold_flat"
        elif position == 1:
            if signal == -1:
                return "reverse_to_short"
            elif signal == 0:
                return "exit_long"
            else:
                return "hold_long"
        else:  # position == -1
            if signal == 1:
                return "reverse_to_long"
            elif signal == 0:
                return "exit_short"
            else:
                return "hold_short"
