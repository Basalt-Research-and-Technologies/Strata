"""
basalt_strata/timeframes.py
===========================
Timeframe constants and helpers.

The library supports ANY timeframe — daily, hourly, 15-min, 5-min, 1-min, weekly.
The only thing you need to tell the Backtest is how many bars are in one year
(bars_per_year) so it can annualise metrics correctly.

This module provides pre-built constants for common NSE timeframes and a
helper to compute bars_per_year for custom timeframes.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Pre-built bars_per_year constants for common NSE timeframes
# ---------------------------------------------------------------------------
# NSE session : 09:15 – 15:29 IST
# Trading days: ~252 per year

_NSE_DAYS = 252

class Timeframe:
    """
    Ready-made bars_per_year values for common NSE timeframes.

    Pass the value directly to Backtest(bars_per_year=...).

    Examples
    --------
    >>> Backtest(..., bars_per_year=Timeframe.MIN1)
    >>> Backtest(..., bars_per_year=Timeframe.MIN15)
    >>> Backtest(..., bars_per_year=Timeframe.DAILY)
    """

    # Intraday — NSE session = 375 minutes (09:15–15:29)
    MIN1   = _NSE_DAYS * 375          # 94,500   (1-minute bars)
    MIN2   = _NSE_DAYS * 187          # 47,124   (2-minute bars)
    MIN3   = _NSE_DAYS * 125          # 31,500   (3-minute bars)
    MIN5   = _NSE_DAYS * 75           # 18,900   (5-minute bars)
    MIN10  = _NSE_DAYS * 37           #  9,324   (10-minute bars)
    MIN15  = _NSE_DAYS * 25           #  6,300   (15-minute bars)
    MIN30  = _NSE_DAYS * 12           #  3,024   (30-minute bars)
    HOUR1  = _NSE_DAYS * 6            #  1,512   (1-hour bars)

    # Daily / weekly / monthly
    DAILY   = _NSE_DAYS               #    252
    WEEKLY  = 52
    MONTHLY = 12

    @staticmethod
    def custom(minutes_per_bar: int, trading_days: int = 252, session_minutes: int = 375) -> int:
        """
        Compute bars_per_year for any custom bar size.

        Parameters
        ----------
        minutes_per_bar : int
            Duration of each bar in minutes (e.g. 1, 5, 15, 60).
            Pass 1440 for daily bars.
        trading_days : int
            Number of trading days per year. NSE default: 252.
        session_minutes : int
            Length of the trading session in minutes.
            NSE default: 375 (09:15 – 15:29).

        Returns
        -------
        int
            bars_per_year to pass to Backtest().

        Examples
        --------
        >>> Timeframe.custom(minutes_per_bar=15)   # 15-min bars
        6300
        >>> Timeframe.custom(minutes_per_bar=60)   # 1-hour bars
        1512
        >>> Timeframe.custom(minutes_per_bar=1440, session_minutes=1440)  # daily
        252
        """
        bars_per_day = max(1, session_minutes // minutes_per_bar)
        return trading_days * bars_per_day


# Convenience aliases
TF = Timeframe
