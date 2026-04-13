"""
examples/my_strategies.py
=========================
THIS IS WHERE YOU DEFINE YOUR STRATEGIES.

Each strategy is just a plain Python function (or a subclass of Strategy).
The function receives the OHLCV DataFrame and must return a pd.Series of
signals: 1 (buy), -1 (sell), 0 (flat/hold).

The library picks up your function via RuleBasedStrategy(rule_fn=your_fn).
Your function never touches money, lots, or costs — those live in Backtest().
"""

from __future__ import annotations
import pandas as pd


# ============================================================
# STRATEGY 1 — EMA Crossover
# ============================================================
# Idea: buy when the fast EMA crosses above the slow EMA,
#       sell when it crosses below.
# Suitable for: trending markets (Nifty, BankNifty daily/15min)

def ema_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.Series:
    """
    EMA(fast) / EMA(slow) crossover.

    Parameters
    ----------
    df   : OHLCV DataFrame from DataFeed
    fast : Fast EMA period (default 10)
    slow : Slow EMA period (default 50)
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[ema_fast > ema_slow] = 1    #  buy
    signal[ema_fast < ema_slow] = -1   #  sell
    return signal


# ============================================================
# STRATEGY 2 — RSI Mean-Reversion
# ============================================================
# Idea: buy when RSI is oversold (<30), sell when overbought (>70).
# Suitable for: sideways / range-bound markets.

def rsi_mean_reversion(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.Series:
    """
    RSI-based mean-reversion.

    Parameters
    ----------
    df         : OHLCV DataFrame from DataFeed
    period     : RSI look-back period (default 14)
    oversold   : RSI level to go long (default 30)
    overbought : RSI level to go short (default 70)
    """
    delta = df["close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, float("nan"))
    rsi   = 100 - (100 / (1 + rs))

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[rsi < oversold]   = 1    # oversold → buy
    signal[rsi > overbought] = -1   # overbought → sell
    return signal.fillna(0).astype(int)


# ============================================================
# STRATEGY 3 — Bollinger Band Breakout
# ============================================================
# Idea: buy on upper-band breakout, sell on lower-band break.
# Suitable for: volatile trending sessions.

def bollinger_breakout(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.Series:
    """
    Bollinger Band price breakout.

    Parameters
    ----------
    df      : OHLCV DataFrame from DataFeed
    period  : Rolling window for mean and std (default 20)
    std_dev : Number of standard deviations for band width (default 2.0)
    """
    mid   = df["close"].rolling(period).mean()
    band  = df["close"].rolling(period).std()
    upper = mid + std_dev * band
    lower = mid - std_dev * band

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[df["close"] > upper] = 1     # breakout above → buy
    signal[df["close"] < lower] = -1    # breakout below → sell
    return signal.fillna(0).astype(int)


# ============================================================
# STRATEGY 4 — VWAP Reversion (intraday)
# ============================================================
# Idea: buy when price dips below VWAP, sell when rises above.
# Suitable for: 15-min / 5-min intraday Nifty/BankNifty.

def vwap_reversion(df: pd.DataFrame) -> pd.Series:
    """
    VWAP mean-reversion. Resets VWAP every calendar day.

    Parameters
    ----------
    df : OHLCV DataFrame from DataFeed (best on intraday timeframes)
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical * df["volume"]).groupby(df.index.date).cumsum()
    cum_vol    = df["volume"].groupby(df.index.date).cumsum()
    vwap       = cum_tp_vol / cum_vol

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[df["close"] < vwap] = 1    # below VWAP → buy
    signal[df["close"] > vwap] = -1   # above VWAP → sell
    return signal.fillna(0).astype(int)


# ============================================================
# STRATEGY 5 — Dual Momentum (custom logic example)
# ============================================================
# Idea: combine EMA trend filter with RSI entry condition.

def dual_momentum(
    df: pd.DataFrame,
    ema_period: int = 200,
    rsi_period: int = 14,
    rsi_entry: float = 40.0,
) -> pd.Series:
    """
    Trend-filtered RSI entry.
    Only goes long when price > EMA(200) AND RSI < rsi_entry.
    Only goes short when price < EMA(200) AND RSI > (100 - rsi_entry).

    Parameters
    ----------
    df         : OHLCV DataFrame
    ema_period : Long-term trend filter period (default 200)
    rsi_period : RSI period (default 14)
    rsi_entry  : RSI threshold for entry (default 40)
    """
    trend_ema = df["close"].ewm(span=ema_period, adjust=False).mean()

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(com=rsi_period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=rsi_period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi   = 100 - (100 / (1 + rs))

    uptrend   = df["close"] > trend_ema
    downtrend = df["close"] < trend_ema

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[uptrend   & (rsi < rsi_entry)]         = 1   # buy dip in uptrend
    signal[downtrend & (rsi > 100 - rsi_entry)]   = -1  # sell rally in downtrend
    return signal.fillna(0).astype(int)


# ============================================================
# STRATEGY 6 — Intraday EMA Crossover (1-min / 5-min aware)
# ============================================================
# Designed for 1-min data. Adds two key intraday rules:
#   1. Only trade during the NSE session (09:15 to 15:20 IST)
#   2. Force-exit (signal=0) at session close (after 15:20)
# This prevents signals from carrying overnight across day gaps.

def intraday_ema_crossover(
    df: pd.DataFrame,
    fast: int = 9,
    slow: int = 21,
    session_open: str = "09:15",
    session_close: str = "15:20",
) -> pd.Series:
    """
    Session-aware EMA crossover for intraday 1-min data.

    Parameters
    ----------
    df            : OHLCV DataFrame from DataFeed (1-min recommended)
    fast          : Fast EMA span (default 9)
    slow          : Slow EMA span (default 21)
    session_open  : Only trade at or after this time (default 09:15)
    session_close : Force flat at or after this time (default 15:20)
    """
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()

    signal = pd.Series(0, index=df.index, dtype=int)

    # Only signal during open session window
    t = df.index
    # Handle both tz-aware and tz-naive index
    if hasattr(t, "time"):
        bar_time = pd.Series(
            [ts.time() for ts in t], index=t
        )
    else:
        bar_time = pd.Series(
            [ts.time() for ts in t], index=t
        )

    open_time  = pd.Timestamp(f"2000-01-01 {session_open}").time()
    close_time = pd.Timestamp(f"2000-01-01 {session_close}").time()
    in_session = (bar_time >= open_time) & (bar_time <= close_time)

    signal[in_session & (ema_fast > ema_slow)] = 1
    signal[in_session & (ema_fast < ema_slow)] = -1
    # Bars outside session remain 0 (flat) — forces exit at session close

    return signal.astype(int)
