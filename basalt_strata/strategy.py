"""
Strategy -- base class and RuleBasedStrategy.

Signal contract (non-negotiable):
    - generate_signals() must return a pd.Series of int dtype.
    - Valid values: 1 (buy/long), -1 (sell/short), 0 (flat/no position).
    - Index must be a DatetimeIndex matching the DataFeed index exactly.
    - Any value outside {-1, 0, 1} raises StrategyError.
    - NaN values in the signal raise StrategyError.

Design decision -- look-ahead bias prevention:
    The Execution module shifts signals by +1 bar before applying them.
    This means: signal generated at bar N is filled at bar N+1's open.
    Strategy authors do NOT need to shift themselves. The library handles it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd

from .datafeed import DataFeed


class StrategyError(Exception):
    """Raised when a strategy returns an invalid signal Series."""


class Strategy(ABC):
    """
    Abstract base class for all strategy types.

    Subclasses must implement generate_signals(data) -> pd.Series.
    """

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Given the clean OHLCV DataFrame from DataFeed, return a signal Series.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFeed.data DataFrame (index=DatetimeIndex, columns=OHLCV+symbol+timeframe).

        Returns
        -------
        pd.Series
            Integer Series with values in {-1, 0, 1}, same index as data.
        """

    def _validate_signals(
        self, signals: pd.Series, expected_index: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Validate and coerce the signal Series returned by generate_signals.

        Enforces:
            - Same length and index as the DataFeed
            - No NaN values
            - Values strictly in {-1, 0, 1}
        """
        if not isinstance(signals, pd.Series):
            raise StrategyError(
                f"generate_signals() must return a pd.Series, got {type(signals).__name__}."
            )

        if len(signals) != len(expected_index):
            raise StrategyError(
                f"Signal length {len(signals)} does not match DataFeed length {len(expected_index)}."
            )

        if signals.isnull().any():
            n = signals.isnull().sum()
            raise StrategyError(
                f"Signal Series contains {n} NaN value(s). Fill or drop before returning."
            )

        signals = signals.astype(int)
        invalid = signals[~signals.isin([-1, 0, 1])]
        if not invalid.empty:
            raise StrategyError(
                f"Signal values must be in {{-1, 0, 1}}. "
                f"Found invalid values: {invalid.unique().tolist()}"
            )

        signals.index = expected_index
        return signals

    def get_signals(self, feed: DataFeed) -> pd.Series:
        """
        Public method called by Backtest. Generates and validates signals.

        Returns a validated pd.Series of int in {-1, 0, 1}.
        """
        raw = self.generate_signals(feed.data)
        return self._validate_signals(raw, feed.data.index)


class RuleBasedStrategy(Strategy):
    """
    Strategy defined by a user-supplied Python function.

    The function receives the OHLCV DataFrame and must return a pd.Series
    of signals in {-1, 0, 1}.

    Parameters
    ----------
    rule_fn : Callable[[pd.DataFrame], pd.Series]
        A function that takes the DataFeed DataFrame and returns a signal Series.

    Example
    -------
    >>> def my_crossover(df):
    ...     signal = pd.Series(0, index=df.index)
    ...     signal[df['ema10'] > df['ema50']] = 1
    ...     signal[df['ema10'] < df['ema50']] = -1
    ...     return signal
    ...
    >>> strategy = RuleBasedStrategy(rule_fn=my_crossover)
    """

    def __init__(self, rule_fn: Callable[[pd.DataFrame], pd.Series]) -> None:
        if not callable(rule_fn):
            raise StrategyError(f"rule_fn must be callable, got {type(rule_fn).__name__}.")
        self.rule_fn = rule_fn

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        try:
            result = self.rule_fn(data)
        except Exception as exc:
            raise StrategyError(
                f"rule_fn raised an exception during signal generation: {exc}"
            ) from exc
        return result
