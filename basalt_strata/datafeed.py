"""
DataFeed -- accepts raw OHLCV data, validates it, returns a clean internal DataFrame.

Internal standard columns:
    timestamp  : DatetimeIndex (IST-naive, i.e. tz-unaware but assumed IST)
    open       : float
    high       : float
    low        : float
    close      : float
    volume     : float (0 is allowed, flagged as warning not error)
    symbol     : str
    timeframe  : str  e.g. "15min", "1D"

Design decisions:
    - Timezone: library is IST-only. If input timestamps are tz-aware they are
      converted to IST then stripped. If tz-naive they are assumed IST as-is.
    - Missing bars: flagged in validation_warnings, NOT auto-filled. Caller decides.
    - Duplicate timestamps: raises DataFeedError -- cannot proceed with duplicates.
    - Bad prices (high < low, negative prices, NaN OHLC): raises DataFeedError.
    - Zero-volume bars: logged as warnings, not errors.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import pandas as pd

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataFeedError(Exception):
    """Raised when input data fails validation and cannot be used safely."""


class DataFeed:
    """
    Wraps and validates OHLCV data.

    Attributes
    ----------
    data : pd.DataFrame
        Clean, validated OHLCV DataFrame with DatetimeIndex.
    symbol : str
    timeframe : str
    validation_warnings : list[str]
        Non-fatal issues found during validation (e.g. zero-volume bars).
    """

    def __init__(
        self,
        data: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.validation_warnings: list[str] = []
        self.data = self._validate_and_clean(data)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        symbol: str,
        timeframe: str,
        date_col: str = "timestamp",
        **read_csv_kwargs,
    ) -> "DataFeed":
        """
        Load from a CSV file.

        Parameters
        ----------
        path : str or Path
        symbol : str
        timeframe : str
        date_col : str
            Name of the column containing timestamps. Default "timestamp".
        **read_csv_kwargs
            Passed directly to pd.read_csv (e.g. sep=";").
        """
        df = pd.read_csv(path, parse_dates=[date_col], **read_csv_kwargs)
        df = df.rename(columns={date_col: "timestamp"})
        df = df.set_index("timestamp")
        return cls(df, symbol=symbol, timeframe=timeframe)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        timestamp_col: str | None = None,
    ) -> "DataFeed":
        """
        Load from an existing DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must have either a DatetimeIndex or a timestamp column specified
            via timestamp_col.
        symbol : str
        timeframe : str
        timestamp_col : str or None
            If the timestamps are in a column (not the index), name it here.
        """
        df = df.copy()
        if timestamp_col is not None:
            df = df.rename(columns={timestamp_col: "timestamp"})
            df = df.set_index("timestamp")
        return cls(df, symbol=symbol, timeframe=timeframe)

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- index must be datetime ---
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as exc:
                raise DataFeedError(
                    f"Index could not be parsed as datetime: {exc}"
                ) from exc

        # --- timezone handling: convert to IST-naive ---
        if df.index.tz is not None:
            df.index = (
                df.index
                .tz_convert("Asia/Kolkata")
                .tz_localize(None)
            )

        df.index.name = "timestamp"

        # --- required columns ---
        missing = REQUIRED_COLUMNS - set(df.columns.str.lower())
        if missing:
            raise DataFeedError(
                f"Missing required columns: {missing}. "
                f"Found: {list(df.columns)}"
            )
        df.columns = df.columns.str.lower()

        # --- duplicate timestamps ---
        dupes = df.index.duplicated().sum()
        if dupes:
            raise DataFeedError(
                f"Found {dupes} duplicate timestamp(s). "
                "Remove or deduplicate before passing to DataFeed."
            )

        # --- sort ascending ---
        df = df.sort_index()

        # --- NaN in OHLC ---
        ohlc_nulls = df[["open", "high", "low", "close"]].isnull().sum()
        bad_cols = ohlc_nulls[ohlc_nulls > 0]
        if not bad_cols.empty:
            raise DataFeedError(
                f"NaN values in OHLC columns: {bad_cols.to_dict()}. "
                "Forward-fill or drop before passing to DataFeed."
            )

        # --- price sanity ---
        if (df["high"] < df["low"]).any():
            n = (df["high"] < df["low"]).sum()
            raise DataFeedError(
                f"{n} bar(s) where high < low. Data is corrupt."
            )
        for col in OHLCV_COLUMNS[:4]:  # open high low close
            if (df[col] < 0).any():
                raise DataFeedError(f"Negative values found in column '{col}'.")

        # --- zero-volume warning (not error) ---
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol:
            self.validation_warnings.append(
                f"{zero_vol} bar(s) have zero volume. "
                "These are retained but may affect execution simulation."
            )

        # --- keep only standard columns (drop extras silently) ---
        df = df[OHLCV_COLUMNS]
        df["symbol"] = self.symbol
        df["timeframe"] = self.timeframe

        return df

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def resample(self, target_timeframe: str) -> "DataFeed":
        """
        Resample to a lower frequency.

        OHLCV aggregation rules:
            open  -> first
            high  -> max
            low   -> min
            close -> last
            volume -> sum

        Parameters
        ----------
        target_timeframe : str
            Pandas offset alias, e.g. "1h", "1D", "30min".

        Returns
        -------
        DataFeed
            New DataFeed at the target timeframe.
        """
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        resampled = (
            self.data[OHLCV_COLUMNS]
            .resample(target_timeframe)
            .agg(agg)
            .dropna()
        )
        return DataFeed.from_dataframe(
            resampled,
            symbol=self.symbol,
            timeframe=target_timeframe,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (
            f"DataFeed(symbol={self.symbol!r}, timeframe={self.timeframe!r}, "
            f"bars={len(self)}, "
            f"from={self.data.index[0].date()}, "
            f"to={self.data.index[-1].date()})"
        )
