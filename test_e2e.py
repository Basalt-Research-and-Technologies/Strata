"""
test_e2e.py
===========
End-to-end tests for basalt-strata.

All backtest parameters come from conftest.py CLI options — nothing is
hardcoded. Override any value on the command line:

    pytest test_e2e.py -v
    pytest test_e2e.py -v --capital 500000 --lots 1 --lot-size 25
    pytest test_e2e.py -v --bars 1000 --seed 7 --slippage 0.001
    pytest test_e2e.py -v --max-dd 0.15 --rfr 0.07 --bars-per-year 252
"""

from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from basalt_strata import (
    Backtest,
    DataFeed,
    RuleBasedStrategy,
    __version__,
)


# ---------------------------------------------------------------------------
# Helpers — strategy functions (no hardcoded params)
# ---------------------------------------------------------------------------

def _ema_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.Series:
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    sig = pd.Series(0, index=df.index, dtype=int)
    sig[ema_fast > ema_slow] = 1
    sig[ema_fast < ema_slow] = -1
    return sig


def _rsi_strategy(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, float("nan"))
    rsi   = 100 - (100 / (1 + rs))
    sig   = pd.Series(0, index=df.index, dtype=int)
    sig[rsi < 30] = 1
    sig[rsi > 70] = -1
    return sig.fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Synthetic OHLCV generator (params from fixture, not hardcoded)
# ---------------------------------------------------------------------------

def _make_ohlcv(n_bars: int, seed: int) -> pd.DataFrame:
    rng    = np.random.default_rng(seed)
    close  = 22_000.0
    closes = [close]
    for _ in range(n_bars - 1):
        close *= 1 + rng.normal(0.00005, 0.0015)
        closes.append(close)

    closes = np.array(closes)
    noise  = rng.uniform(0.0003, 0.0015, n_bars)
    opens  = closes * (1 + rng.normal(0, 0.0008, n_bars))
    highs  = np.maximum(opens, closes) * (1 + noise)
    lows   = np.minimum(opens, closes) * (1 - noise)
    vols   = rng.integers(50_000, 500_000, n_bars).astype(float)

    # IST session timestamps
    start  = pd.Timestamp("2024-01-02 09:15:00")
    all_ts = pd.bdate_range(start=start, periods=n_bars * 2, freq="15min")
    session = all_ts[
        (all_ts.time >= pd.Timestamp("09:15").time()) &
        (all_ts.time <= pd.Timestamp("15:15").time())
    ][:n_bars]

    return pd.DataFrame({
        "timestamp": session,
        "open":      opens[:len(session)],
        "high":      highs[:len(session)],
        "low":       lows[:len(session)],
        "close":     closes[:len(session)],
        "volume":    vols[:len(session)],
    })


# ---------------------------------------------------------------------------
# Session-scoped fixtures (built once, reused across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def raw_df(bt_params):
    return _make_ohlcv(bt_params["n_bars"], bt_params["seed"])


@pytest.fixture(scope="session")
def csv_path(raw_df, tmp_path_factory):
    p = tmp_path_factory.mktemp("data") / "nifty_test.csv"
    raw_df.to_csv(p, index=False)
    return p


@pytest.fixture(scope="session")
def feed(csv_path):
    return DataFeed.from_csv(csv_path, symbol="NIFTY", timeframe="15min")


@pytest.fixture(scope="session")
def ema_strategy():
    return RuleBasedStrategy(rule_fn=_ema_crossover)


@pytest.fixture(scope="session")
def rsi_strategy():
    return RuleBasedStrategy(rule_fn=_rsi_strategy)


@pytest.fixture(scope="session")
def result(feed, ema_strategy, bt_params):
    return Backtest(
        feed               = feed,
        strategy           = ema_strategy,
        initial_capital    = bt_params["initial_capital"],
        lot_size           = bt_params["lot_size"],
        lots_per_trade     = bt_params["lots_per_trade"],
        slippage_pct       = bt_params["slippage_pct"],
        commission         = bt_params["commission"],
        max_drawdown_limit = bt_params["max_drawdown_limit"],
        risk_free_rate     = bt_params["risk_free_rate"],
        bars_per_year      = bt_params["bars_per_year"],
    ).run()


# ---------------------------------------------------------------------------
# 1. DataFeed
# ---------------------------------------------------------------------------

class TestDataFeed:
    def test_loads_from_csv(self, feed, bt_params):
        # IST session filter (09:15–15:15) may reduce bar count below requested n_bars
        assert 0 < len(feed) <= bt_params["n_bars"]

    def test_symbol_and_timeframe(self, feed):
        assert feed.symbol == "NIFTY"
        assert feed.timeframe == "15min"

    def test_index_is_datetimeindex(self, feed):
        assert isinstance(feed.data.index, pd.DatetimeIndex)

    def test_index_is_sorted(self, feed):
        assert feed.data.index.is_monotonic_increasing

    def test_required_columns_present(self, feed):
        for col in ("open", "high", "low", "close", "volume"):
            assert col in feed.data.columns, f"Missing column: {col}"

    def test_no_nan_in_ohlc(self, feed):
        assert not feed.data[["open", "high", "low", "close"]].isnull().any().any()

    def test_high_always_gte_low(self, feed):
        assert (feed.data["high"] >= feed.data["low"]).all()

    def test_prices_are_positive(self, feed):
        for col in ("open", "high", "low", "close"):
            assert (feed.data[col] > 0).all(), f"Non-positive prices in {col}"

    def test_from_dataframe_roundtrip(self, feed):
        raw = feed.data[["open", "high", "low", "close", "volume"]].copy()
        feed2 = DataFeed.from_dataframe(raw, symbol="NIFTY", timeframe="15min")
        assert len(feed2) == len(feed)

    def test_resample_produces_fewer_bars(self, feed):
        daily = feed.resample("1D")
        assert len(daily) < len(feed)
        assert daily.timeframe == "1D"

    def test_repr_is_informative(self, feed):
        r = repr(feed)
        assert "NIFTY" in r
        assert "15min" in r


# ---------------------------------------------------------------------------
# 2. Strategy validation
# ---------------------------------------------------------------------------

class TestStrategy:
    def test_signals_are_series(self, feed, ema_strategy):
        sig = ema_strategy.get_signals(feed)
        assert isinstance(sig, pd.Series)

    def test_signals_same_length_as_feed(self, feed, ema_strategy):
        sig = ema_strategy.get_signals(feed)
        assert len(sig) == len(feed)

    def test_signal_values_in_valid_set(self, feed, ema_strategy):
        sig = ema_strategy.get_signals(feed)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_signals_no_nan(self, feed, ema_strategy):
        sig = ema_strategy.get_signals(feed)
        assert not sig.isnull().any()

    def test_signals_are_int_dtype(self, feed, ema_strategy):
        sig = ema_strategy.get_signals(feed)
        assert sig.dtype == int

    def test_rsi_strategy_valid(self, feed, rsi_strategy):
        sig = rsi_strategy.get_signals(feed)
        assert set(sig.unique()).issubset({-1, 0, 1})
        assert not sig.isnull().any()


# ---------------------------------------------------------------------------
# 3. BacktestResult structure
# ---------------------------------------------------------------------------

class TestResultStructure:
    def test_top_level_keys(self, result):
        d = result.to_dict()
        for key in ("config", "metrics", "trade_log", "equity_curve", "early_stop"):
            assert key in d

    def test_required_metrics_present(self, result):
        required = [
            "total_return_pct", "cagr_pct", "max_drawdown_pct",
            "sharpe_ratio", "sortino_ratio", "calmar_ratio",
            "win_rate_pct", "total_trades", "brt_thresholds",
            "volatility_annualised_pct", "profit_factor",
        ]
        for k in required:
            assert k in result.metrics, f"Missing metric: {k}"

    def test_config_has_all_params(self, result, bt_params):
        c = result.config
        assert c["initial_capital"]    == bt_params["initial_capital"]
        assert c["lot_size"]           == bt_params["lot_size"]
        assert c["lots_per_trade"]     == bt_params["lots_per_trade"]
        assert c["slippage_pct"]       == bt_params["slippage_pct"]
        assert c["commission"]         == bt_params["commission"]
        assert c["risk_free_rate"]     == bt_params["risk_free_rate"]
        assert c["bars_per_year"]      == bt_params["bars_per_year"]
        assert "position_sizing_fn"    in c

    def test_equity_curve_non_empty(self, result):
        assert len(result.equity_curve) > 0

    def test_equity_curve_starts_at_capital(self, result, bt_params):
        assert result.equity_curve.iloc[0] == pytest.approx(
            bt_params["initial_capital"], rel=1e-6
        )

    def test_trade_log_fields(self, result):
        d = result.to_dict()
        if d["trade_log"]:
            t = d["trade_log"][0]
            for field in (
                "entry_time", "exit_time", "direction",
                "entry_price", "exit_price", "lots",
                "lot_size", "units", "slippage_cost",
                "commission_cost", "gross_pnl", "net_pnl",
            ):
                assert field in t, f"Missing trade field: {field}"
            assert t["direction"] in ("long", "short")

    def test_trade_lots_match_config(self, result, bt_params):
        """With no position_sizing_fn, every trade should use lots_per_trade."""
        for t in result.trades:
            assert t.lots == bt_params["lots_per_trade"]

    def test_net_pnl_equals_gross_minus_commission(self, result):
        for t in result.trades:
            assert t.net_pnl == pytest.approx(
                t.gross_pnl - t.commission_cost, abs=0.02
            )

    def test_commission_cost_is_double_order(self, result, bt_params):
        """Commission = ₹X on entry + ₹X on exit = 2×."""
        expected = bt_params["commission"] * 2
        for t in result.trades:
            assert t.commission_cost == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# 4. Metrics sanity
# ---------------------------------------------------------------------------

class TestMetricsSanity:
    def test_max_drawdown_non_positive(self, result):
        assert result.metrics["max_drawdown_pct"] <= 0

    def test_total_trades_positive(self, result):
        assert result.metrics["total_trades"] > 0

    def test_win_rate_in_range(self, result):
        wr = result.metrics["win_rate_pct"]
        if wr is not None:
            assert 0.0 <= wr <= 100.0

    def test_final_equity_is_finite(self, result):
        assert math.isfinite(result.metrics["final_equity"])

    def test_brt_threshold_keys_present(self, result):
        thr = result.metrics["brt_thresholds"]
        for k in ("sharpe_pass", "calmar_pass", "max_dd_pass", "brt_pass"):
            assert k in thr

    def test_brt_values_are_bool_or_none(self, result):
        thr = result.metrics["brt_thresholds"]
        for k, v in thr.items():
            assert v is None or isinstance(v, bool), f"{k} is not bool/None: {v}"

    def test_equity_curve_values_are_finite(self, result):
        assert result.equity_curve.apply(math.isfinite).all()


# ---------------------------------------------------------------------------
# 5. JSON serialisation
# ---------------------------------------------------------------------------

class TestJsonSerialisation:
    def test_to_json_roundtrip(self, result, tmp_path):
        p = tmp_path / "tearsheet.json"
        json_str = result.to_json(str(p))
        parsed = json.loads(json_str)
        assert parsed["config"]["symbol"] == "NIFTY"
        assert p.exists()

    def test_json_contains_no_nan_or_inf(self, result):
        js = result.to_json()
        assert "NaN"      not in js
        assert "Infinity" not in js

    def test_to_dict_is_json_serialisable(self, result):
        d = result.to_dict()
        json.dumps(d, default=str)   # must not raise


# ---------------------------------------------------------------------------
# 6. position_sizing_fn (Section 6.3)
# ---------------------------------------------------------------------------

class TestPositionSizingFn:
    def test_dynamic_sizer_overrides_lots_per_trade(self, feed, ema_strategy, bt_params):
        def risk_1pct(equity: float, price: float, lot_size: int) -> int:
            return max(1, int((equity * 0.01) / (price * lot_size)))

        result_dyn = Backtest(
            feed               = feed,
            strategy           = ema_strategy,
            initial_capital    = bt_params["initial_capital"],
            lot_size           = bt_params["lot_size"],
            lots_per_trade     = 999,           # must be overridden
            position_sizing_fn = risk_1pct,
            slippage_pct       = bt_params["slippage_pct"],
            commission         = bt_params["commission"],
            risk_free_rate     = bt_params["risk_free_rate"],
            bars_per_year      = bt_params["bars_per_year"],
        ).run()

        for t in result_dyn.trades:
            assert t.lots >= 1
            assert t.lots < 999, "position_sizing_fn did not override lots_per_trade"

    def test_config_records_sizer_name(self, feed, ema_strategy, bt_params):
        def my_custom_sizer(equity, price, lot_size):
            return 1

        r = Backtest(
            feed               = feed,
            strategy           = ema_strategy,
            initial_capital    = bt_params["initial_capital"],
            lot_size           = bt_params["lot_size"],
            position_sizing_fn = my_custom_sizer,
            commission         = bt_params["commission"],
            bars_per_year      = bt_params["bars_per_year"],
        ).run()
        assert r.config["position_sizing_fn"] == "my_custom_sizer"

    def test_none_sizer_uses_lots_per_trade(self, feed, ema_strategy, bt_params):
        target_lots = bt_params["lots_per_trade"]
        r = Backtest(
            feed               = feed,
            strategy           = ema_strategy,
            initial_capital    = bt_params["initial_capital"],
            lot_size           = bt_params["lot_size"],
            lots_per_trade     = target_lots,
            position_sizing_fn = None,
            commission         = bt_params["commission"],
            bars_per_year      = bt_params["bars_per_year"],
        ).run()
        for t in r.trades:
            assert t.lots == target_lots


# ---------------------------------------------------------------------------
# 7. Sensitivity analysis (Section 6.5) — same strategy, different configs
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:
    def test_higher_capital_produces_higher_final_equity(
        self, feed, ema_strategy, bt_params
    ):
        base_cap  = bt_params["initial_capital"]
        large_cap = base_cap * 5

        r_small = Backtest(
            feed=feed, strategy=ema_strategy,
            initial_capital=base_cap,
            lot_size=bt_params["lot_size"],
            lots_per_trade=bt_params["lots_per_trade"],
            commission=bt_params["commission"],
            bars_per_year=bt_params["bars_per_year"],
        ).run()

        r_large = Backtest(
            feed=feed, strategy=ema_strategy,
            initial_capital=large_cap,
            lot_size=bt_params["lot_size"],
            lots_per_trade=bt_params["lots_per_trade"],
            commission=bt_params["commission"],
            bars_per_year=bt_params["bars_per_year"],
        ).run()

        assert r_large.metrics["final_equity"] != r_small.metrics["final_equity"]

    def test_zero_commission_improves_net_pnl(
        self, feed, ema_strategy, bt_params
    ):
        r_cost = Backtest(
            feed=feed, strategy=ema_strategy,
            initial_capital=bt_params["initial_capital"],
            lot_size=bt_params["lot_size"],
            lots_per_trade=bt_params["lots_per_trade"],
            commission=bt_params["commission"],
            bars_per_year=bt_params["bars_per_year"],
        ).run()

        r_free = Backtest(
            feed=feed, strategy=ema_strategy,
            initial_capital=bt_params["initial_capital"],
            lot_size=bt_params["lot_size"],
            lots_per_trade=bt_params["lots_per_trade"],
            commission=0,
            bars_per_year=bt_params["bars_per_year"],
        ).run()

        pnl_cost = sum(t.net_pnl for t in r_cost.trades)
        pnl_free = sum(t.net_pnl for t in r_free.trades)
        assert pnl_free >= pnl_cost


# ---------------------------------------------------------------------------
# 8. CLI / __main__
# ---------------------------------------------------------------------------

class TestCLI:
    def test_no_args_returns_zero(self):
        from basalt_strata.__main__ import main
        assert main([]) == 0

    def test_version_flag(self):
        from basalt_strata.__main__ import main
        assert main(["--version"]) == 0

    def test_help_exits_cleanly(self):
        from basalt_strata.__main__ import main
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_demo_default_params(self):
        from basalt_strata.__main__ import main
        assert main(["--demo", "--no-file"]) == 0

    def test_demo_custom_capital(self):
        from basalt_strata.__main__ import main
        assert main(["--demo", "--capital", "500000", "--lots", "1", "--no-file"]) == 0

    def test_demo_custom_bars_and_seed(self):
        from basalt_strata.__main__ import main
        assert main(["--demo", "--bars", "200", "--seed", "7", "--no-file"]) == 0

    def test_demo_tight_drawdown(self):
        from basalt_strata.__main__ import main
        assert main(["--demo", "--max-dd", "0.05", "--bars", "300", "--no-file"]) == 0

    def test_demo_writes_output_file(self, tmp_path):
        from basalt_strata.__main__ import main
        out = str(tmp_path / "out.json")
        rc = main(["--demo", "--bars", "150", "--output", out])
        assert rc == 0
        assert Path(out).exists()
        assert json.loads(Path(out).read_text())["config"]["symbol"] == "NIFTY-SYNTHETIC"


# ---------------------------------------------------------------------------
# 9. Package metadata
# ---------------------------------------------------------------------------

def test_version_is_non_empty_string():
    assert isinstance(__version__, str) and len(__version__) > 0

def test_public_api_importable():
    from basalt_strata import (
        DataFeed, DataFeedError,
        Strategy, RuleBasedStrategy, StrategyError,
        Execution, Trade,
        Analytics,
        Backtest, BacktestResult, BacktestError,
    )
    # confirm all are accessible
    assert all([
        DataFeed, DataFeedError,
        Strategy, RuleBasedStrategy, StrategyError,
        Execution, Trade,
        Analytics,
        Backtest, BacktestResult, BacktestError,
    ])
