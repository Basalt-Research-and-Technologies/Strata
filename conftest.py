"""
conftest.py
===========
Pytest configuration for basalt-strata tests.

All backtest parameters are configurable via CLI options so no values
are hardcoded in the test suite itself.

Usage examples
--------------
    pytest test_e2e.py -v
    pytest test_e2e.py -v --capital 500000 --lots 2 --lot-size 25
    pytest test_e2e.py -v --bars 1000 --seed 99
    pytest test_e2e.py -v --slippage 0.001 --commission 0 --max-dd 0.15
"""

from __future__ import annotations
import pytest


# ---------------------------------------------------------------------------
# Register custom CLI options — every backtest param is user-controllable
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    grp = parser.getgroup("basalt_strata", "Basalt Strata backtest parameters")

    # Data generation
    grp.addoption("--bars",   type=int,   default=2000,
                  help="Number of synthetic bars to generate (default: 2000)")
    grp.addoption("--seed",   type=int,   default=42,
                  help="Random seed for synthetic data (default: 42)")

    # Backtest config
    grp.addoption("--capital",       type=float, default=1_000_000,
                  help="Starting capital in ₹ (default: 1000000)")
    grp.addoption("--lot-size",      type=int,   default=50,
                  help="Units per lot (default: 50)")
    grp.addoption("--lots",          type=int,   default=2,
                  help="Lots per trade signal (default: 2)")
    grp.addoption("--slippage",      type=float, default=0.0005,
                  help="Slippage fraction of price (default: 0.0005)")
    grp.addoption("--commission",    type=float, default=20.0,
                  help="Flat ₹ per order (default: 20)")
    grp.addoption("--max-dd",        type=float, default=0.20,
                  help="Max drawdown limit fraction (default: 0.20)")
    grp.addoption("--rfr",           type=float, default=0.065,
                  help="Annual risk-free rate (default: 0.065)")
    grp.addoption("--bars-per-year", type=int,   default=6300,
                  help="Bars per year for annualisation (default: 6300 = 15-min NSE)")


# ---------------------------------------------------------------------------
# Fixtures that expose the CLI options to tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def bt_params(request) -> dict:
    """
    Single dict holding all user-chosen backtest parameters.
    Passed into every test that needs backtest config.
    """
    return {
        "n_bars":         request.config.getoption("--bars"),
        "seed":           request.config.getoption("--seed"),
        "initial_capital":request.config.getoption("--capital"),
        "lot_size":       request.config.getoption("--lot-size"),
        "lots_per_trade": request.config.getoption("--lots"),
        "slippage_pct":   request.config.getoption("--slippage"),
        "commission":     request.config.getoption("--commission"),
        "max_drawdown_limit": request.config.getoption("--max-dd"),
        "risk_free_rate": request.config.getoption("--rfr"),
        "bars_per_year":  request.config.getoption("--bars-per-year"),
    }
