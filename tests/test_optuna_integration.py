"""Test Optuna hyperparameter optimization integration.

Tests that optuna_best_params function correctly searches the parameter grid
and returns valid parameter combinations for quantitative strategies.
"""
import numpy as np
import pandas as pd

from openquant.optimization.optuna_search import optuna_best_params
from openquant.strategies.registry import make_strategy
from openquant.backtest.engine import backtest_signals


def _make_df(n=200):
    """Create synthetic OHLCV data for testing."""
    idx = pd.date_range('2020-01-01', periods=n, freq='D', tz='UTC')
    # Create trending price for Kalman to work with
    price = np.linspace(100.0, 120.0, n) + np.random.normal(0, 0.5, size=n)
    df = pd.DataFrame({
        'Open': price,
        'High': price + 0.5,
        'Low': price - 0.5,
        'Close': price,
        'Volume': np.ones(n) * 1000
    }, index=idx)
    return df


def test_optuna_best_params_returns_valid_choice():
    """Test that Optuna returns valid parameters from the Hurst strategy grid.

    Uses Hurst strategy since we removed retail TA indicators (EMA, SMA, etc.)
    The grid uses integer values which Optuna will sample categorically.

    HurstExponentStrategy accepts: lookback, trend_threshold, mr_threshold
    """
    df = _make_df()

    # Use Hurst strategy parameters (lookback is an integer)
    # Small grids (<=5 values) use categorical sampling to ensure exact matches
    grid = {"lookback": [50, 100]}

    best = optuna_best_params(
        "hurst", df, grid,
        fee_bps=0.0, weight=1.0, timeframe='1d', n_trials=5
    )

    # Verify returned keys match grid keys
    assert set(best.keys()) == set(grid.keys())

    # Verify returned value is from the grid
    assert best['lookback'] in grid['lookback'], \
        f"Returned lookback={best['lookback']} not in grid {grid['lookback']}"

    # Backtest with chosen params to ensure strategy runs
    strat = make_strategy('hurst', **best)
    sig = strat.generate_signals(df)
    res = backtest_signals(df, sig, fee_bps=0.0, weight=1.0)
    assert len(res.returns) == len(df)

