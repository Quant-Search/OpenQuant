import numpy as np
import pandas as pd

from openquant.optimization.optuna_search import optuna_best_params
from openquant.strategies.registry import make_strategy
from openquant.backtest.engine import backtest_signals


def _make_df(n=200):
    idx = pd.date_range('2020-01-01', periods=n, freq='D', tz='UTC')
    price = np.linspace(100.0, 120.0, n) + np.random.normal(0, 0.5, size=n)
    df = pd.DataFrame({
        'Open': price,
        'High': price + 0.5,
        'Low': price - 0.5,
        'Close': price,
        'Volume': np.ones(n)
    }, index=idx)
    return df


def test_optuna_best_params_returns_valid_choice():
    df = _make_df()
    grid = {"fast": [3, 5], "slow": [10, 12]}
    best = optuna_best_params("ema", df, grid, fee_bps=0.0, weight=1.0, timeframe='1d', n_trials=5)
    assert set(best.keys()) == set(grid.keys())
    assert best['fast'] in grid['fast'] and best['slow'] in grid['slow']
    # Backtest with chosen params to ensure it runs
    strat = make_strategy('ema', **best)
    sig = strat.generate_signals(df)
    res = backtest_signals(df, sig, fee_bps=0.0, weight=1.0)
    assert len(res.returns) == len(df)

