import pandas as pd
import numpy as np
from openquant.evaluation.wfo import walk_forward_evaluate, WFOSpec
from openquant.strategies.registry import make_strategy


def _make_trending_df(n=200):
    rng = pd.date_range("2020-01-01", periods=n, freq="D")
    # Simple geometric Brownian motion drift
    drift = 0.0005
    vol = 0.01
    rets = np.random.normal(drift, vol, size=n)
    price = 100 * (1 + pd.Series(rets)).cumprod()
    df = pd.DataFrame({
        "Open": price.shift(1).fillna(price.iloc[0]),
        "High": price * 1.01,
        "Low": price * 0.99,
        "Close": price,
        "Volume": 1_000,
    }, index=rng)
    return df


def test_wfo_returns_keys_and_finite_values():
    df = _make_trending_df(240)
    grid = {"fast": [10, 20], "slow": [50, 100]}
    res = walk_forward_evaluate(
        df,
        lambda **pp: make_strategy("ema", **pp),
        grid,
        fee_bps=0.0,
        weight=1.0,
        wfo=WFOSpec(n_splits=3, train_frac=0.7),
    )
    assert set(["test_sharpes", "mean_test_sharpe", "best_params_per_split"]).issubset(res.keys())
    # Ensure finite numbers
    mts = float(res.get("mean_test_sharpe", 0.0))
    assert np.isfinite(mts)
    for v in res.get("test_sharpes", []):
        assert np.isfinite(v)

