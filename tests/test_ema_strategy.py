import pandas as pd
from openquant.features.ta_features import ema
from openquant.strategies.rule_based.ema import EMACrossoverStrategy

def test_ema_feature_monotonic():
    idx = pd.date_range("2021-01-01", periods=20, freq="D", tz="UTC")
    df = pd.DataFrame({"Close": pd.Series(range(100, 120), index=idx)})
    e10 = ema(df, 10)
    # EMA of a strictly increasing series should be non-decreasing after warmup
    d = e10.dropna().diff().fillna(0)
    assert d.min() >= -1e-9


def test_ema_crossover_signals_shape():
    idx = pd.date_range("2021-01-01", periods=60, freq="D", tz="UTC")
    close = pd.Series(range(100, 160), index=idx)
    df = pd.DataFrame({"Close": close})
    strat = EMACrossoverStrategy(fast=10, slow=30)
    sig = strat.generate_signals(df)
    assert list(sig.unique()) == [0, 1] or list(sig.unique()) == [1, 0] or  set(sig.unique()) == {0,1}
    assert sig.index.equals(df.index)

