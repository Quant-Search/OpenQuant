import pandas as pd
from openquant.strategies.rule_based.momentum import SMACrossoverStrategy


def test_sma_signals():
    idx = pd.date_range("2020-01-01", periods=50, freq="D", tz="UTC")
    price = pd.Series(range(50), index=idx)
    df = pd.DataFrame({"Close": price})
    s = SMACrossoverStrategy(fast=3, slow=10)
    sig = s.generate_signals(df)
    assert set(sig.unique()).issubset({0,1})
    assert sig.index.equals(df.index)

