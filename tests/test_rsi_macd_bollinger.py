import pandas as pd
import numpy as np
from openquant.features.ta_features import rsi, macd_features, bollinger_bands
from openquant.strategies.rule_based.rsi import RSICrossStrategy
from openquant.strategies.rule_based.macd import MACDCrossoverStrategy
from openquant.strategies.rule_based.bollinger import BollingerMeanReversionStrategy


def _df_linear(n=200):
    idx = pd.date_range("2021-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100, 200, n), index=idx)
    return pd.DataFrame({"Close": close})


def test_rsi_feature_bounds():
    df = _df_linear(100)
    r = rsi(df, 14)
    r = r.dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_macd_features_shape():
    df = _df_linear(100)
    m = macd_features(df)
    assert set(["macd","signal","hist"]) <= set(m.columns)
    assert m.index.equals(df.index)


def test_bollinger_features():
    df = _df_linear(60)
    bb = bollinger_bands(df, length=20, k=2.0)
    assert set(["mid","upper","lower","std"]) <= set(bb.columns)


def test_rsi_strategy_signals():
    df = _df_linear(80)
    strat = RSICrossStrategy(length=14, threshold=50)
    sig = strat.generate_signals(df)
    assert sig.index.equals(df.index)
    assert set(sig.unique()) <= {0,1}


def test_macd_strategy_signals():
    df = _df_linear(80)
    strat = MACDCrossoverStrategy(fast=12, slow=26, signal=9)
    sig = strat.generate_signals(df)
    assert sig.index.equals(df.index)
    assert set(sig.unique()) <= {0,1}


def test_bollinger_strategy_signals():
    df = _df_linear(80)
    strat = BollingerMeanReversionStrategy(length=20, k=2.0)
    sig = strat.generate_signals(df)
    assert sig.index.equals(df.index)
    assert set(sig.unique()) <= {0,1}

