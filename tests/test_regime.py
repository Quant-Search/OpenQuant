import numpy as np
import pandas as pd
from openquant.evaluation.regime import compute_regime_features


def _make_price_series_trending(n=200, slope=0.001):
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    noise = np.random.normal(scale=0.002, size=n)
    price = 100.0 * (1.0 + slope) ** np.arange(n) * (1 + noise).cumprod()
    return pd.DataFrame({"Close": price}, index=idx)


def _make_price_series_random(n=200):
    idx = pd.date_range("2020-01-01", periods=n, freq="h")
    noise = np.random.normal(scale=0.002, size=n)
    price = 100.0 * (1 + noise).cumprod()
    return pd.DataFrame({"Close": price}, index=idx)


def test_regime_features_finite():
    df = _make_price_series_trending()
    feats = compute_regime_features(df)
    assert set(["trend_score", "volatility"]).issubset(feats.keys())
    assert np.isfinite(feats["trend_score"]) and np.isfinite(feats["volatility"]) 


def test_trending_has_higher_trend_score_than_random():
    df_tr = _make_price_series_trending()
    df_rd = _make_price_series_random()
    sc_tr = compute_regime_features(df_tr)["trend_score"]
    sc_rd = compute_regime_features(df_rd)["trend_score"]
    assert sc_tr > sc_rd - 1e-9

