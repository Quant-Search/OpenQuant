import pandas as pd
from openquant.backtest.engine import backtest_signals


def test_backtest_long_trend():
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    close = pd.Series(range(100, 110), index=idx)
    df = pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1})
    sig = pd.Series(1, index=idx)
    res = backtest_signals(df, sig, fee_bps=0.0)
    assert res.equity_curve.iloc[-1] > 1.0
    assert res.trades.sum() == 1  # initial entry




def test_position_weight_scales_returns():
    idx = pd.date_range("2020-01-01", periods=10, freq="D", tz="UTC")
    close = pd.Series(range(100, 110), index=idx)
    df = pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1})
    sig = pd.Series(1, index=idx)

    full = backtest_signals(df, sig, fee_bps=0.0, weight=1.0)
    half = backtest_signals(df, sig, fee_bps=0.0, weight=0.5)

    assert half.equity_curve.iloc[-1] > 1.0
    assert full.equity_curve.iloc[-1] > half.equity_curve.iloc[-1]


def test_weight_scales_fees():
    idx = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")
    close = pd.Series([100.0, 100.0, 100.0], index=idx)
    df = pd.DataFrame({"Close": close, "Open": close, "High": close, "Low": close, "Volume": 1})
    sig = pd.Series(1, index=idx)

    res = backtest_signals(df, sig, fee_bps=50.0, weight=0.5)  # 0.5% fee scaled by weight 0.5 => 0.25%
    expected_end = (1.0 - 0.0025)  # only one entry trade
    assert abs(res.equity_curve.iloc[-1] - expected_end) < 1e-6
