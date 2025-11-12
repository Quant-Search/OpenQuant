import pandas as pd
import numpy as np
from openquant.validation.data_validator import validate_ohlcv, is_valid_ohlcv
from openquant.validation.strategy_validator import validate_signals
from openquant.validation.backtest_validator import validate_backtest


def _df_ok(n=50):
    idx = pd.date_range('2020-01-01', periods=n, freq='D', tz='UTC')
    px = np.linspace(100, 110, n)
    df = pd.DataFrame({
        'Open': px,
        'High': px + 1,
        'Low': px - 1,
        'Close': px + 0.5,
        'Volume': np.ones(n)
    }, index=idx)
    return df


def test_validate_ohlcv_ok():
    df = _df_ok()
    issues = validate_ohlcv(df)
    assert issues == []
    assert is_valid_ohlcv(df)


def test_validate_signals_ok():
    df = _df_ok()
    sig = pd.Series(0, index=df.index)
    issues = validate_signals(df, sig)
    assert issues == []


def test_validate_backtest_ok():
    class Res:
        def __init__(self, ret):
            self.returns = pd.Series(ret)
            self.equity_curve = (1 + self.returns.fillna(0)).cumprod()
            self.trades = pd.Series(np.zeros_like(ret))
    res = Res(np.random.normal(0, 0.01, size=50))
    issues = validate_backtest(res)
    assert issues == []

