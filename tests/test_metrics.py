import numpy as np
import pandas as pd
from openquant.backtest.metrics import sharpe, max_drawdown


def test_sharpe_basic():
    r = pd.Series([0.01] * 100)
    s = sharpe(r, freq="1d")
    assert s > 0


def test_max_drawdown():
    ec = pd.Series([1.0, 1.1, 1.05, 1.2, 1.0])
    dd = max_drawdown(ec)
    assert dd < 0  # negative drawdown

