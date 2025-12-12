"""Test adaptive sizing and fractional backtest.

Verifies Kelly criterion, volatility sizing, and engine support for float signals.
"""
import pytest
import pandas as pd
import numpy as np
from openquant.risk.adaptive_sizing import kelly_criterion, volatility_target_sizing, AdaptiveSizer
from openquant.backtest.engine import backtest_signals

def test_kelly_criterion():
    # Win rate 50%, Win/Loss 2.0 -> f = 0.5 - 0.5/2 = 0.25
    k = kelly_criterion(0.5, 2.0, fraction=1.0)
    assert abs(k - 0.25) < 1e-6
    
    # Half Kelly
    k_half = kelly_criterion(0.5, 2.0, fraction=0.5)
    assert abs(k_half - 0.125) < 1e-6
    
    # Losing strategy -> 0
    k_loss = kelly_criterion(0.4, 1.0) # Edge = 0.4*1 - 0.6*1 = -0.2
    assert k_loss == 0.0

def test_volatility_sizing():
    # Current vol 40%, Target 20% -> Size 0.5
    size = volatility_target_sizing(0.40, target_volatility=0.20)
    assert abs(size - 0.5) < 1e-6
    
    # Current vol 10%, Target 20% -> Size 2.0 (capped at max_leverage=1.0 default)
    size_capped = volatility_target_sizing(0.10, target_volatility=0.20, max_leverage=1.0)
    assert size_capped == 1.0
    
    # With leverage
    size_lev = volatility_target_sizing(0.10, target_volatility=0.20, max_leverage=2.0)
    assert abs(size_lev - 2.0) < 1e-6

def test_adaptive_sizer_update():
    sizer = AdaptiveSizer(method="kelly")
    
    # Add some trades
    sizer.update(100) # Win
    sizer.update(-50) # Loss
    
    assert sizer.wins == 1
    assert sizer.losses == 1
    assert sizer.avg_win == 100.0
    assert sizer.avg_loss == 50.0
    
    # Not enough trades yet (<10)
    assert sizer.get_size() == 0.1

def test_fractional_backtest():
    # Create simple data
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1h')
    close = pd.Series(100.0, index=dates)
    # Price goes up 1% every step
    close = close * (1.01 ** np.arange(n))
    
    df = pd.DataFrame({'Close': close}, index=dates)
    
    # Signal 0.5 (Half position)
    signals = pd.Series(0.5, index=dates)
    
    result = backtest_signals(df, signals, fee_bps=0.0)
    
    # Returns should be 0.5 * market returns
    market_ret = df['Close'].pct_change().fillna(0)
    strat_ret = result.returns
    
    # Ignore first bar (no pos)
    np.testing.assert_allclose(strat_ret.iloc[1:], market_ret.iloc[1:] * 0.5, rtol=1e-5)
    
    # Positions should be 0.5
    assert (result.positions.iloc[1:] == 0.5).all()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
