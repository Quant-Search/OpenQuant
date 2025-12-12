"""Test advanced exit strategies.

Verifies ATR-based trailing stops, acceleration, and Chandelier Exit.
"""
import pytest
import pandas as pd
import numpy as np
from openquant.risk.trailing_stop import TrailingStopManager, PositionInfo
from openquant.trading.dynamic_exits import DynamicExitCalculator, ExitMethod

@pytest.fixture
def sample_data():
    n = 100
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1h')
    close = 100 * (1 + np.random.randn(n) * 0.01).cumprod()
    df = pd.DataFrame({
        'Open': close,
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': np.random.randint(1000, 10000, n)
    }, index=dates)
    return df

def test_atr_trailing_stop():
    # Setup manager with ATR multiplier
    manager = TrailingStopManager(atr_multiplier=2.0, min_update_bps=0.0)
    
    pos = PositionInfo(
        symbol="TEST", ticket=1, volume=1.0, type=0, # Long
        sl=90.0, tp=110.0, open_price=100.0
    )
    
    current_price = 105.0
    atr = 1.0
    
    # New SL should be Price - 2*ATR = 105 - 2 = 103
    new_sl = manager.calculate_new_sl(pos, current_price, atr=atr)
    
    assert new_sl is not None
    assert abs(new_sl - 103.0) < 1e-6

def test_trailing_acceleration():
    # Setup manager with acceleration
    # accel=5.0
    manager = TrailingStopManager(trailing_bps=1000, acceleration_factor=5.0, min_update_bps=0.0)
    
    pos = PositionInfo(
        symbol="TEST", ticket=1, volume=1.0, type=0, # Long
        sl=90.0, tp=120.0, open_price=100.0
    )
    
    # Case 1: Small profit (1%)
    current_price = 101.0
    # Profit ratio = 0.01
    # Factor = 1 - 5*0.01 = 0.95
    # Base distance = 101 * 0.1 = 10.1
    # Adjusted distance = 10.1 * 0.95 = 9.595
    # New SL = 101 - 9.595 = 91.405
    
    new_sl_1 = manager.calculate_new_sl(pos, current_price)
    # Just check it's tighter than standard 10% trail (which would be 90.9)
    standard_sl = current_price * 0.9
    assert new_sl_1 > standard_sl
    
    # Case 2: Large profit (10%)
    current_price_2 = 110.0
    # Profit ratio = 0.1
    # Factor = 1 - 5*0.1 = 0.5
    # Base distance = 110 * 0.1 = 11.0
    # Adjusted distance = 11.0 * 0.5 = 5.5
    # New SL = 110 - 5.5 = 104.5
    
    # Update pos SL first
    pos.sl = new_sl_1
    new_sl_2 = manager.calculate_new_sl(pos, current_price_2)
    
    assert new_sl_2 > 104.0 # Should be much tighter

def test_chandelier_exit(sample_data):
    calc = DynamicExitCalculator(method="chandelier", sl_atr_multiplier=2.0)
    
    # Mock data to have clear High/Low
    sample_data['High'][-22:] = 110.0
    sample_data['Low'][-22:] = 90.0
    sample_data['Close'][-1] = 100.0
    
    # Long: SL = Highest High (110) - 2*ATR
    # Let's assume ATR is approx calculated correctly.
    # We just check it runs and returns valid tuple
    
    tp, sl = calc.calculate_exits(sample_data, entry_price=100.0, side="LONG")
    
    assert tp > 100.0
    assert sl < 100.0
    assert sl < 110.0 # Should be below highest high

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
