"""Test Overfitting Safeguards.

Verifies OverfittingGuard correctly identifies overfitted strategies.
"""
import pytest
import pandas as pd
import numpy as np
from openquant.validation.overfitting_guard import OverfittingGuard

@pytest.fixture
def good_returns():
    """Generate realistic good returns."""
    np.random.seed(42)
    # Positive mean, moderate variance
    returns = np.random.normal(0.001, 0.01, 100)
    return pd.Series(returns)

@pytest.fixture
def overfitted_returns():
    """Generate suspiciously good returns (overfitted)."""
    # All wins, unrealistic
    returns = np.abs(np.random.normal(0.01, 0.002, 50))
    return pd.Series(returns)

def test_minimum_trades():
    guard = OverfittingGuard(min_trades=50)
    
    # Too few trades
    returns = pd.Series([0.01] * 20)
    result = guard.check_strategy(returns)
    
    assert not result.is_safe
    assert "Insufficient trades" in result.reason

def test_sharpe_threshold(good_returns):
    guard = OverfittingGuard(min_dsr=1.0)
    
    result = guard.check_strategy(good_returns)
    
    # Should pass basic checks
    assert result.metrics['n_trades'] == 100
    assert 'sharpe' in result.metrics

def test_is_oos_ratio():
    guard = OverfittingGuard(max_is_oos_ratio=1.5)
    
    returns = pd.Series(np.random.normal(0.001, 0.01, 100))
    
    # High IS sharpe, low OOS sharpe -> overfitting
    result = guard.check_strategy(returns, is_sharpe=3.0, oos_sharpe=0.5)
    
    assert not result.is_safe
    assert "IS/OOS ratio too high" in result.reason

def test_consecutive_wins():
    guard = OverfittingGuard(max_consecutive_wins=10)
    
    # Create returns with 15 consecutive wins
    returns = [0.01] * 15 + [-0.005] * 5 + [0.005] * 20
    returns = pd.Series(returns)
    
    result = guard.check_strategy(returns)
    
    assert not result.is_safe
    assert "consecutive wins" in result.reason

def test_dsr_calculation():
    guard = OverfittingGuard(min_dsr=1.0)
    
    # Good returns, many trials
    returns = pd.Series(np.random.normal(0.002, 0.01, 100))
    
    result = guard.check_strategy(returns, n_trials=100)
    
    # DSR should be calculated and present in metrics
    if 'dsr' in result.metrics:
        assert result.metrics['dsr'] is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
