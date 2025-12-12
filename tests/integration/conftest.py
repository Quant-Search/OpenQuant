"""Fixtures for integration tests."""
from __future__ import annotations
from typing import Any, Dict
from unittest.mock import MagicMock, Mock
import pandas as pd
import numpy as np
import pytest


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create a sample OHLCV DataFrame for testing."""
    n = 200
    idx = pd.date_range('2023-01-01', periods=n, freq='1h', tz='UTC')
    
    base_price = 100.0
    trend = np.linspace(0, 10, n)
    noise = np.random.RandomState(42).normal(0, 2, n)
    close = base_price + trend + noise
    
    df = pd.DataFrame({
        'Open': close - np.random.RandomState(42).uniform(0.1, 0.5, n),
        'High': close + np.random.RandomState(43).uniform(0.1, 1.0, n),
        'Low': close - np.random.RandomState(44).uniform(0.1, 1.0, n),
        'Close': close,
        'Volume': np.random.RandomState(45).uniform(1000, 10000, n),
    }, index=idx)
    
    return df


@pytest.fixture
def mock_mt5_module():
    """Create a mock MetaTrader5 module with all required attributes."""
    mt5_mock = MagicMock()
    
    mt5_mock.initialize = Mock(return_value=True)
    mt5_mock.shutdown = Mock(return_value=None)
    mt5_mock.login = Mock(return_value=True)
    
    account_info = Mock()
    account_info.equity = 100000.0
    account_info.balance = 100000.0
    account_info.margin = 0.0
    account_info.login = 12345
    mt5_mock.account_info = Mock(return_value=account_info)
    
    mt5_mock.positions_get = Mock(return_value=[])
    
    symbol_info = Mock()
    symbol_info.bid = 1.1000
    symbol_info.ask = 1.1002
    symbol_info.trade_contract_size = 100000.0
    symbol_info.volume_min = 0.01
    symbol_info.volume_max = 100.0
    symbol_info.volume_step = 0.01
    mt5_mock.symbol_info = Mock(return_value=symbol_info)
    
    tick = Mock()
    tick.bid = 1.1000
    tick.ask = 1.1002
    tick.last = 1.1001
    mt5_mock.symbol_info_tick = Mock(return_value=tick)
    
    mt5_mock.symbol_select = Mock(return_value=True)
    
    order_result = Mock()
    order_result.retcode = 10009
    order_result.order = 12345
    order_result.volume = 0.1
    order_result.price = 1.1001
    order_result.comment = "Success"
    mt5_mock.order_send = Mock(return_value=order_result)
    
    mt5_mock.TRADE_ACTION_DEAL = 1
    mt5_mock.TRADE_ACTION_SLTP = 2
    mt5_mock.ORDER_TYPE_BUY = 0
    mt5_mock.ORDER_TYPE_SELL = 1
    mt5_mock.TRADE_RETCODE_DONE = 10009
    
    terminal_info = Mock()
    terminal_info.data_path = "C:/Program Files/MetaTrader 5/MQL5/Files"
    mt5_mock.terminal_info = Mock(return_value=terminal_info)
    
    mt5_mock.last_error = Mock(return_value=(0, "Success"))
    
    return mt5_mock


@pytest.fixture
def mock_data_loader():
    """Create a mock DataLoader."""
    from openquant.data.loader import DataLoader
    
    loader = DataLoader()
    
    original_get_ohlcv = loader.get_ohlcv
    
    def mock_get_ohlcv(source, symbol, timeframe='1d', start=None, end=None, limit=None):
        n = 200
        idx = pd.date_range('2023-01-01', periods=n, freq='1h', tz='UTC')
        base = 100.0 + hash(symbol) % 20
        trend = np.linspace(0, 10, n)
        noise = np.random.RandomState(hash(symbol) % 1000).normal(0, 2, n)
        close = base + trend + noise
        
        return pd.DataFrame({
            'Open': close,
            'High': close + 0.5,
            'Low': close - 0.5,
            'Close': close,
            'Volume': 1000,
        }, index=idx)
    
    loader.get_ohlcv = Mock(side_effect=mock_get_ohlcv)
    
    return loader


@pytest.fixture
def temp_state_files(tmp_path):
    """Create temporary state files for kill switch and circuit breaker."""
    kill_switch_file = tmp_path / "STOP"
    circuit_breaker_file = tmp_path / "circuit_breaker_state.json"
    
    return {
        'kill_switch': kill_switch_file,
        'circuit_breaker': circuit_breaker_file,
        'temp_dir': tmp_path,
    }


@pytest.fixture
def clean_risk_state(temp_state_files):
    """Ensure clean state for risk management tests."""
    from openquant.risk.kill_switch import KILL_SWITCH
    from openquant.risk.circuit_breaker import CIRCUIT_BREAKER
    
    KILL_SWITCH.trigger_file = temp_state_files['kill_switch']
    CIRCUIT_BREAKER.state_file = temp_state_files['circuit_breaker']
    
    if temp_state_files['kill_switch'].exists():
        temp_state_files['kill_switch'].unlink()
    
    CIRCUIT_BREAKER.reset()
    
    yield
    
    if temp_state_files['kill_switch'].exists():
        temp_state_files['kill_switch'].unlink()


@pytest.fixture
def small_param_grid() -> Dict[str, Any]:
    """Small parameter grid for fast testing."""
    return {
        "process_noise": [1e-5, 1e-4],
        "measurement_noise": [1e-3, 1e-2],
        "threshold": [1.0, 1.5],
    }
