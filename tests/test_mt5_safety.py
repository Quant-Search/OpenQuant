import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from unittest.mock import MagicMock, patch
from openquant.paper import mt5_bridge

def test_validate_credentials_valid():
    """Test validation with valid inputs."""
    valid, msg = mt5_bridge.validate_credentials(login=123456, server="MetaQuotes-Demo")
    assert valid
    assert msg == "OK"

def test_validate_credentials_invalid_login():
    """Test validation with non-integer login."""
    valid, msg = mt5_bridge.validate_credentials(login="abc", server="MetaQuotes-Demo")
    assert not valid
    assert "Login must be an integer" in msg

def test_validate_credentials_missing_server():
    """Test validation with login but no server."""
    valid, msg = mt5_bridge.validate_credentials(login=123456)
    assert not valid
    assert "Server must be provided" in msg

def test_validate_credentials_bad_path():
    """Test validation with non-existent path."""
    valid, msg = mt5_bridge.validate_credentials(terminal_path="/non/existent/path/terminal64.exe")
    assert not valid
    assert "Terminal path does not exist" in msg

@patch("openquant.paper.mt5_bridge._lazy_import")
def test_init_missing_module(mock_import):
    """Test init when MetaTrader5 module is missing."""
    mock_import.return_value = None
    assert not mt5_bridge.init()

@patch("openquant.paper.mt5_bridge._lazy_import")
def test_init_connection_failure(mock_import):
    """Test init when mt5.initialize fails."""
    mock_mt5 = MagicMock()
    mock_mt5.initialize.return_value = False
    mock_mt5.last_error.return_value = (-1, "Generic Error")
    mock_import.return_value = mock_mt5
    
    # Ensure account_info raises or returns None to trigger init path
    mock_mt5.account_info.return_value = None
    
    assert not mt5_bridge.init()

@patch("openquant.paper.mt5_bridge._lazy_import")
def test_init_success(mock_import):
    """Test successful init."""
    mock_mt5 = MagicMock()
    mock_mt5.initialize.return_value = True
    mock_mt5.account_info.return_value = None # Simulate not connected initially
    mock_import.return_value = mock_mt5
    
    assert mt5_bridge.init()

@patch("openquant.paper.mt5_bridge._lazy_import")
def test_close_all_positions(mock_import):
    """Test emergency close logic."""
    mock_mt5 = MagicMock()
    mock_import.return_value = mock_mt5
    
    # Mock positions
    pos1 = MagicMock()
    pos1.symbol = "EURUSD"
    pos1.ticket = 100
    pos1.volume = 1.0
    pos1.type = 0 # BUY
    
    pos2 = MagicMock()
    pos2.symbol = "GBPUSD"
    pos2.ticket = 101
    pos2.volume = 0.5
    pos2.type = 1 # SELL
    
    mock_mt5.positions_get.return_value = [pos1, pos2]
    
    # Mock ticks
    tick = MagicMock()
    tick.bid = 1.05
    tick.ask = 1.06
    mock_mt5.symbol_info_tick.return_value = tick
    
    # Mock order result
    res = MagicMock()
    res.retcode = 10009 # TRADE_RETCODE_DONE
    mock_mt5.order_send.return_value = res
    mock_mt5.TRADE_RETCODE_DONE = 10009
    mock_mt5.ORDER_TYPE_BUY = 0
    mock_mt5.ORDER_TYPE_SELL = 1
    mock_mt5.TRADE_ACTION_DEAL = 1
    
    count = mt5_bridge.close_all_positions()
    assert count == 2
    assert mock_mt5.order_send.call_count == 2
