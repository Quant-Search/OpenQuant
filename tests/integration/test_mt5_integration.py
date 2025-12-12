"""Integration test: MT5 Broker integration with mocking."""
from __future__ import annotations
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import pytest
from openquant.paper.state import PortfolioState


def test_mt5_initialization_mock(mock_mt5_module):
    """Test MT5 initialization with mocked module."""
    with patch.dict('sys.modules', {'MetaTrader5': mock_mt5_module}):
        from openquant.paper import mt5_bridge
        
        mt5_bridge._MT5 = mock_mt5_module
        
        result = mt5_bridge.init(login=12345, password="test", server="Demo")
        
        assert result is True
        mock_mt5_module.initialize.assert_called()


def test_mt5_account_info_mock(mock_mt5_module):
    """Test retrieving account info from mocked MT5."""
    with patch.dict('sys.modules', {'MetaTrader5': mock_mt5_module}):
        from openquant.paper import mt5_bridge
        
        mt5_bridge._MT5 = mock_mt5_module
        
        equity = mt5_bridge.account_equity(mock_mt5_module)
        
        assert equity == 100000.0


def test_mt5_positions_mock(mock_mt5_module):
    """Test retrieving positions from mocked MT5."""
    position_mock = Mock()
    position_mock.symbol = "EURUSD"
    position_mock.volume = 0.1
    position_mock.type = 0
    
    mock_mt5_module.positions_get = Mock(return_value=[position_mock])
    
    with patch.dict('sys.modules', {'MetaTrader5': mock_mt5_module}):
        from openquant.paper import mt5_bridge
        
        mt5_bridge._MT5 = mock_mt5_module
        
        positions = mt5_bridge.positions_by_symbol(mock_mt5_module)
        
        assert "EURUSD" in positions
        assert positions["EURUSD"] == 0.1


def test_mt5_order_send_mock(mock_mt5_module):
    """Test sending orders through mocked MT5."""
    with patch.dict('sys.modules', {'MetaTrader5': mock_mt5_module}):
        from openquant.paper import mt5_bridge
        
        mt5_bridge._MT5 = mock_mt5_module
        
        allocation = [
            {
                "symbol": "BTC/USDT",
                "weight": 0.1,
                "sl": 0.0,
                "tp": 0.0,
            }
        ]
        
        mt5_bridge.init = Mock(return_value=True)
        
        try:
            targets = mt5_bridge.apply_allocation_to_mt5(
                allocation,
                volume_min_floor=0.01,
                login=12345,
                password="test",
                server="Demo"
            )
            
            assert isinstance(targets, dict)
        except RuntimeError:
            pass


def test_mt5_symbol_mapping(mock_mt5_module):
    """Test symbol mapping from CCXT to MT5 format."""
    with patch.dict('sys.modules', {'MetaTrader5': mock_mt5_module}):
        from openquant.paper import mt5_bridge
        
        mt5_bridge._MT5 = mock_mt5_module
        
        test_cases = [
            ("BTC/USDT", "BTCUSD"),
            ("ETH/USDT", "ETHUSD"),
            ("EUR/USD", "EURUSD"),
        ]
        
        for binance_symbol, expected_mt5 in test_cases:
            result = mt5_bridge.map_symbol(binance_symbol)
            assert result == expected_mt5


def test_mt5_close_all_positions_mock(mock_mt5_module):
    """Test closing all positions through mocked MT5."""
    position1 = Mock()
    position1.symbol = "EURUSD"
    position1.volume = 0.1
    position1.type = 0
    position1.ticket = 123
    
    position2 = Mock()
    position2.symbol = "GBPUSD"
    position2.volume = 0.2
    position2.type = 1
    position2.ticket = 124
    
    mock_mt5_module.positions_get = Mock(return_value=[position1, position2])
    mock_mt5_module.ORDER_TYPE_BUY = 0
    mock_mt5_module.ORDER_TYPE_SELL = 1
    
    with patch.dict('sys.modules', {'MetaTrader5': mock_mt5_module}):
        from openquant.paper import mt5_bridge
        
        mt5_bridge._MT5 = mock_mt5_module
        
        count = mt5_bridge.close_all_positions()
        
        assert count == 2
        assert mock_mt5_module.order_send.call_count == 2


def test_mt5_modify_position_mock(mock_mt5_module):
    """Test modifying SL/TP of existing position."""
    position = Mock()
    position.symbol = "EURUSD"
    position.ticket = 123
    position.sl = 1.0900
    position.tp = 1.1100
    
    mock_mt5_module.positions_get = Mock(return_value=[position])
    
    with patch.dict('sys.modules', {'MetaTrader5': mock_mt5_module}):
        from openquant.paper import mt5_bridge
        
        mt5_bridge._MT5 = mock_mt5_module
        
        result = mt5_bridge.modify_position("EURUSD", sl=1.0950, tp=1.1150)
        
        assert result is True
        mock_mt5_module.order_send.assert_called_once()


def test_mt5_export_signals_mock(mock_mt5_module, tmp_path):
    """Test exporting signals to CSV for MT5 EA."""
    with patch.dict('sys.modules', {'MetaTrader5': mock_mt5_module}):
        from openquant.paper import mt5_bridge
        
        mt5_bridge._MT5 = mock_mt5_module
        
        allocations = [
            {"symbol": "EURUSD", "weight": 0.1},
            {"symbol": "GBPUSD", "weight": 0.2},
            {"symbol": "USDJPY", "weight": -0.1},
        ]
        
        output_path = tmp_path / "signals.csv"
        
        mt5_bridge.export_signals_to_csv(allocations, path=str(output_path))
        
        assert output_path.exists()
        
        content = output_path.read_text()
        assert "EURUSD" in content
        assert "GBPUSD" in content
        assert "BUY" in content
        assert "SELL" in content


def test_mt5_integration_with_paper_trading(mock_mt5_module, sample_ohlcv_df, clean_risk_state):
    """Test full integration: backtest → signals → MT5 allocation."""
    with patch.dict('sys.modules', {'MetaTrader5': mock_mt5_module}):
        from openquant.paper import mt5_bridge
        from openquant.strategies.registry import make_strategy
        from openquant.backtest.engine import backtest_signals
        
        mt5_bridge._MT5 = mock_mt5_module
        
        df = sample_ohlcv_df
        
        strategy = make_strategy('kalman', process_noise=1e-5, measurement_noise=1e-3, threshold=1.0)
        signals = strategy.generate_signals(df)
        result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
        
        last_signal = signals.iloc[-1]
        
        allocation = []
        if last_signal == 1:
            allocation.append({
                "symbol": "BTC/USDT",
                "weight": 0.1,
                "sl": 0.0,
                "tp": 0.0,
            })
        
        if allocation:
            mt5_bridge.init = Mock(return_value=True)
            
            try:
                targets = mt5_bridge.apply_allocation_to_mt5(
                    allocation,
                    volume_min_floor=0.01,
                    login=12345,
                    password="test",
                    server="Demo"
                )
                
                assert isinstance(targets, dict)
            except RuntimeError:
                pass


def test_mt5_error_handling_mock(mock_mt5_module):
    """Test error handling in MT5 integration."""
    mock_mt5_module.initialize = Mock(return_value=False)
    
    with patch.dict('sys.modules', {'MetaTrader5': mock_mt5_module}):
        from openquant.paper import mt5_bridge
        
        mt5_bridge._MT5 = mock_mt5_module
        
        result = mt5_bridge.init(login=12345, password="test", server="Demo")
        
        assert result is False


def test_mt5_credential_validation():
    """Test credential validation before connection."""
    from openquant.paper.mt5_bridge import validate_credentials
    
    is_valid, msg = validate_credentials(login=12345, server="Demo", terminal_path="/path/to/mt5")
    assert is_valid is True
    
    is_valid, msg = validate_credentials(login="invalid", server="Demo")
    assert is_valid is False
    
    is_valid, msg = validate_credentials(login=12345, server=None)
    assert is_valid is False


def test_mt5_with_risk_checks(mock_mt5_module, clean_risk_state, temp_state_files):
    """Test MT5 integration respects risk management controls."""
    from openquant.risk.kill_switch import KILL_SWITCH
    
    with patch.dict('sys.modules', {'MetaTrader5': mock_mt5_module}):
        from openquant.paper import mt5_bridge
        
        mt5_bridge._MT5 = mock_mt5_module
        mt5_bridge.init = Mock(return_value=True)
        
        allocation = [{"symbol": "EURUSD", "weight": 0.1, "sl": 0.0, "tp": 0.0}]
        
        KILL_SWITCH.activate()
        
        with pytest.raises(RuntimeError, match="KILL SWITCH"):
            mt5_bridge.apply_allocation_to_mt5(
                allocation,
                login=12345,
                password="test",
                server="Demo"
            )
