"""Tests for centralized configuration management."""
import pytest
from pathlib import Path
from pydantic import ValidationError

from openquant.config import ConfigManager, get_config, set_global_config, reset_global_config


def test_load_default_config():
    """Test loading default configuration."""
    config = ConfigManager.from_yaml("configs/default.yaml")
    
    assert config.get("risk_limits.dd_limit") == 0.20
    assert config.get("risk_limits.daily_loss_cap") == 0.05
    assert config.get("circuit_breaker.daily_loss_limit") == 0.02


def test_get_section():
    """Test getting configuration section."""
    config = ConfigManager.from_yaml("configs/default.yaml")
    
    risk_config = config.get_section("risk_limits")
    assert risk_config.dd_limit == 0.20
    assert risk_config.daily_loss_cap == 0.05
    assert risk_config.cvar_limit == 0.08


def test_set_value():
    """Test setting configuration value."""
    config = ConfigManager()
    
    config.set("risk_limits.dd_limit", 0.15)
    assert config.get("risk_limits.dd_limit") == 0.15


def test_from_dict():
    """Test creating config from dictionary."""
    config_dict = {
        "risk_limits": {
            "dd_limit": 0.10,
            "daily_loss_cap": 0.02,
            "cvar_limit": 0.05,
            "max_exposure_per_symbol": 0.15
        }
    }
    
    config = ConfigManager.from_dict(config_dict)
    assert config.get("risk_limits.dd_limit") == 0.10


def test_validation_error():
    """Test that invalid config raises ValidationError."""
    invalid_config = {
        "risk_limits": {
            "dd_limit": 1.5,  # Invalid: must be <= 1.0
            "daily_loss_cap": 0.05,
            "cvar_limit": 0.08,
            "max_exposure_per_symbol": 0.20
        }
    }
    
    with pytest.raises(ValidationError):
        ConfigManager.from_dict(invalid_config)


def test_merge_configs():
    """Test merging configurations."""
    config = ConfigManager()
    
    override = {
        "risk_limits": {
            "dd_limit": 0.18,
            "daily_loss_cap": 0.04
        }
    }
    
    config.merge(override)
    assert config.get("risk_limits.dd_limit") == 0.18
    assert config.get("risk_limits.daily_loss_cap") == 0.04


def test_to_dict():
    """Test exporting config to dictionary."""
    config = ConfigManager()
    config_dict = config.to_dict()
    
    assert isinstance(config_dict, dict)
    assert "risk_limits" in config_dict
    assert "circuit_breaker" in config_dict


def test_global_config():
    """Test global configuration management."""
    reset_global_config()
    
    config1 = get_config()
    config2 = get_config()
    
    # Should return same instance
    assert config1 is config2
    
    # Clean up
    reset_global_config()


def test_set_global_config():
    """Test setting global configuration."""
    reset_global_config()
    
    custom_config = ConfigManager.from_dict({
        "risk_limits": {
            "dd_limit": 0.12,
            "daily_loss_cap": 0.03,
            "cvar_limit": 0.06,
            "max_exposure_per_symbol": 0.18
        }
    })
    
    set_global_config(custom_config)
    
    config = get_config()
    assert config.get("risk_limits.dd_limit") == 0.12
    
    # Clean up
    reset_global_config()


def test_forex_config():
    """Test forex configuration access."""
    config = ConfigManager.from_yaml("configs/default.yaml")
    forex_config = config.get_section("forex")
    
    assert "EURUSD" in forex_config.symbols
    eurusd = forex_config.symbols["EURUSD"]
    assert eurusd.spread_bps == 0.5
    assert eurusd.pip_value == 0.0001
    
    assert "london" in forex_config.sessions
    london = forex_config.sessions["london"]
    assert london.start == 8
    assert london.end == 17


def test_backtest_config():
    """Test backtest configuration."""
    config = ConfigManager.from_yaml("configs/default.yaml")
    bt_config = config.get_section("backtest")
    
    assert bt_config.fee_bps == 1.0
    assert bt_config.leverage == 1.0
    assert bt_config.weight == 1.0


def test_adaptive_sizing_config():
    """Test adaptive sizing configuration."""
    config = ConfigManager.from_yaml("configs/default.yaml")
    sizing_config = config.get_section("adaptive_sizing")
    
    assert sizing_config.method == "volatility"
    assert sizing_config.target_risk == 0.01
    assert sizing_config.max_drawdown == 0.50
    assert sizing_config.aggressive_mode is False


def test_stationarity_config():
    """Test stationarity configuration."""
    config = ConfigManager.from_yaml("configs/default.yaml")
    stat_config = config.get_section("stationarity")
    
    assert stat_config.adf_threshold == 0.05
    assert stat_config.hurst_mean_reverting == 0.45
    assert stat_config.hurst_trending == 0.55


def test_production_config():
    """Test production configuration has conservative values."""
    config = ConfigManager.from_yaml("configs/production.yaml")
    
    # Production should have lower risk limits
    assert config.get("risk_limits.dd_limit") <= 0.20
    assert config.get("risk_limits.daily_loss_cap") <= 0.05
    assert config.get("circuit_breaker.daily_loss_limit") <= 0.02


def test_aggressive_config():
    """Test aggressive configuration has higher risk values."""
    config = ConfigManager.from_yaml("configs/aggressive.yaml")
    
    # Aggressive should have higher risk tolerance
    assert config.get("risk_limits.dd_limit") >= 0.20
    assert config.get("backtest.leverage") >= 1.0
    
    sizing = config.get_section("adaptive_sizing")
    assert sizing.aggressive_mode is True
