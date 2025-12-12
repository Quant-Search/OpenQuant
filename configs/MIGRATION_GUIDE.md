# Configuration Migration Guide

This guide explains how to migrate from hardcoded constants to the centralized configuration system.

## Overview

The new configuration system provides:
- **Centralized Management**: All configuration in one place
- **Schema Validation**: Pydantic-based validation prevents invalid values
- **YAML-based**: Human-readable configuration files
- **Type Safety**: Strong typing with validation
- **Environment Support**: Different configs for dev/prod/testing

## Migration Steps

### 1. Update Imports

**Old:**
```python
from openquant.config.forex import FOREX_CONFIG, get_spread_bps
```

**New:**
```python
from openquant.config import get_config

config = get_config()
spread = config.get("forex.symbols.EURUSD.spread_bps")
```

### 2. Replace Hardcoded Constants

**Old:**
```python
def __init__(self):
    self.dd_limit = 0.20
    self.daily_loss_cap = 0.05
```

**New:**
```python
def __init__(self, config=None):
    if config is None:
        from openquant.config.manager import get_config
        config = get_config()
    
    risk_limits = config.get_section("risk_limits")
    self.dd_limit = risk_limits.dd_limit
    self.daily_loss_cap = risk_limits.daily_loss_cap
```

### 3. Make Parameters Optional with Config Fallback

**Old:**
```python
def backtest_signals(df, signals, fee_bps=1.0, leverage=1.0):
    ...
```

**New:**
```python
def backtest_signals(df, signals, fee_bps=None, leverage=None, config=None):
    if config is None:
        from openquant.config.manager import get_config
        config = get_config()
    
    bt_config = config.get_section("backtest")
    fee_bps = fee_bps if fee_bps is not None else bt_config.fee_bps
    leverage = leverage if leverage is not None else bt_config.leverage
    ...
```

## Module-Specific Changes

### Forex Configuration

**Old:**
```python
from openquant.config.forex import get_spread_bps, get_swap_cost, is_optimal_session

spread = get_spread_bps("EURUSD")
swap = get_swap_cost("EURUSD", "long")
optimal = is_optimal_session("EURUSD", 14)
```

**New:**
```python
from openquant.config import get_config

config = get_config()
forex_config = config.get_section("forex")

# Direct access
spread = forex_config.symbols["EURUSD"].spread_bps
swap = forex_config.symbols["EURUSD"].swap_long

# Or using dot notation
spread = config.get("forex.symbols.EURUSD.spread_bps")

# Session check
def is_optimal_session(symbol: str, hour_utc: int) -> bool:
    symbol_config = forex_config.symbols[symbol]
    for session_name in symbol_config.optimal_sessions:
        session = forex_config.sessions[session_name]
        if session.start <= hour_utc < session.end:
            return True
    return False
```

### Risk Management

**Old:**
```python
from openquant.risk.portfolio_guard import PortfolioGuard

guard = PortfolioGuard()
# Uses hardcoded limits: dd_limit=0.20, daily_loss_cap=0.05, etc.
```

**New:**
```python
from openquant.risk.portfolio_guard import PortfolioGuard
from openquant.config import get_config

# Uses config limits
guard = PortfolioGuard()

# Or with custom config
config = get_config()
config.set("risk_limits.dd_limit", 0.15)
guard = PortfolioGuard(config=config)
```

### Circuit Breaker

**Old:**
```python
from openquant.risk.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(daily_loss_limit=0.02, drawdown_limit=0.10)
```

**New:**
```python
from openquant.risk.circuit_breaker import CircuitBreaker

# Uses config values by default
breaker = CircuitBreaker()

# Or override specific values
breaker = CircuitBreaker(daily_loss_limit=0.015)
```

### Backtest Engine

**Old:**
```python
from openquant.backtest.engine import backtest_signals

result = backtest_signals(df, signals, fee_bps=1.0, leverage=1.0)
```

**New:**
```python
from openquant.backtest.engine import backtest_signals

# Uses config values by default
result = backtest_signals(df, signals)

# Or override specific values
result = backtest_signals(df, signals, fee_bps=2.0)
```

### Strategy Mixer

**Old:**
```python
# Hardcoded threshold = 0.2
mixer = StrategyMixer(strategies, weights)
signals = mixer.generate_signals(df)
```

**New:**
```python
# Uses config value (strategy_mixer.threshold)
mixer = StrategyMixer(strategies, weights)
signals = mixer.generate_signals(df)
```

### Adaptive Sizing

**Old:**
```python
from openquant.risk.adaptive_sizing import AdaptiveSizer

sizer = AdaptiveSizer(method="volatility", target_risk=0.01, max_drawdown=0.50)
```

**New:**
```python
from openquant.risk.adaptive_sizing import AdaptiveSizer

# Uses config values by default
sizer = AdaptiveSizer()

# Or override specific values
sizer = AdaptiveSizer(method="kelly")
```

### Stationarity Testing

**Old:**
```python
from openquant.quant.stationarity import classify_regime

regime = classify_regime(series)
# Uses hardcoded thresholds: h < 0.45 for mean-reverting, etc.
```

**New:**
```python
from openquant.quant.stationarity import classify_regime

# Uses config thresholds
regime = classify_regime(series)
```

## Using Different Configurations

### Development
```bash
export OPENQUANT_CONFIG=configs/default.yaml
python scripts/run_robot_cli.py
```

### Production
```bash
export OPENQUANT_CONFIG=configs/production.yaml
python scripts/run_robot_cli.py
```

### Aggressive Trading
```bash
export OPENQUANT_CONFIG=configs/aggressive.yaml
python scripts/run_robot_cli.py
```

### Custom Configuration
```python
from openquant.config import ConfigManager, set_global_config

# Load base config
config = ConfigManager.from_yaml("configs/production.yaml")

# Customize
config.set("risk_limits.dd_limit", 0.12)
config.set("backtest.leverage", 1.5)

# Set as global
set_global_config(config)

# Now all modules will use this config
from openquant.backtest.engine import backtest_signals
result = backtest_signals(df, signals)  # Uses dd_limit=0.12, leverage=1.5
```

## Backward Compatibility

All migrated modules maintain backward compatibility:
- Functions accept explicit parameters that override config values
- Old function signatures still work
- `openquant.config.forex` module is deprecated but still functional

## Validation

All configuration values are validated by Pydantic schemas:

```python
from openquant.config import ConfigManager

try:
    config = ConfigManager.from_yaml("configs/custom.yaml")
except ValidationError as e:
    print(f"Invalid configuration: {e}")
```

Common validation rules:
- `dd_limit`, `daily_loss_cap`, etc. must be between 0 and 1
- `leverage` must be >= 1
- `fee_bps`, `slippage_bps` must be >= 0
- Session `start` and `end` hours must be between 0 and 24
- `threshold` must be between 0 and 1

## Best Practices

1. **Use Default Config**: Let modules load from global config
2. **Override Sparingly**: Only override when testing specific scenarios
3. **Validate Early**: Load and validate config at application startup
4. **Document Changes**: When creating custom configs, document why values differ from defaults
5. **Environment Variables**: Use `OPENQUANT_CONFIG` env var for environment-specific configs
6. **Version Control**: Keep configs in version control, but exclude sensitive credentials

## Testing

When writing tests, use isolated configs:

```python
from openquant.config import ConfigManager, set_global_config, reset_global_config

def test_my_feature():
    # Create test config
    test_config = ConfigManager.from_dict({
        "risk_limits": {
            "dd_limit": 0.10,
            "daily_loss_cap": 0.02,
            "cvar_limit": 0.05,
            "max_exposure_per_symbol": 0.15
        },
        # ... other sections
    })
    
    # Set as global for this test
    set_global_config(test_config)
    
    try:
        # Run test
        ...
    finally:
        # Clean up
        reset_global_config()
```
