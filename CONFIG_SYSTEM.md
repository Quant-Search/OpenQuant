# OpenQuant Configuration System

## Overview

A centralized configuration management system with Pydantic schema validation for OpenQuant trading platform.

## Features

- ✅ **Centralized Management**: All configuration in YAML files in `configs/` directory
- ✅ **Schema Validation**: Pydantic-based type checking prevents invalid configurations
- ✅ **Type Safety**: Strong typing with IDE autocomplete support
- ✅ **Flexibility**: Override defaults, merge configs, per-module configurations
- ✅ **Environment Support**: Different configs for development, production, and aggressive trading
- ✅ **Backward Compatible**: Existing code continues to work with deprecation warnings

## Architecture

### Core Components

1. **openquant/config/schemas.py** - Pydantic models defining configuration structure
2. **openquant/config/manager.py** - ConfigManager class for loading, accessing, and saving configs
3. **openquant/config/forex.py** - Backward compatibility wrapper (deprecated)
4. **configs/** - YAML configuration files (default, production, aggressive)

### Configuration Sections

| Section | Purpose |
|---------|---------|
| `forex` | Forex symbol spreads, swaps, pip values, trading sessions |
| `risk_limits` | Portfolio risk management limits (drawdown, daily loss, CVaR) |
| `circuit_breaker` | Automatic trading halt thresholds |
| `backtest` | Backtest engine parameters (fees, leverage, slippage) |
| `strategy_mixer` | Strategy ensemble settings |
| `adaptive_sizing` | Position sizing configuration (Kelly, volatility targeting) |
| `stationarity` | Statistical test thresholds (ADF, KPSS, Hurst) |
| `concentration_limits` | Portfolio concentration constraints |
| `paper_trading` | Paper trading simulator settings |

## File Structure

```
openquant/
├── config/
│   ├── __init__.py          # Exports ConfigManager, get_config, etc.
│   ├── manager.py           # ConfigManager implementation
│   ├── schemas.py           # Pydantic configuration schemas
│   ├── forex.py             # Backward compatibility (deprecated)
│   └── README.md            # Configuration system documentation
├── risk/
│   ├── portfolio_guard.py   # ✅ Migrated to use ConfigManager
│   ├── circuit_breaker.py   # ✅ Migrated to use ConfigManager
│   └── adaptive_sizing.py   # ✅ Migrated to use ConfigManager
├── backtest/
│   └── engine.py            # ✅ Migrated to use ConfigManager
├── strategies/
│   └── mixer.py             # ✅ Migrated to use ConfigManager
├── quant/
│   └── stationarity.py      # ✅ Migrated to use ConfigManager
└── paper/
    └── simulator.py         # ✅ Migrated to use ConfigManager

configs/
├── README.md                # Configuration file usage guide
├── MIGRATION_GUIDE.md       # Migration from hardcoded constants
├── default.yaml             # Default balanced configuration
├── production.yaml          # Conservative production configuration
└── aggressive.yaml          # Higher risk aggressive configuration

scripts/
└── example_config_usage.py  # Example script demonstrating usage

tests/
└── test_config_manager.py   # Configuration system tests
```

## Quick Start

### 1. Basic Usage

```python
from openquant.config import get_config

# Get global configuration
config = get_config()

# Access values using dot notation
dd_limit = config.get("risk_limits.dd_limit")
print(f"Max drawdown limit: {dd_limit}")  # 0.20

# Get entire section
risk_config = config.get_section("risk_limits")
print(f"Daily loss cap: {risk_config.daily_loss_cap}")  # 0.05
```

### 2. Load Specific Configuration

```python
from openquant.config import ConfigManager

# Load production config
config = ConfigManager.from_yaml("configs/production.yaml")

# Modules automatically use this config
from openquant.risk.portfolio_guard import PortfolioGuard
guard = PortfolioGuard(config=config)
```

### 3. Environment Variable

```bash
# Set environment variable
export OPENQUANT_CONFIG=configs/production.yaml

# Run application - will automatically load production config
python scripts/run_robot_cli.py
```

### 4. Modify and Save

```python
from openquant.config import ConfigManager

config = ConfigManager.from_yaml("configs/default.yaml")

# Modify values
config.set("risk_limits.dd_limit", 0.15)
config.set("backtest.leverage", 2.0)

# Save to new file
config.to_yaml("configs/custom.yaml")
```

## Configuration Files

### default.yaml

Balanced settings suitable for most scenarios:
- Max Drawdown: 20%
- Daily Loss Cap: 5%
- Leverage: 1.0x
- Fee: 1.0 bps

### production.yaml

Conservative settings for live trading:
- Max Drawdown: 15%
- Daily Loss Cap: 3%
- Leverage: 1.0x
- Fee: 2.0 bps (realistic)
- Slippage: 1.0 bps
- Concentration limits enforced

### aggressive.yaml

Higher risk settings:
- Max Drawdown: 30%
- Daily Loss Cap: 8%
- Leverage: 2.0x
- Aggressive sizing mode enabled
- Higher target volatility (35%)

## Migrated Modules

The following modules have been migrated to use centralized configuration:

### Risk Management

- **PortfolioGuard** (`openquant/risk/portfolio_guard.py`)
  - Now loads risk limits from config
  - `dd_limit`, `daily_loss_cap`, `cvar_limit`, `max_exposure_per_symbol`

- **CircuitBreaker** (`openquant/risk/circuit_breaker.py`)
  - Loads thresholds from config
  - `daily_loss_limit`, `drawdown_limit`, `volatility_limit`

- **AdaptiveSizer** (`openquant/risk/adaptive_sizing.py`)
  - Loads sizing parameters from config
  - `method`, `target_risk`, `max_drawdown`, `aggressive_mode`, `target_volatility`, `max_leverage`

### Backtest Engine

- **backtest_signals** (`openquant/backtest/engine.py`)
  - Loads default parameters from config
  - `fee_bps`, `slippage_bps`, `weight`, `spread_bps`, `leverage`, `impact_coeff`

### Strategy Components

- **StrategyMixer** (`openquant/strategies/mixer.py`)
  - Loads signal threshold from config
  - `threshold` for long/short decisions

### Statistical Tools

- **classify_regime** (`openquant/quant/stationarity.py`)
  - Loads thresholds from config
  - `hurst_mean_reverting`, `hurst_trending`, `hurst_high_confidence`

### Paper Trading

- **check_daily_loss** (`openquant/paper/simulator.py`)
  - Loads loss limit from config
  - `daily_loss_limit`

### Forex Configuration

- **forex.py** (`openquant/config/forex.py`)
  - Deprecated wrapper for backward compatibility
  - Loads from ConfigManager internally
  - Issues deprecation warnings

## Usage Examples

### Example 1: Using Global Config

```python
from openquant.config import get_config
from openquant.risk.portfolio_guard import PortfolioGuard
from openquant.risk.circuit_breaker import CircuitBreaker

# All modules use global config automatically
config = get_config()
guard = PortfolioGuard()
breaker = CircuitBreaker()

print(f"DD Limit: {guard.limits['dd_limit']}")  # From config
print(f"Circuit Breaker Limit: {breaker.drawdown_limit}")  # From config
```

### Example 2: Custom Configuration

```python
from openquant.config import ConfigManager
from openquant.backtest.engine import backtest_signals

# Create custom config
config = ConfigManager.from_dict({
    "backtest": {
        "fee_bps": 2.5,
        "slippage_bps": 1.0,
        "leverage": 1.5,
        "weight": 0.8,
        "spread_bps": 0.5,
        "impact_coeff": 0.001
    }
})

# Pass to backtest
result = backtest_signals(df, signals, config=config)
```

### Example 3: Override Specific Values

```python
from openquant.backtest.engine import backtest_signals

# Uses config for most params, overrides fee_bps
result = backtest_signals(df, signals, fee_bps=3.0)
```

### Example 4: Forex Configuration

```python
from openquant.config import get_config

config = get_config()
forex_config = config.get_section("forex")

# Get EURUSD configuration
eurusd = forex_config.symbols["EURUSD"]
print(f"Spread: {eurusd.spread_bps} bps")
print(f"Swap Long: {eurusd.swap_long} pips")
print(f"Swap Short: {eurusd.swap_short} pips")
print(f"Optimal Sessions: {eurusd.optimal_sessions}")

# Check if in optimal session
london = forex_config.sessions["london"]
is_london_hours = london.start <= current_hour < london.end
```

## Validation

All configurations are validated using Pydantic schemas:

### Validation Rules

- **risk_limits.dd_limit**: 0 < value ≤ 1
- **risk_limits.daily_loss_cap**: 0 < value ≤ 1
- **circuit_breaker.daily_loss_limit**: 0 < value ≤ 1
- **backtest.fee_bps**: value ≥ 0
- **backtest.leverage**: value ≥ 1
- **strategy_mixer.threshold**: 0 ≤ value ≤ 1
- **forex.sessions.start**: 0 ≤ value ≤ 23
- **forex.sessions.end**: 0 ≤ value ≤ 24

### Example Validation

```python
from openquant.config import ConfigManager
from pydantic import ValidationError

try:
    config = ConfigManager.from_dict({
        "risk_limits": {
            "dd_limit": 1.5,  # ❌ Invalid: must be ≤ 1.0
            "daily_loss_cap": 0.05,
            "cvar_limit": 0.08,
            "max_exposure_per_symbol": 0.20
        }
    })
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Testing

Run configuration tests:

```bash
pytest tests/test_config_manager.py -v
```

Example test output:
```
tests/test_config_manager.py::test_load_default_config PASSED
tests/test_config_manager.py::test_get_section PASSED
tests/test_config_manager.py::test_set_value PASSED
tests/test_config_manager.py::test_validation_error PASSED
tests/test_config_manager.py::test_forex_config PASSED
```

Run example script:

```bash
python scripts/example_config_usage.py
```

## Migration Guide

See `configs/MIGRATION_GUIDE.md` for detailed instructions on migrating from hardcoded constants.

### Quick Migration Steps

1. **Replace imports**:
   ```python
   # Old
   from openquant.config.forex import FOREX_CONFIG
   
   # New
   from openquant.config import get_config
   config = get_config()
   ```

2. **Replace hardcoded constants**:
   ```python
   # Old
   def __init__(self):
       self.dd_limit = 0.20
   
   # New
   def __init__(self, config=None):
       if config is None:
           from openquant.config.manager import get_config
           config = get_config()
       self.dd_limit = config.get("risk_limits.dd_limit")
   ```

3. **Make parameters optional with config fallback**:
   ```python
   # Old
   def backtest(df, signals, fee_bps=1.0):
       ...
   
   # New
   def backtest(df, signals, fee_bps=None, config=None):
       if config is None:
           config = get_config()
       fee_bps = fee_bps if fee_bps is not None else config.get("backtest.fee_bps")
   ```

## Benefits

1. **Centralization**: All configuration in one place instead of scattered across modules
2. **Validation**: Invalid configurations caught early with clear error messages
3. **Type Safety**: IDE autocomplete and type checking for configuration values
4. **Flexibility**: Easy to switch between different configurations (dev/prod/aggressive)
5. **Testing**: Easy to create isolated test configurations
6. **Documentation**: YAML files are self-documenting
7. **Version Control**: Configuration changes tracked in git
8. **Environment Support**: Different configs for different environments

## Best Practices

1. ✅ Use global config for production code
2. ✅ Pass explicit config for testing
3. ✅ Set `OPENQUANT_CONFIG` environment variable for environment-specific configs
4. ✅ Validate configuration at application startup
5. ✅ Document custom configurations
6. ✅ Keep configs in version control
7. ✅ Use production config for live trading
8. ✅ Use default config for development

## Future Enhancements

Potential future improvements:

- [ ] Hot-reload configuration without restarting
- [ ] Configuration versioning and migration scripts
- [ ] Web UI for configuration management
- [ ] Configuration templates for different trading styles
- [ ] Configuration diff tool
- [ ] Remote configuration storage (S3, database)
- [ ] Encrypted configuration for sensitive values
- [ ] Configuration audit logging

## Support

For questions or issues:

1. Check `openquant/config/README.md` for API documentation
2. See `configs/README.md` for configuration file usage
3. Review `configs/MIGRATION_GUIDE.md` for migration help
4. Run `python scripts/example_config_usage.py` for working examples
5. Run tests: `pytest tests/test_config_manager.py`

## Summary

The centralized configuration system provides a robust, type-safe, and flexible way to manage OpenQuant configuration. All major modules have been migrated to use the system while maintaining backward compatibility. The system is production-ready and includes comprehensive documentation, examples, and tests.
