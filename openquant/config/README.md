# OpenQuant Configuration System

Centralized configuration management with Pydantic schema validation.

## Overview

The configuration system provides:

- **Centralized Management**: All configuration in YAML files
- **Schema Validation**: Pydantic-based type checking and validation
- **Type Safety**: Strong typing with IDE support
- **Flexibility**: Override defaults, merge configs, per-module configs
- **Environment Support**: Different configs for dev/prod/aggressive trading

## Quick Start

### Basic Usage

```python
from openquant.config import get_config

# Get global configuration
config = get_config()

# Access values using dot notation
dd_limit = config.get("risk_limits.dd_limit")
print(f"Max drawdown limit: {dd_limit}")

# Get entire section
risk_config = config.get_section("risk_limits")
print(f"Daily loss cap: {risk_config.daily_loss_cap}")
```

### Loading from File

```python
from openquant.config import ConfigManager

# Load specific config file
config = ConfigManager.from_yaml("configs/production.yaml")

# Access configuration
spread = config.get("forex.symbols.EURUSD.spread_bps")
```

### Environment Variable

Set `OPENQUANT_CONFIG` to specify which config to load:

```bash
export OPENQUANT_CONFIG=configs/production.yaml
python scripts/run_robot_cli.py
```

## Architecture

### Components

1. **schemas.py** - Pydantic models defining configuration structure
2. **manager.py** - ConfigManager class for loading, accessing, and saving configs
3. **forex.py** - Backward compatibility wrapper (deprecated)

### Configuration Sections

- **forex** - Forex symbol spreads, swaps, sessions
- **risk_limits** - Portfolio risk management limits
- **circuit_breaker** - Automatic trading halt thresholds
- **backtest** - Backtest engine parameters
- **strategy_mixer** - Strategy ensemble settings
- **adaptive_sizing** - Position sizing configuration
- **stationarity** - Statistical test thresholds
- **concentration_limits** - Portfolio concentration constraints
- **paper_trading** - Paper trading simulator settings

## Configuration Files

Located in `configs/` directory:

- **default.yaml** - Default balanced settings
- **production.yaml** - Conservative settings for live trading
- **aggressive.yaml** - Higher risk settings

## API Reference

### ConfigManager

```python
class ConfigManager:
    def __init__(self, config: Optional[Config] = None)
    
    @classmethod
    def from_yaml(cls, path: str | Path) -> ConfigManager
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> ConfigManager
    
    def get(self, key: str, default: Any = None) -> Any
    
    def get_section(self, section: str) -> Optional[Any]
    
    def set(self, key: str, value: Any) -> None
    
    def to_dict(self) -> Dict[str, Any]
    
    def to_yaml(self, path: str | Path) -> None
    
    def validate(self) -> bool
    
    def reload(self) -> None
    
    def merge(self, other: Dict[str, Any] | ConfigManager) -> None
```

### Global Functions

```python
def get_config() -> ConfigManager
    """Get or create global configuration instance."""

def set_global_config(config: ConfigManager) -> None
    """Set global configuration instance."""

def reset_global_config() -> None
    """Reset global configuration."""
```

## Examples

### Example 1: Load and Access

```python
from openquant.config import ConfigManager

config = ConfigManager.from_yaml("configs/default.yaml")

# Dot notation access
dd_limit = config.get("risk_limits.dd_limit")

# Section access
risk_config = config.get_section("risk_limits")
print(f"DD Limit: {risk_config.dd_limit}")
print(f"Daily Loss Cap: {risk_config.daily_loss_cap}")
```

### Example 2: Modify and Save

```python
from openquant.config import ConfigManager

config = ConfigManager.from_yaml("configs/default.yaml")

# Modify values
config.set("risk_limits.dd_limit", 0.15)
config.set("backtest.leverage", 2.0)

# Save to new file
config.to_yaml("configs/custom.yaml")
```

### Example 3: Use with Modules

```python
from openquant.config import ConfigManager
from openquant.risk.portfolio_guard import PortfolioGuard
from openquant.backtest.engine import backtest_signals

# Load config
config = ConfigManager.from_yaml("configs/production.yaml")

# Pass to modules
guard = PortfolioGuard(config=config)
result = backtest_signals(df, signals, config=config)

# Modules use config values automatically
print(f"DD Limit: {guard.limits['dd_limit']}")
```

### Example 4: Merge Configurations

```python
from openquant.config import ConfigManager

# Load base config
config = ConfigManager.from_yaml("configs/default.yaml")

# Create override
override = {
    "risk_limits": {
        "dd_limit": 0.18,
        "daily_loss_cap": 0.04
    }
}

# Merge
config.merge(override)
```

### Example 5: Forex Configuration

```python
from openquant.config import get_config

config = get_config()
forex_config = config.get_section("forex")

# Get symbol info
eurusd = forex_config.symbols["EURUSD"]
print(f"Spread: {eurusd.spread_bps} bps")
print(f"Swap Long: {eurusd.swap_long} pips")

# Check session
london = forex_config.sessions["london"]
print(f"London: {london.start}:00-{london.end}:00 UTC")
```

## Module Integration

### Risk Management

```python
from openquant.risk.portfolio_guard import PortfolioGuard

# Uses global config automatically
guard = PortfolioGuard()

# Or pass custom config
from openquant.config import ConfigManager
config = ConfigManager.from_yaml("configs/production.yaml")
guard = PortfolioGuard(config=config)
```

### Circuit Breaker

```python
from openquant.risk.circuit_breaker import CircuitBreaker

# Uses global config
breaker = CircuitBreaker()

# Or override specific values
breaker = CircuitBreaker(daily_loss_limit=0.015)
```

### Backtest Engine

```python
from openquant.backtest.engine import backtest_signals

# Uses global config for defaults
result = backtest_signals(df, signals)

# Override specific parameters
result = backtest_signals(df, signals, fee_bps=2.0, leverage=1.5)
```

### Adaptive Sizing

```python
from openquant.risk.adaptive_sizing import AdaptiveSizer

# Uses global config
sizer = AdaptiveSizer()

# Override method
sizer = AdaptiveSizer(method="kelly")
```

## Validation

All configurations are validated using Pydantic schemas:

```python
from openquant.config import ConfigManager
from pydantic import ValidationError

try:
    config = ConfigManager.from_dict({
        "risk_limits": {
            "dd_limit": 1.5,  # Invalid: must be <= 1.0
            "daily_loss_cap": 0.05,
            "cvar_limit": 0.08,
            "max_exposure_per_symbol": 0.20
        }
    })
except ValidationError as e:
    print(f"Validation error: {e}")
```

### Validation Rules

- **risk_limits.dd_limit**: 0 < value <= 1
- **risk_limits.daily_loss_cap**: 0 < value <= 1
- **risk_limits.cvar_limit**: 0 < value <= 1
- **circuit_breaker.daily_loss_limit**: 0 < value <= 1
- **backtest.fee_bps**: value >= 0
- **backtest.leverage**: value >= 1
- **strategy_mixer.threshold**: 0 <= value <= 1
- **forex.sessions.start**: 0 <= value <= 23
- **forex.sessions.end**: 0 <= value <= 24

## Testing

Use isolated configs in tests:

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
        }
    })
    
    set_global_config(test_config)
    
    try:
        # Run test
        from openquant.risk.portfolio_guard import PortfolioGuard
        guard = PortfolioGuard()
        assert guard.limits["dd_limit"] == 0.10
    finally:
        reset_global_config()
```

## Migration

See `configs/MIGRATION_GUIDE.md` for detailed migration instructions from hardcoded constants.

## Best Practices

1. **Use Global Config**: Let modules load from global config by default
2. **Override Sparingly**: Only override when testing specific scenarios
3. **Validate Early**: Load and validate config at application startup
4. **Document Changes**: When creating custom configs, document why values differ
5. **Environment Variables**: Use `OPENQUANT_CONFIG` for environment-specific configs
6. **Version Control**: Keep configs in version control (except sensitive data)

## Troubleshooting

### Config Not Found

```python
FileNotFoundError: Configuration file not found: configs/custom.yaml
```

**Solution**: Check file path is correct relative to project root.

### Validation Error

```python
ValidationError: 1 validation error for Config
risk_limits.dd_limit
  ensure this value is less than or equal to 1
```

**Solution**: Fix invalid value in YAML file or dict.

### Global Config Not Loading

**Problem**: Modules use wrong config values.

**Solution**: 
- Check `OPENQUANT_CONFIG` environment variable
- Call `get_config()` to verify what's loaded
- Use `set_global_config()` to set explicitly

## Support

For issues or questions:
1. Check `configs/README.md` for usage examples
2. See `configs/MIGRATION_GUIDE.md` for migration help
3. Run `python scripts/example_config_usage.py` for working examples
