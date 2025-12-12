"""Example script demonstrating centralized configuration usage."""
from openquant.config import ConfigManager, get_config, set_global_config
from openquant.risk.portfolio_guard import PortfolioGuard
from openquant.risk.circuit_breaker import CircuitBreaker
from openquant.risk.adaptive_sizing import AdaptiveSizer


def example_load_config():
    """Example 1: Loading configuration from YAML."""
    print("=" * 60)
    print("Example 1: Loading Configuration from YAML")
    print("=" * 60)
    
    # Load default configuration
    config = ConfigManager.from_yaml("configs/default.yaml")
    
    # Access values using dot notation
    dd_limit = config.get("risk_limits.dd_limit")
    print(f"Max Drawdown Limit: {dd_limit:.2%}")
    
    # Access entire sections
    risk_config = config.get_section("risk_limits")
    print(f"Daily Loss Cap: {risk_config.daily_loss_cap:.2%}")
    print(f"CVaR Limit: {risk_config.cvar_limit:.2%}")
    print(f"Max Exposure per Symbol: {risk_config.max_exposure_per_symbol:.2%}")
    print()


def example_modify_config():
    """Example 2: Modifying configuration values."""
    print("=" * 60)
    print("Example 2: Modifying Configuration")
    print("=" * 60)
    
    config = ConfigManager.from_yaml("configs/default.yaml")
    
    # Before modification
    print(f"Before: dd_limit = {config.get('risk_limits.dd_limit'):.2%}")
    
    # Modify value
    config.set("risk_limits.dd_limit", 0.15)
    
    # After modification
    print(f"After: dd_limit = {config.get('risk_limits.dd_limit'):.2%}")
    
    # Save to new file
    config.to_yaml("configs/custom.yaml")
    print("Modified config saved to configs/custom.yaml")
    print()


def example_global_config():
    """Example 3: Using global configuration."""
    print("=" * 60)
    print("Example 3: Using Global Configuration")
    print("=" * 60)
    
    # Get global config (loads from OPENQUANT_CONFIG env var or default)
    config = get_config()
    
    # All modules will use this config automatically
    dd_limit = config.get("risk_limits.dd_limit")
    print(f"Global config dd_limit: {dd_limit:.2%}")
    
    # Modules automatically use global config
    guard = PortfolioGuard()
    print(f"PortfolioGuard dd_limit: {guard.limits['dd_limit']:.2%}")
    
    breaker = CircuitBreaker()
    print(f"CircuitBreaker drawdown_limit: {breaker.drawdown_limit:.2%}")
    print()


def example_custom_config():
    """Example 4: Using custom configuration per module."""
    print("=" * 60)
    print("Example 4: Using Custom Config per Module")
    print("=" * 60)
    
    # Create custom config
    custom_config = ConfigManager.from_dict({
        "risk_limits": {
            "dd_limit": 0.10,
            "daily_loss_cap": 0.02,
            "cvar_limit": 0.05,
            "max_exposure_per_symbol": 0.15
        },
        "circuit_breaker": {
            "daily_loss_limit": 0.015,
            "drawdown_limit": 0.08,
            "volatility_limit": 0.04
        },
        "adaptive_sizing": {
            "method": "kelly",
            "target_risk": 0.01,
            "max_drawdown": 0.30,
            "aggressive_mode": False,
            "target_volatility": 0.15,
            "max_leverage": 1.0
        }
    })
    
    # Pass config to modules
    guard = PortfolioGuard(config=custom_config)
    breaker = CircuitBreaker(config=custom_config)
    sizer = AdaptiveSizer(config=custom_config)
    
    print(f"Custom PortfolioGuard dd_limit: {guard.limits['dd_limit']:.2%}")
    print(f"Custom CircuitBreaker drawdown_limit: {breaker.drawdown_limit:.2%}")
    print(f"Custom AdaptiveSizer method: {sizer.method}")
    print()


def example_merge_configs():
    """Example 5: Merging configurations."""
    print("=" * 60)
    print("Example 5: Merging Configurations")
    print("=" * 60)
    
    # Load base config
    config = ConfigManager.from_yaml("configs/default.yaml")
    print(f"Base dd_limit: {config.get('risk_limits.dd_limit'):.2%}")
    
    # Create override config
    override = {
        "risk_limits": {
            "dd_limit": 0.18,
            "daily_loss_cap": 0.04
        }
    }
    
    # Merge
    config.merge(override)
    print(f"After merge dd_limit: {config.get('risk_limits.dd_limit'):.2%}")
    print(f"After merge daily_loss_cap: {config.get('risk_limits.daily_loss_cap'):.2%}")
    print()


def example_forex_config():
    """Example 6: Accessing forex configuration."""
    print("=" * 60)
    print("Example 6: Accessing Forex Configuration")
    print("=" * 60)
    
    config = ConfigManager.from_yaml("configs/default.yaml")
    forex_config = config.get_section("forex")
    
    # Get symbol configuration
    eurusd = forex_config.symbols["EURUSD"]
    print(f"EURUSD Spread: {eurusd.spread_bps} bps")
    print(f"EURUSD Swap Long: {eurusd.swap_long} pips")
    print(f"EURUSD Swap Short: {eurusd.swap_short} pips")
    print(f"EURUSD Optimal Sessions: {', '.join(eurusd.optimal_sessions)}")
    
    # Get session configuration
    london = forex_config.sessions["london"]
    print(f"\nLondon Session: {london.start}:00 - {london.end}:00 UTC")
    print()


def example_validation():
    """Example 7: Configuration validation."""
    print("=" * 60)
    print("Example 7: Configuration Validation")
    print("=" * 60)
    
    try:
        # Try to create invalid config (dd_limit > 1.0)
        invalid_config = ConfigManager.from_dict({
            "risk_limits": {
                "dd_limit": 1.5,  # Invalid: must be <= 1.0
                "daily_loss_cap": 0.05,
                "cvar_limit": 0.08,
                "max_exposure_per_symbol": 0.20
            }
        })
    except Exception as e:
        print(f"Validation error (expected): {type(e).__name__}")
        print(f"Invalid dd_limit=1.5 rejected (must be <= 1.0)")
    
    # Valid config
    valid_config = ConfigManager.from_dict({
        "risk_limits": {
            "dd_limit": 0.25,  # Valid
            "daily_loss_cap": 0.05,
            "cvar_limit": 0.08,
            "max_exposure_per_symbol": 0.20
        }
    })
    print(f"Valid dd_limit=0.25 accepted: {valid_config.get('risk_limits.dd_limit'):.2%}")
    print()


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("OpenQuant Configuration System Examples")
    print("=" * 60 + "\n")
    
    example_load_config()
    example_modify_config()
    example_global_config()
    example_custom_config()
    example_merge_configs()
    example_forex_config()
    example_validation()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
