"""
Example: Trade Validator Usage

Demonstrates how to use the comprehensive trade-level risk validator
which integrates asset limits, kill switch, circuit breaker, and correlation checks.
"""
from openquant.risk import TRADE_VALIDATOR, KILL_SWITCH, CIRCUIT_BREAKER, ASSET_LIMITS


def example_basic_validation():
    """Example: Basic trade validation"""
    print("\n=== Basic Trade Validation ===")
    
    # Setup: Current portfolio state
    portfolio_value = 100000.0
    current_positions = {
        "EURUSD": 10000.0,  # $10k notional position
        "GBPUSD": 5000.0,   # $5k notional position
    }
    
    # Validate a new trade
    result = TRADE_VALIDATOR.validate_trade(
        symbol="AUDUSD",
        quantity=1.0,  # 1 lot
        price=70000.0,  # $70k per lot (example)
        side="buy",
        portfolio_value=portfolio_value,
        current_positions=current_positions,
        current_equity=portfolio_value,
    )
    
    if result.allowed:
        print(f"✓ Trade ALLOWED: {result.reason}")
        if result.warnings:
            print(f"  Warnings: {', '.join(result.warnings)}")
    else:
        print(f"✗ Trade REJECTED: {result.reason}")
        print(f"  Failed checks: {', '.join(result.failed_checks)}")


def example_kill_switch():
    """Example: Kill switch blocking trades"""
    print("\n=== Kill Switch Example ===")
    
    # Activate kill switch
    print("Activating kill switch...")
    KILL_SWITCH.activate()
    
    # Try to validate a trade
    result = TRADE_VALIDATOR.validate_trade(
        symbol="BTCUSD",
        quantity=0.1,
        price=50000.0,
        side="buy",
        portfolio_value=100000.0,
        current_positions={},
    )
    
    print(f"Trade allowed: {result.allowed}")
    print(f"Reason: {result.reason}")
    
    # Deactivate kill switch
    KILL_SWITCH.deactivate()
    print("Kill switch deactivated")


def example_circuit_breaker():
    """Example: Circuit breaker triggering on losses"""
    print("\n=== Circuit Breaker Example ===")
    
    # Initialize circuit breaker with tight limits for demo
    from openquant.risk.circuit_breaker import CircuitBreaker
    breaker = CircuitBreaker(
        daily_loss_limit=0.02,  # 2% daily loss limit
        drawdown_limit=0.05,    # 5% max drawdown
    )
    
    # Set initial equity
    start_equity = 100000.0
    breaker.update(start_equity)
    print(f"Starting equity: ${start_equity:,.0f}")
    
    # Simulate a 3% loss
    current_equity = start_equity * 0.97
    breaker.update(current_equity)
    print(f"Current equity after loss: ${current_equity:,.0f}")
    print(f"Circuit breaker tripped: {breaker.is_tripped()}")
    
    # Try to validate a trade
    from openquant.risk.trade_validator import TradeValidator
    validator = TradeValidator(circuit_breaker=breaker)
    
    result = validator.validate_trade(
        symbol="EURUSD",
        quantity=1.0,
        price=10000.0,
        side="buy",
        portfolio_value=current_equity,
        current_positions={},
        current_equity=current_equity,
    )
    
    print(f"Trade allowed: {result.allowed}")
    print(f"Reason: {result.reason}")
    
    # Reset breaker
    breaker.reset()
    print("Circuit breaker reset")


def example_asset_limits():
    """Example: Asset limits enforcement"""
    print("\n=== Asset Limits Example ===")
    
    # Configure asset limits
    ASSET_LIMITS.config.default.max_pct_portfolio = 0.10  # 10% max per asset
    ASSET_LIMITS.config.max_total_positions = 5  # Max 5 positions
    ASSET_LIMITS.config.max_leverage = 1.0  # No leverage
    
    portfolio_value = 100000.0
    current_positions = {
        "EURUSD": 9000.0,  # Already at 9% allocation
    }
    
    # Try to buy more (would exceed 10% limit)
    result = TRADE_VALIDATOR.validate_trade(
        symbol="EURUSD",
        quantity=1.0,
        price=2000.0,  # Would add another $2k (total 11%)
        side="buy",
        portfolio_value=portfolio_value,
        current_positions=current_positions,
    )
    
    print(f"Trade allowed: {result.allowed}")
    print(f"Reason: {result.reason}")
    
    # Try to buy a different asset (should be allowed)
    result = TRADE_VALIDATOR.validate_trade(
        symbol="BTCUSD",
        quantity=0.1,
        price=5000.0,  # $500 (5% of portfolio)
        side="buy",
        portfolio_value=portfolio_value,
        current_positions=current_positions,
    )
    
    print(f"\nDifferent asset trade allowed: {result.allowed}")
    print(f"Reason: {result.reason}")


def example_correlation_check():
    """Example: Correlation constraint warnings"""
    print("\n=== Correlation Check Example ===")
    
    portfolio_value = 100000.0
    current_positions = {
        "EURUSD": 10000.0,  # Hold EUR/USD
    }
    
    # Try to buy GBP/USD (highly correlated with EUR/USD)
    result = TRADE_VALIDATOR.validate_trade(
        symbol="GBPUSD",
        quantity=1.0,
        price=10000.0,
        side="buy",
        portfolio_value=portfolio_value,
        current_positions=current_positions,
    )
    
    print(f"Trade allowed: {result.allowed}")
    print(f"Reason: {result.reason}")
    if result.warnings:
        print(f"Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")


def example_batch_validation():
    """Example: Batch order validation"""
    print("\n=== Batch Order Validation ===")
    
    portfolio_value = 100000.0
    current_positions = {}
    
    # Multiple orders to validate
    orders = [
        ("EURUSD", 1.0, 10000.0, "buy"),
        ("GBPUSD", 1.0, 8000.0, "buy"),
        ("USDJPY", 1.0, 7000.0, "buy"),
        ("AUDUSD", 1.0, 6000.0, "buy"),
        ("NZDUSD", 1.0, 5000.0, "buy"),
        ("USDCAD", 1.0, 4000.0, "buy"),  # This might be rejected if limits reached
    ]
    
    allowed, rejected = TRADE_VALIDATOR.validate_order_batch(
        orders=orders,
        portfolio_value=portfolio_value,
        current_positions=current_positions,
    )
    
    print(f"Allowed orders: {len(allowed)}")
    for symbol, qty, price, side in allowed:
        print(f"  ✓ {symbol}: {qty} @ {price} ({side})")
    
    print(f"\nRejected orders: {len(rejected)}")
    for symbol, reason in rejected:
        print(f"  ✗ {symbol}: {reason}")


def example_validation_status():
    """Example: Get validation status"""
    print("\n=== Validation Status ===")
    
    status = TRADE_VALIDATOR.get_validation_status()
    
    print(f"Kill Switch Active: {status['kill_switch_active']}")
    print(f"Circuit Breaker Tripped: {status['circuit_breaker']['is_tripped']}")
    print(f"Max Total Positions: {status['asset_limits']['max_total_positions']}")
    print(f"Max Leverage: {status['asset_limits']['max_leverage']}x")
    print(f"Correlation Threshold: {status['asset_limits']['correlation_threshold']}")


if __name__ == "__main__":
    print("=" * 60)
    print("Trade Validator Examples")
    print("=" * 60)
    
    try:
        example_basic_validation()
        example_asset_limits()
        example_correlation_check()
        example_batch_validation()
        example_validation_status()
        
        # These modify global state, so run last
        example_kill_switch()
        example_circuit_breaker()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Examples completed")
    print("=" * 60)
