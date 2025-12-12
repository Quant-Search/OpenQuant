"""
Example usage of enhanced transaction cost modeling in backtests.

This example demonstrates:
1. Time-of-day spread adjustments
2. Volume-dependent market impact (square-root model)
3. Tick size constraints

Run with:
    python scripts/example_enhanced_costs.py
"""
import pandas as pd
import numpy as np
from openquant.backtest import (
    backtest_signals,
    SpreadSchedule,
    MarketImpactModel,
    TickRounder,
    TransactionCostModel
)


def create_sample_data(n: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data with timestamps."""
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    
    # Generate synthetic price data
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2
    volume = np.random.lognormal(15, 1, n)  # Realistic volume distribution
    
    return pd.DataFrame({
        'Close': close,
        'High': high,
        'Low': low,
        'Volume': volume
    }, index=dates)


def create_sample_signals(n: int = 1000) -> pd.Series:
    """Create simple momentum signals."""
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    
    # Simple momentum: long when price above SMA, flat otherwise
    np.random.seed(42)
    signals = np.where(np.random.rand(n) > 0.7, 1, 0)
    signals = pd.Series(signals, index=dates)
    
    return signals


def example_basic_enhanced_costs():
    """Example 1: Enable enhanced costs with defaults."""
    print("=" * 60)
    print("Example 1: Basic Enhanced Cost Model")
    print("=" * 60)
    
    df = create_sample_data(1000)
    signals = create_sample_signals(1000)
    
    # Basic enhanced costs (uses defaults)
    result = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=1.0,
        spread_bps=5.0,
        use_enhanced_costs=True  # Enable enhanced modeling
    )
    
    final_equity = result.equity_curve.iloc[-1]
    total_return = (final_equity - 1.0) * 100
    print(f"Final Equity: {final_equity:.4f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {result.trades.sum():.0f}")
    print()


def example_custom_spread_schedule():
    """Example 2: Custom spread schedule for specific market hours."""
    print("=" * 60)
    print("Example 2: Custom Spread Schedule (Forex Hours)")
    print("=" * 60)
    
    df = create_sample_data(1000)
    signals = create_sample_signals(1000)
    
    # Custom spread schedule for Forex (wider during Asian session)
    spread_schedule = SpreadSchedule(
        base_spread_bps=3.0,
        hour_multipliers={
            0: 2.5, 1: 2.5, 2: 2.5, 3: 2.5,   # Asian overnight
            4: 2.0, 5: 2.0, 6: 2.0, 7: 2.0,   # Asian morning
            8: 1.5, 9: 1.2,                    # London open
            10: 1.0, 11: 1.0, 12: 1.0,         # London active
            13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, # NY overlap
            17: 1.2, 18: 1.3, 19: 1.5,         # After hours
            20: 1.8, 21: 2.0, 22: 2.2, 23: 2.5 # Late evening
        }
    )
    
    result = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=1.0,
        spread_schedule=spread_schedule,
        use_enhanced_costs=True
    )
    
    final_equity = result.equity_curve.iloc[-1]
    total_return = (final_equity - 1.0) * 100
    print(f"Final Equity: {final_equity:.4f}")
    print(f"Total Return: {total_return:.2f}%")
    print()


def example_market_impact_model():
    """Example 3: Volume-dependent market impact."""
    print("=" * 60)
    print("Example 3: Market Impact Model (Square-Root)")
    print("=" * 60)
    
    df = create_sample_data(1000)
    signals = create_sample_signals(1000)
    
    # Configure market impact based on order size vs volume
    impact_model = MarketImpactModel(
        impact_coeff=0.15,        # Market-specific calibration
        volume_lookback=20,       # 20-period average volume
        volatility_lookback=20,   # 20-period volatility
        min_impact_bps=0.5,       # Minimum impact
        max_impact_bps=30.0       # Cap for extreme cases
    )
    
    result = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=1.0,
        spread_bps=5.0,
        impact_model=impact_model,
        use_enhanced_costs=True
    )
    
    final_equity = result.equity_curve.iloc[-1]
    total_return = (final_equity - 1.0) * 100
    print(f"Final Equity: {final_equity:.4f}")
    print(f"Total Return: {total_return:.2f}%")
    print()


def example_tick_constraints():
    """Example 4: Tick size constraints."""
    print("=" * 60)
    print("Example 4: Tick Size Constraints")
    print("=" * 60)
    
    df = create_sample_data(1000)
    signals = create_sample_signals(1000)
    
    # Different tick sizes for different markets
    tick_rounder_stocks = TickRounder(tick_size=0.01)    # $0.01 for stocks
    tick_rounder_forex = TickRounder(tick_size=0.00001)  # 0.1 pip for forex
    tick_rounder_btc = TickRounder(tick_size=1.0)        # $1 for BTC
    
    # Test with stock tick size
    result = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=1.0,
        spread_bps=5.0,
        tick_rounder=tick_rounder_stocks,
        use_enhanced_costs=True
    )
    
    final_equity = result.equity_curve.iloc[-1]
    total_return = (final_equity - 1.0) * 100
    print(f"Tick Size $0.01 (Stocks):")
    print(f"  Final Equity: {final_equity:.4f}")
    print(f"  Total Return: {total_return:.2f}%")
    print()


def example_integrated_model():
    """Example 5: Full integrated transaction cost model."""
    print("=" * 60)
    print("Example 5: Integrated Transaction Cost Model")
    print("=" * 60)
    
    df = create_sample_data(1000)
    signals = create_sample_signals(1000)
    
    # Create integrated model
    cost_model = TransactionCostModel(
        spread_schedule=SpreadSchedule(base_spread_bps=3.0),
        impact_model=MarketImpactModel(impact_coeff=0.1),
        tick_rounder=TickRounder(tick_size=0.01)
    )
    
    # Compare with and without enhanced costs
    result_basic = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=1.0,
        spread_bps=5.0,
        slippage_bps=2.0,
        use_enhanced_costs=False  # Basic model
    )
    
    result_enhanced = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=1.0,
        spread_schedule=cost_model.spread_schedule,
        impact_model=cost_model.impact_model,
        tick_rounder=cost_model.tick_rounder,
        use_enhanced_costs=True  # Enhanced model
    )
    
    basic_return = (result_basic.equity_curve.iloc[-1] - 1.0) * 100
    enhanced_return = (result_enhanced.equity_curve.iloc[-1] - 1.0) * 100
    
    print(f"Basic Model Return:    {basic_return:.2f}%")
    print(f"Enhanced Model Return: {enhanced_return:.2f}%")
    print(f"Difference:            {enhanced_return - basic_return:.2f}%")
    print()
    print("Note: Enhanced model typically shows lower returns due to")
    print("      more realistic cost modeling (time-varying spreads,")
    print("      volume-dependent slippage, tick constraints).")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ENHANCED TRANSACTION COST MODELING EXAMPLES")
    print("=" * 60 + "\n")
    
    example_basic_enhanced_costs()
    example_custom_spread_schedule()
    example_market_impact_model()
    example_tick_constraints()
    example_integrated_model()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
