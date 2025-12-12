"""Example demonstrating enhanced transaction cost models in backtesting.

This example shows how to use:
- Time-of-day spread modeling
- Volume-dependent slippage
- Market impact for large orders
- Funding rates for perpetual swaps
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from openquant.backtest import (
    backtest_signals,
    CostPreset,
    estimate_total_cost,
    compare_presets,
    TOD_MULTIPLIERS_CRYPTO_MAJOR,
)


def generate_sample_data(days: int = 365, freq: str = "1h") -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    start_date = datetime(2023, 1, 1)
    periods = days * 24 if freq == "1h" else days * 6 if freq == "4h" else days
    
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Generate synthetic price with trend and noise
    trend = np.linspace(10000, 12000, periods)
    noise = np.random.normal(0, 100, periods)
    close = trend + noise
    
    # OHLC
    high = close + np.abs(np.random.normal(0, 50, periods))
    low = close - np.abs(np.random.normal(0, 50, periods))
    open_price = close + np.random.normal(0, 30, periods)
    
    # Volume with time-of-day patterns
    base_volume = 1000000
    tod_factor = np.array([
        0.6 if h < 8 or h > 20 else 1.0 if 14 <= h <= 16 else 0.8
        for h in dates.hour
    ])
    volume = base_volume * tod_factor * (1 + np.random.normal(0, 0.3, periods))
    
    df = pd.DataFrame({
        "Open": open_price,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)
    
    return df


def generate_sample_signals(df: pd.DataFrame) -> pd.Series:
    """Generate simple trend-following signals."""
    close = df["Close"]
    sma_fast = close.rolling(20).mean()
    sma_slow = close.rolling(50).mean()
    
    signals = pd.Series(0, index=df.index)
    signals[sma_fast > sma_slow] = 1
    signals[sma_fast < sma_slow] = -1
    
    return signals


def example_basic_usage():
    """Example 1: Basic usage with enhanced cost models."""
    print("=" * 60)
    print("Example 1: Basic Usage with Enhanced Cost Models")
    print("=" * 60)
    
    # Generate data and signals
    df = generate_sample_data(days=180)
    signals = generate_sample_signals(df)
    
    # Backtest with time-of-day spread and volume slippage
    result = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=2.5,
        spread_bps=2.0,
        slippage_bps=1.0,
        weight=1.0,
        leverage=1.0,
        use_tod_spread=True,  # Enable time-of-day spread
        tod_multipliers=TOD_MULTIPLIERS_CRYPTO_MAJOR,
        use_volume_slippage=True,  # Enable volume-dependent slippage
        volume_impact_coeff=0.15,
    )
    
    final_equity = result.equity_curve.iloc[-1]
    total_return = (final_equity - 1.0) * 100
    num_trades = result.trades[result.trades > 0].sum()
    
    print(f"Final Equity: {final_equity:.4f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Number of Trades: {num_trades:.0f}")
    print()


def example_market_impact():
    """Example 2: Market impact for large orders."""
    print("=" * 60)
    print("Example 2: Market Impact for Large Orders")
    print("=" * 60)
    
    df = generate_sample_data(days=180)
    signals = generate_sample_signals(df)
    
    # Compare without and with market impact
    result_no_impact = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=2.5,
        spread_bps=2.0,
        weight=1.0,
        use_market_impact=False,
    )
    
    result_with_impact = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=2.5,
        spread_bps=2.0,
        weight=1.0,
        use_market_impact=True,
        participation_rate=0.05,
        impact_exponent=0.6,
    )
    
    print("Without Market Impact:")
    print(f"  Final Return: {(result_no_impact.equity_curve.iloc[-1] - 1) * 100:.2f}%")
    
    print("With Market Impact:")
    print(f"  Final Return: {(result_with_impact.equity_curve.iloc[-1] - 1) * 100:.2f}%")
    
    impact_cost = (result_no_impact.equity_curve.iloc[-1] - result_with_impact.equity_curve.iloc[-1]) * 100
    print(f"  Impact Cost: {impact_cost:.2f}%")
    print()


def example_funding_rates():
    """Example 3: Funding rates for perpetual swaps."""
    print("=" * 60)
    print("Example 3: Funding Rates for Perpetual Swaps")
    print("=" * 60)
    
    df = generate_sample_data(days=180)
    signals = generate_sample_signals(df)
    
    # Backtest with dynamic funding rates
    result = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=2.5,
        spread_bps=1.5,
        weight=1.0,
        use_dynamic_funding=True,
        funding_rate_bps=1.0,  # 0.01% base funding rate
        funding_interval_hours=8,  # Every 8 hours
        premium_sensitivity=0.15,
    )
    
    final_return = (result.equity_curve.iloc[-1] - 1) * 100
    print(f"Final Return with Funding: {final_return:.2f}%")
    
    # Compare without funding
    result_no_funding = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=2.5,
        spread_bps=1.5,
        weight=1.0,
        use_dynamic_funding=False,
    )
    
    funding_cost = (result_no_funding.equity_curve.iloc[-1] - result.equity_curve.iloc[-1]) * 100
    print(f"Total Funding Cost: {funding_cost:.2f}%")
    print()


def example_cost_presets():
    """Example 4: Using cost presets."""
    print("=" * 60)
    print("Example 4: Using Cost Presets")
    print("=" * 60)
    
    df = generate_sample_data(days=180)
    signals = generate_sample_signals(df)
    
    # Use crypto spot tier-1 preset
    preset = CostPreset.crypto_spot_tier1()
    
    result = backtest_signals(
        df=df,
        signals=signals,
        weight=1.0,
        leverage=1.0,
        **preset
    )
    
    print("Using CostPreset.crypto_spot_tier1():")
    print(f"  Final Return: {(result.equity_curve.iloc[-1] - 1) * 100:.2f}%")
    print()
    
    # Compare presets
    print("Comparing different presets:")
    comparison = compare_presets(
        "fx_major_ecn",
        "crypto_spot_tier1",
        "crypto_perp_tier1",
        "altcoin_spot"
    )
    print(comparison[["fee_bps", "spread_bps", "slippage_bps"]])
    print()


def example_cost_estimation():
    """Example 5: Estimate total trading costs."""
    print("=" * 60)
    print("Example 5: Cost Estimation")
    print("=" * 60)
    
    preset = CostPreset.crypto_spot_tier1()
    
    # Estimate costs for a typical strategy
    costs = estimate_total_cost(
        preset=preset,
        avg_position_changes_per_day=4.0,  # 4 trades per day
        avg_holding_period_days=0.5,  # Hold for 12 hours
    )
    
    print("Estimated costs for crypto spot tier-1:")
    print(f"  Per-trade cost: {costs['per_trade_cost_bps']:.2f} bps")
    print(f"  Daily trading cost: {costs['daily_trading_cost_bps']:.2f} bps")
    print(f"  Total cycle cost: {costs['total_cycle_cost_bps']:.2f} bps ({costs['total_cycle_cost_pct']:.4f}%)")
    print(f"  Breakeven return: {costs['breakeven_return_bps']:.2f} bps")
    print()
    
    # Compare with perpetual swaps
    perp_preset = CostPreset.crypto_perp_tier1(avg_funding_rate_bps=1.0)
    perp_costs = estimate_total_cost(
        preset=perp_preset,
        avg_position_changes_per_day=4.0,
        avg_holding_period_days=0.5,
        funding_rate_bps_if_perp=1.0,
    )
    
    print("Estimated costs for crypto perpetual tier-1:")
    print(f"  Per-trade cost: {perp_costs['per_trade_cost_bps']:.2f} bps")
    print(f"  Holding cost/day: {perp_costs['holding_cost_per_day_bps']:.2f} bps")
    print(f"  Total cycle cost: {perp_costs['total_cycle_cost_bps']:.2f} bps ({perp_costs['total_cycle_cost_pct']:.4f}%)")
    print()


def example_all_features_combined():
    """Example 6: All features combined."""
    print("=" * 60)
    print("Example 6: All Enhanced Features Combined")
    print("=" * 60)
    
    df = generate_sample_data(days=365)
    signals = generate_sample_signals(df)
    
    # Full-featured backtest
    result = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=2.5,
        spread_bps=2.0,
        slippage_bps=1.0,
        weight=1.0,
        leverage=2.0,  # 2x leverage
        use_tod_spread=True,
        tod_multipliers=TOD_MULTIPLIERS_CRYPTO_MAJOR,
        use_volume_slippage=True,
        volume_impact_coeff=0.15,
        use_market_impact=True,
        participation_rate=0.05,
        impact_exponent=0.6,
        use_dynamic_funding=True,
        funding_rate_bps=1.0,
        funding_interval_hours=8,
        premium_sensitivity=0.15,
    )
    
    final_return = (result.equity_curve.iloc[-1] - 1) * 100
    num_trades = result.trades[result.trades > 0].sum()
    
    print("Full-featured backtest results:")
    print(f"  Final Return: {final_return:.2f}%")
    print(f"  Number of Trades: {num_trades:.0f}")
    print(f"  Annualized Return: {(result.equity_curve.iloc[-1] ** (365 / len(df)) - 1) * 100:.2f}%")
    print()


if __name__ == "__main__":
    example_basic_usage()
    example_market_impact()
    example_funding_rates()
    example_cost_presets()
    example_cost_estimation()
    example_all_features_combined()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
