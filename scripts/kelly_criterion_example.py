"""Example demonstrating Kelly Criterion adaptive position sizing.

This script shows how to use the Kelly Criterion implementation
with volatility adjustment and drawdown-based scaling in paper trading.
"""
from __future__ import annotations
import numpy as np
from openquant.paper.state import PortfolioState
from openquant.paper.simulator import (
    MarketSnapshot,
    rebalance_to_targets_with_kelly,
)
from openquant.risk.kelly_criterion import KellyCriterion


def run_kelly_example():
    """Demonstrate Kelly Criterion position sizing."""
    print("=" * 60)
    print("Kelly Criterion Adaptive Position Sizing Example")
    print("=" * 60)
    
    # Initialize portfolio state
    state = PortfolioState(cash=100_000.0)
    
    # Define a trading key (exchange, symbol, timeframe, strategy)
    key = ("paper", "BTC/USD", "1h", "test_strategy")
    
    # Initialize Kelly sizers (one per trading key)
    kelly_sizers = {
        key: KellyCriterion(
            kelly_fraction=0.5,  # Half Kelly for safety
            min_trades=20,       # Conservative until 20 trades
            volatility_target=0.20,  # Target 20% annualized vol
            max_drawdown_threshold=0.15,  # Reduce size after 15% DD
            max_position_size=1.0,  # Max 100% of capital
        )
    }
    
    # Simulate price history for volatility calculation
    # Generate 100 bars of price data
    np.random.seed(42)
    base_price = 50000.0
    returns = np.random.normal(0.0001, 0.02, 100)
    prices = base_price * np.exp(np.cumsum(returns))
    
    print(f"\nInitial Setup:")
    print(f"  Starting capital: ${state.cash:,.2f}")
    print(f"  Trading key: {key}")
    print(f"  Kelly fraction: {kelly_sizers[key].kelly_fraction}")
    print(f"  Price history length: {len(prices)} bars")
    print(f"  Current price: ${prices[-1]:,.2f}")
    
    # Create market snapshot
    snap = MarketSnapshot(
        prices={key: prices[-1]},
        price_history={key: prices},
    )
    
    # Example 1: First trade (no history yet)
    print("\n" + "=" * 60)
    print("Example 1: First Trade (No History)")
    print("=" * 60)
    
    # Signal: +1 (long)
    targets = [(key, 1.0)]
    
    summary, kelly_stats = rebalance_to_targets_with_kelly(
        state=state,
        targets=targets,
        snap=snap,
        kelly_sizers=kelly_sizers,
        fee_bps=10,  # 0.1% fees
        volatility_window=20,
        annualization_factor=252 * 24,  # Hourly bars -> annual
    )
    
    print(f"\nTrade Summary:")
    print(f"  Orders executed: {summary['orders']}")
    print(f"  Turnover: ${summary.get('turnover', 0):,.2f}")
    
    print(f"\nKelly Statistics:")
    stats = kelly_stats[key]
    print(f"  Kelly fraction: {stats['kelly_fraction']:.4f}")
    print(f"  Win rate: {stats['win_rate']:.2%}")
    print(f"  Total trades: {stats['total_trades']}")
    print(f"  Current drawdown: {stats['current_drawdown']:.2%}")
    
    print(f"\nPortfolio State:")
    print(f"  Cash: ${state.cash:,.2f}")
    print(f"  Position: {state.position(key):.6f} units")
    print(f"  Position value: ${state.position(key) * prices[-1]:,.2f}")
    
    # Simulate some trades to build history
    print("\n" + "=" * 60)
    print("Example 2: Simulating Trade History")
    print("=" * 60)
    
    # Manually add trade history to Kelly sizer
    kelly = kelly_sizers[key]
    
    # Simulate 30 trades with 60% win rate
    print("\nSimulating 30 trades (60% win rate, 1.5 payoff ratio)...")
    for i in range(30):
        if i < 18:  # 60% wins
            pnl = np.random.uniform(100, 200)
            kelly.record_trade(pnl=pnl, entry_price=50000, exit_price=51000, size=0.1)
        else:  # 40% losses
            pnl = np.random.uniform(-150, -50)
            kelly.record_trade(pnl=pnl, entry_price=50000, exit_price=49000, size=0.1)
    
    kelly.update_equity(state.cash + state.position(key) * prices[-1])
    
    stats = kelly.get_stats()
    print(f"\nUpdated Kelly Statistics:")
    print(f"  Win rate: {stats.win_rate:.2%}")
    print(f"  Avg win: ${stats.avg_win:.2f}")
    print(f"  Avg loss: ${stats.avg_loss:.2f}")
    print(f"  Payoff ratio: {stats.payoff_ratio:.2f}")
    print(f"  Kelly fraction: {stats.kelly_fraction:.4f}")
    print(f"  Expectancy: ${stats.expectancy():.2f}")
    
    # Example 3: Trade with updated Kelly
    print("\n" + "=" * 60)
    print("Example 3: Trade with Updated Kelly Statistics")
    print("=" * 60)
    
    # New price
    new_price = prices[-1] * 1.02
    snap = MarketSnapshot(
        prices={key: new_price},
        price_history={key: prices},
    )
    
    # Close previous position first
    targets = [(key, 0.0)]
    summary, kelly_stats = rebalance_to_targets_with_kelly(
        state=state,
        targets=targets,
        snap=snap,
        kelly_sizers=kelly_sizers,
        fee_bps=10,
    )
    
    print(f"\nClosed previous position")
    print(f"  Cash: ${state.cash:,.2f}")
    
    # Open new position with updated Kelly
    targets = [(key, 1.0)]
    snap.prices[key] = new_price * 1.01
    
    summary, kelly_stats = rebalance_to_targets_with_kelly(
        state=state,
        targets=targets,
        snap=snap,
        kelly_sizers=kelly_sizers,
        fee_bps=10,
    )
    
    print(f"\nNew Trade Summary:")
    print(f"  Orders executed: {summary['orders']}")
    print(f"  Turnover: ${summary.get('turnover', 0):,.2f}")
    
    stats = kelly_stats[key]
    print(f"\nFinal Kelly Statistics:")
    print(f"  Kelly fraction: {stats['kelly_fraction']:.4f}")
    print(f"  Total trades: {stats['total_trades']}")
    print(f"  Win rate: {stats['win_rate']:.2%}")
    print(f"  Payoff ratio: {stats['payoff_ratio']:.2f}")
    
    # Example 4: Drawdown impact
    print("\n" + "=" * 60)
    print("Example 4: Drawdown Impact on Position Size")
    print("=" * 60)
    
    # Simulate drawdown
    kelly.peak_equity = 120_000
    kelly.current_equity = 100_000
    kelly.current_drawdown = 0.167  # 16.7% drawdown
    
    print(f"\nSimulated Drawdown:")
    print(f"  Peak equity: ${kelly.peak_equity:,.2f}")
    print(f"  Current equity: ${kelly.current_equity:,.2f}")
    print(f"  Drawdown: {kelly.current_drawdown:.2%}")
    
    # Compute position size with drawdown
    base_size = kelly.compute_base_kelly_size()
    volatility = 0.25  # 25% volatility
    final_size = kelly.compute_position_size(volatility=volatility)
    
    print(f"\nPosition Sizing:")
    print(f"  Base Kelly size: {base_size:.2%}")
    print(f"  Asset volatility: {volatility:.2%}")
    print(f"  Final size (with DD scaling): {final_size:.2%}")
    print(f"  Reduction due to DD: {(1 - final_size/base_size):.1%}")
    
    print("\n" + "=" * 60)
    print("Kelly Criterion Example Complete")
    print("=" * 60)


if __name__ == "__main__":
    run_kelly_example()
