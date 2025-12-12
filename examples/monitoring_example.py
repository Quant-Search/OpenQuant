"""Example usage of the PerformanceTracker monitoring system.

Demonstrates:
- Exponentially-weighted rolling Sharpe (30-day half-life)
- Drawdown tracking from peak
- Correlation drift detection
- Alert triggering on metric degradation
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.monitoring import PerformanceTracker


def simulate_trading_session():
    """Simulate a trading session with performance monitoring."""
    
    # Create reference correlation matrix from "backtest"
    # Simulating 3 assets with known correlations
    backtest_corr = np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])
    
    # Initialize tracker with backtest reference metrics
    tracker = PerformanceTracker(
        backtest_correlation_matrix=backtest_corr,
        backtest_sharpe=2.5,
        backtest_max_drawdown=0.15,
        alert_threshold=0.20,
        sharpe_halflife_days=30,
        freq="1d"
    )
    
    print("=== Performance Tracker Initialized ===")
    print(f"Backtest Sharpe: 2.5")
    print(f"Backtest Max DD: 15%")
    print(f"Alert Threshold: 20%")
    print()
    
    # Simulate trading with 100 days of returns
    np.random.seed(42)
    initial_equity = 100000.0
    current_equity = initial_equity
    
    # Three trading symbols
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    
    print("=== Simulating 100 Days of Trading ===")
    
    # First 50 days: good performance (matches backtest)
    for day in range(50):
        # Generate correlated returns for 3 symbols
        mean_returns = [0.001, 0.0008, 0.0012]
        cov_matrix = backtest_corr * 0.02 ** 2  # 2% daily volatility
        symbol_rets = np.random.multivariate_normal(mean_returns, cov_matrix)
        
        # Portfolio return (equal weight)
        portfolio_return = np.mean(symbol_rets)
        current_equity *= (1 + portfolio_return)
        
        # Update tracker
        symbol_returns = {symbol: ret for symbol, ret in zip(symbols, symbol_rets)}
        tracker.update(
            equity=current_equity,
            returns=portfolio_return,
            symbol_returns=symbol_returns
        )
        
        if day % 10 == 0:
            print(f"Day {day}: Equity=${current_equity:,.2f}, Return={portfolio_return:.4f}")
    
    print("\n--- Performance after 50 days (good period) ---")
    print(tracker.get_summary())
    print()
    
    # Next 25 days: performance degrades, correlation structure changes
    print("=== Degradation Period (Days 51-75) ===")
    
    # Modified correlation matrix (drift)
    degraded_corr = np.array([
        [1.0, 0.2, -0.1],  # BTC becomes less correlated
        [0.2, 1.0, 0.8],   # ETH and SOL become more correlated
        [-0.1, 0.8, 1.0]
    ])
    
    for day in range(50, 75):
        # Lower mean returns, higher volatility
        mean_returns = [-0.0005, -0.0003, -0.0007]
        cov_matrix = degraded_corr * 0.04 ** 2  # 4% daily volatility (doubled)
        symbol_rets = np.random.multivariate_normal(mean_returns, cov_matrix)
        
        portfolio_return = np.mean(symbol_rets)
        current_equity *= (1 + portfolio_return)
        
        symbol_returns = {symbol: ret for symbol, ret in zip(symbols, symbol_rets)}
        tracker.update(
            equity=current_equity,
            returns=portfolio_return,
            symbol_returns=symbol_returns
        )
        
        if day % 5 == 0:
            print(f"Day {day}: Equity=${current_equity:,.2f}, Return={portfolio_return:.4f}")
    
    print("\n--- Performance after degradation period ---")
    print(tracker.get_summary())
    print()
    
    # Last 25 days: recovery
    print("=== Recovery Period (Days 76-100) ===")
    
    for day in range(75, 100):
        mean_returns = [0.0015, 0.0012, 0.0018]
        cov_matrix = backtest_corr * 0.02 ** 2  # Back to normal
        symbol_rets = np.random.multivariate_normal(mean_returns, cov_matrix)
        
        portfolio_return = np.mean(symbol_rets)
        current_equity *= (1 + portfolio_return)
        
        symbol_returns = {symbol: ret for symbol, ret in zip(symbols, symbol_rets)}
        tracker.update(
            equity=current_equity,
            returns=portfolio_return,
            symbol_returns=symbol_returns
        )
        
        if day % 10 == 0:
            print(f"Day {day}: Equity=${current_equity:,.2f}, Return={portfolio_return:.4f}")
    
    print("\n=== Final Performance Summary ===")
    print(tracker.get_summary())
    print()
    
    # Get detailed metrics
    metrics = tracker.get_current_metrics()
    print("=== Detailed Metrics ===")
    for key, value in metrics.items():
        if value is not None and key not in ['timestamp']:
            if isinstance(value, float):
                if 'pct' in key or 'degradation' in key:
                    print(f"{key}: {value:.2f}%")
                else:
                    print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    print()
    
    # Show metrics history
    history = tracker.get_metrics_history(lookback_days=100)
    print(f"=== Metrics History ===")
    print(f"Total observations: {len(history)}")
    if len(history) >= 10:
        print("\nFirst 5 observations:")
        for i, obs in enumerate(history[:5]):
            print(f"  {i+1}. Sharpe={obs['sharpe_ratio']:.3f}, DD={obs['drawdown']:.3%}")
        
        print("\nLast 5 observations:")
        for i, obs in enumerate(history[-5:]):
            print(f"  {len(history)-4+i}. Sharpe={obs['sharpe_ratio']:.3f}, DD={obs['drawdown']:.3%}")


def demonstrate_alert_triggering():
    """Demonstrate alert triggering when metrics degrade."""
    
    print("\n" + "="*60)
    print("=== Demonstrating Alert Triggering ===")
    print("="*60 + "\n")
    
    # Create tracker with high reference Sharpe
    tracker = PerformanceTracker(
        backtest_sharpe=3.0,
        backtest_max_drawdown=0.10,
        alert_threshold=0.20,
        freq="1d"
    )
    
    # Reset alerts to ensure they fire
    tracker.reset_alerts()
    
    initial_equity = 100000.0
    current_equity = initial_equity
    
    print("Backtest Sharpe: 3.0, Backtest Max DD: 10%")
    print("Alert threshold: 20% degradation\n")
    
    # Good period first
    print("--- Period 1: Good performance (30 days) ---")
    np.random.seed(123)
    for day in range(30):
        ret = np.random.normal(0.002, 0.015)  # Good Sharpe
        current_equity *= (1 + ret)
        tracker.update(equity=current_equity, returns=ret)
    
    metrics = tracker.get_current_metrics()
    print(f"Current Sharpe: {metrics['sharpe_ratio']:.3f}")
    print(f"Current DD: {metrics['drawdown']:.2%}")
    print()
    
    # Bad period - should trigger alerts
    print("--- Period 2: Poor performance (30 days) - Expect alerts! ---")
    for day in range(30):
        ret = np.random.normal(-0.001, 0.03)  # Poor returns, high vol
        current_equity *= (1 + ret)
        tracker.update(equity=current_equity, returns=ret)
    
    metrics = tracker.get_current_metrics()
    print(f"\nCurrent Sharpe: {metrics['sharpe_ratio']:.3f}")
    print(f"Current DD: {metrics['drawdown']:.2%}")
    if metrics['sharpe_degradation_pct'] is not None:
        print(f"Sharpe degradation: {metrics['sharpe_degradation_pct']:.1f}%")
    if metrics['drawdown_increase_pct'] is not None:
        print(f"Drawdown increase: {metrics['drawdown_increase_pct']:.1f}%")


if __name__ == "__main__":
    simulate_trading_session()
    demonstrate_alert_triggering()
    
    print("\n" + "="*60)
    print("Example completed! Check console output above for alerts.")
    print("="*60)
