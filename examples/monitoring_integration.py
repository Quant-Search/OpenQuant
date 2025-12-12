"""Integration example: Using PerformanceTracker with paper trading.

Shows how to integrate real-time monitoring into a trading loop.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from datetime import datetime, timedelta
from openquant.monitoring import PerformanceTracker


class SimpleMockBroker:
    """Mock broker for demonstration purposes."""
    
    def __init__(self, initial_balance=100000.0):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        
    def get_equity(self):
        return self.balance + sum(pos['value'] for pos in self.positions.values())
    
    def execute_trade(self, symbol, return_pct):
        """Simulate a trade execution."""
        position_size = self.balance * 0.1  # 10% of balance
        pnl = position_size * return_pct
        self.balance += pnl
        return pnl


def trading_loop_with_monitoring():
    """Example trading loop with real-time performance monitoring."""
    
    print("=== Trading Loop with Real-Time Monitoring ===\n")
    
    # Initialize broker
    broker = SimpleMockBroker(initial_balance=100000.0)
    
    # Load backtest results (normally from file/database)
    backtest_sharpe = 2.2
    backtest_max_dd = 0.12
    backtest_corr = np.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    
    # Initialize performance tracker
    tracker = PerformanceTracker(
        backtest_correlation_matrix=backtest_corr,
        backtest_sharpe=backtest_sharpe,
        backtest_max_drawdown=backtest_max_dd,
        alert_threshold=0.20,
        sharpe_halflife_days=30,
        freq="1h"  # Hourly trading
    )
    
    print(f"Initial Balance: ${broker.initial_balance:,.2f}")
    print(f"Backtest Sharpe: {backtest_sharpe:.2f}")
    print(f"Backtest Max DD: {backtest_max_dd:.1%}")
    print(f"Alert Threshold: 20%\n")
    
    # Simulate trading loop
    symbols = ["BTC/USD", "ETH/USD"]
    np.random.seed(42)
    
    for hour in range(168):  # 1 week of hourly trading
        # Get current equity
        current_equity = broker.get_equity()
        
        # Generate returns for each symbol (simulated)
        symbol_returns = {}
        for symbol in symbols:
            # Simulate strategy signal and execution
            ret = np.random.normal(0.0002, 0.005)  # Hourly return
            pnl = broker.execute_trade(symbol, ret)
            symbol_returns[symbol] = ret
        
        # Calculate portfolio return
        prev_equity = current_equity
        current_equity = broker.get_equity()
        portfolio_return = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        
        # Update performance tracker
        tracker.update(
            equity=current_equity,
            returns=portfolio_return,
            symbol_returns=symbol_returns,
            timestamp=datetime.now() + timedelta(hours=hour)
        )
        
        # Periodic reporting
        if hour % 24 == 0:  # Every day
            metrics = tracker.get_current_metrics()
            print(f"\n--- Day {hour // 24} Report ---")
            print(f"Equity: ${current_equity:,.2f}")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
            print(f"Drawdown: {metrics['drawdown']:.2%}")
            if metrics['correlation_drift'] is not None:
                print(f"Correlation Drift: {metrics['correlation_drift']:.4f}")
            
            # Check for performance degradation
            if metrics['sharpe_degradation_pct']:
                deg = metrics['sharpe_degradation_pct']
                if deg > 20:
                    print(f"⚠️  WARNING: Sharpe degraded by {deg:.1f}%")
            
            if metrics['drawdown'] > backtest_max_dd * 1.5:
                print(f"⚠️  WARNING: Drawdown exceeded 150% of backtest!")
    
    # Final summary
    print("\n" + "="*60)
    print("=== Final Performance Summary ===")
    print("="*60)
    print(tracker.get_summary())
    
    final_equity = broker.get_equity()
    total_return = (final_equity - broker.initial_balance) / broker.initial_balance
    print(f"\nTotal Return: {total_return:+.2%}")
    print(f"Final Equity: ${final_equity:,.2f}")


def circuit_breaker_example():
    """Example: Using PerformanceTracker as a circuit breaker."""
    
    print("\n" + "="*60)
    print("=== Circuit Breaker Example ===")
    print("="*60 + "\n")
    
    tracker = PerformanceTracker(
        backtest_sharpe=2.5,
        backtest_max_drawdown=0.15,
        alert_threshold=0.20,
        freq="1d"
    )
    
    broker = SimpleMockBroker(initial_balance=100000.0)
    
    # Trading loop with circuit breaker
    np.random.seed(123)
    trading_enabled = True
    
    for day in range(60):
        if not trading_enabled:
            print(f"Day {day}: Trading HALTED due to circuit breaker")
            continue
        
        # Simulate trading
        ret = np.random.normal(-0.003, 0.04)  # Simulate poor conditions
        broker.execute_trade("BTC/USD", ret)
        
        equity = broker.get_equity()
        tracker.update(equity=equity, returns=ret)
        
        # Check circuit breaker conditions
        metrics = tracker.get_current_metrics()
        
        # Halt trading if:
        # 1. Drawdown exceeds 25%
        # 2. Sharpe degrades by more than 40%
        if metrics['drawdown'] > 0.25:
            print(f"\n🛑 CIRCUIT BREAKER TRIGGERED on Day {day}")
            print(f"   Reason: Excessive drawdown ({metrics['drawdown']:.1%})")
            trading_enabled = False
        elif (metrics['sharpe_degradation_pct'] and 
              metrics['sharpe_degradation_pct'] > 40):
            print(f"\n🛑 CIRCUIT BREAKER TRIGGERED on Day {day}")
            print(f"   Reason: Sharpe degradation ({metrics['sharpe_degradation_pct']:.1f}%)")
            trading_enabled = False
        
        if day % 10 == 0:
            print(f"Day {day}: Equity=${equity:,.2f}, Sharpe={metrics['sharpe_ratio']:.2f}, DD={metrics['drawdown']:.1%}")
    
    print("\n" + tracker.get_summary())


if __name__ == "__main__":
    trading_loop_with_monitoring()
    circuit_breaker_example()
