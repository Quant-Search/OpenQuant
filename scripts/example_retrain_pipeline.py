"""Example: Automated Model Retraining Pipeline Usage.

Demonstrates how to use the automated retraining pipeline to monitor
strategy performance and retrain models when DSR drops below threshold.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from openquant.ml.retrain_pipeline import (
    RetrainingPipeline,
    RetrainingConfig
)
from openquant.strategies.quant.stat_arb import StatArb
from openquant.backtest.engine import backtest_signals


def generate_synthetic_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(seed)
    
    dates = pd.date_range(start='2023-01-01', periods=n, freq='1h')
    
    # Random walk with drift
    returns = np.random.normal(0.0001, 0.02, n)
    price = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Open': price * (1 + np.random.uniform(-0.01, 0.01, n)),
        'High': price * (1 + np.random.uniform(0, 0.02, n)),
        'Low': price * (1 - np.random.uniform(0, 0.02, n)),
        'Close': price,
        'Volume': np.random.uniform(1000, 10000, n)
    }, index=dates)
    
    return df


def strategy_factory(**params):
    """Factory function for creating strategy instances."""
    return StatArb(**params)


def simulate_live_trading(
    df: pd.DataFrame,
    pipeline: RetrainingPipeline,
    initial_params: dict,
    param_grid: dict,
    strategy_id: str = "stat_arb_btc",
    monitoring_window: int = 500
):
    """Simulate live trading with periodic retraining checks."""
    current_params = initial_params.copy()
    deployed_params_history = [{"timestamp": df.index[0], "params": current_params}]
    
    print(f"\n{'='*80}")
    print(f"Starting live trading simulation for {strategy_id}")
    print(f"Initial params: {current_params}")
    print(f"{'='*80}\n")
    
    # Simulate live trading by processing data in chunks
    for i in range(monitoring_window, len(df), monitoring_window // 2):
        window_start = max(0, i - monitoring_window * 2)
        window_end = i
        df_window = df.iloc[window_start:window_end]
        
        print(f"\n--- Time: {df.index[i]} (sample {i}/{len(df)}) ---")
        
        # Generate signals with current parameters
        strategy = strategy_factory(**current_params)
        signals = strategy.generate_signals(df_window)
        
        # Backtest to get performance metrics
        result = backtest_signals(
            df_window,
            signals,
            fee_bps=2.0,
            weight=1.0
        )
        
        # Update pipeline and check if retraining is needed
        should_retrain, reason = pipeline.update_and_check(
            strategy_id=strategy_id,
            equity_curve=result.equity_curve,
            returns=result.returns,
            num_trials=10  # Number of strategies tested
        )
        
        print(f"Performance check: {reason}")
        
        if should_retrain:
            print(f"\n🔄 TRIGGERING RETRAINING for {strategy_id}")
            print(f"Reason: {reason}")
            
            # Run full retraining cycle
            deployed, new_params = pipeline.run_retrain_cycle(
                strategy_id=strategy_id,
                df=df_window,
                strategy_factory=strategy_factory,
                current_params=current_params,
                param_grid=param_grid,
                fee_bps=2.0,
                weight=1.0
            )
            
            if deployed:
                print(f"✅ NEW MODEL DEPLOYED")
                print(f"Old params: {current_params}")
                print(f"New params: {new_params}")
                current_params = new_params
                deployed_params_history.append({
                    "timestamp": df.index[i],
                    "params": new_params
                })
            else:
                print(f"❌ New model did not meet deployment threshold")
        
        # Print current stats
        stats = pipeline.get_stats(strategy_id)
        if stats["total_retrains"] > 0:
            print(f"\nRetraining Stats:")
            print(f"  Total retrains: {stats['total_retrains']}")
            print(f"  Deployments: {stats['deployments']} ({stats['deployment_rate']:.1%})")
            print(f"  Avg improvement: {stats['avg_improvement']:.2%}")
    
    print(f"\n{'='*80}")
    print(f"Simulation completed")
    print(f"Final params: {current_params}")
    print(f"Total parameter updates: {len(deployed_params_history) - 1}")
    print(f"{'='*80}\n")
    
    return deployed_params_history


def main():
    """Main example demonstrating the retraining pipeline."""
    print("Automated Model Retraining Pipeline Example")
    print("=" * 80)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    df = generate_synthetic_data(n=3000, seed=42)
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    
    # Configure retraining pipeline
    print("\n2. Configuring retraining pipeline...")
    config = RetrainingConfig(
        dsr_threshold=1.0,              # Trigger retraining when DSR < 1.0
        improvement_threshold=0.10,     # Deploy only if >10% improvement
        min_samples_retrain=500,        # Minimum samples needed
        lookback_window=1000,           # Use last 1000 samples for retraining
        holdout_fraction=0.15,          # 15% holdout for validation
        monitoring_interval_hours=6,    # Check every 6 hours (in simulation)
        max_retrain_per_day=4,          # Max 4 retrains per day
        wfo_n_splits=4,                 # 4-fold walk-forward optimization
        wfo_train_frac=0.7,             # 70% train, 30% test per split
        enable_cpcv=False,              # Use standard WFO (not CPCV)
        model_save_dir=Path("data/models"),
        metrics_save_dir=Path("data/retrain_metrics")
    )
    print(f"   DSR threshold: {config.dsr_threshold}")
    print(f"   Improvement threshold: {config.improvement_threshold:.0%}")
    print(f"   Lookback window: {config.lookback_window}")
    
    # Initialize pipeline
    print("\n3. Initializing retraining pipeline...")
    pipeline = RetrainingPipeline(config=config)
    print("   Pipeline initialized")
    
    # Define initial parameters and search grid
    print("\n4. Defining strategy parameters...")
    initial_params = {
        "lookback": 50,
        "entry_threshold": 2.0,
        "exit_threshold": 0.5
    }
    
    param_grid = {
        "lookback": [30, 40, 50, 60, 70],
        "entry_threshold": [1.5, 2.0, 2.5],
        "exit_threshold": [0.3, 0.5, 0.7]
    }
    print(f"   Initial params: {initial_params}")
    print(f"   Parameter grid size: {len(param_grid['lookback']) * len(param_grid['entry_threshold']) * len(param_grid['exit_threshold'])} combinations")
    
    # Run simulation
    print("\n5. Running live trading simulation...")
    deployed_history = simulate_live_trading(
        df=df,
        pipeline=pipeline,
        initial_params=initial_params,
        param_grid=param_grid,
        strategy_id="stat_arb_btc",
        monitoring_window=500
    )
    
    # Print final statistics
    print("\n6. Final Statistics")
    print("=" * 80)
    stats = pipeline.get_stats("stat_arb_btc")
    print(f"Total retraining attempts: {stats['total_retrains']}")
    print(f"Successful deployments: {stats['deployments']}")
    print(f"Deployment rate: {stats['deployment_rate']:.1%}")
    print(f"Average improvement: {stats['avg_improvement']:.2%}")
    print(f"Maximum improvement: {stats['max_improvement']:.2%}")
    if stats['last_retrain']:
        print(f"Last retrain: {stats['last_retrain']}")
    if stats['last_deployed']:
        print(f"Last deployment: {stats['last_deployed']}")
    
    # Show parameter evolution
    print("\n7. Parameter Evolution")
    print("=" * 80)
    for i, entry in enumerate(deployed_history):
        print(f"Version {i}: {entry['timestamp']}")
        print(f"  Params: {entry['params']}")
    
    print("\n✅ Example completed successfully!")


if __name__ == "__main__":
    main()
