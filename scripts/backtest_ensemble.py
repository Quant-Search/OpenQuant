"""Backtest the Ensemble Strategy.

Run this to validate the ensemble strategy performance on historical data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def run_ensemble_backtest():
    """Run backtest on ensemble strategy."""
    print("=" * 60)
    print("Ensemble Strategy Backtest")
    print("=" * 60)
    print()
    
    # Import modules
    from openquant.strategies.ensemble_strategy import EnsembleStrategy
    from openquant.backtest.gpu_backtest import backtest_signals_gpu, is_gpu_backtest_available
    
    gpu_available = is_gpu_backtest_available()
    print(f"GPU Available: {gpu_available}")
    
    # Generate synthetic data for testing (replace with real data in production)
    print("\n1. Generating test data...")
    np.random.seed(42)
    n_bars = 2000  # About 3 months of hourly data
    
    # Generate trending market data with some noise
    trend = np.cumsum(np.random.randn(n_bars) * 0.0015 + 0.0001)
    noise = np.random.randn(n_bars) * 0.002
    price = 1.10 + trend + noise  # EURUSD-like
    
    df = pd.DataFrame({
        'Open': price + np.random.randn(n_bars) * 0.0005,
        'High': price + abs(np.random.randn(n_bars) * 0.001),
        'Low': price - abs(np.random.randn(n_bars) * 0.001),
        'Close': price,
        'Volume': np.random.randint(1000, 100000, n_bars)
    }, index=pd.date_range('2024-01-01', periods=n_bars, freq='1h'))
    
    print(f"   Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print(f"   Price range: {df['Close'].min():.4f} - {df['Close'].max():.4f}")
    
    # Initialize strategy
    print("\n2. Initializing Ensemble Strategy...")
    strategy = EnsembleStrategy(
        min_agreement=0.6,
        probability_threshold=0.55,
        use_regime_filter=True
    )
    
    # Generate signals for each bar (walk-forward style)
    print("\n3. Generating signals (this may take a moment)...")
    signals = []
    lookback = 200
    
    for i in range(lookback, len(df)):
        # Get historical slice
        hist = df.iloc[i-lookback:i+1].copy()
        
        try:
            sig = strategy.generate_signals(hist)
            signal_val = sig['signal'].iloc[-1] if 'signal' in sig.columns else 0
            prob = sig['probability'].iloc[-1] if 'probability' in sig.columns else 0.5
        except Exception as e:
            signal_val = 0
            prob = 0.5
            
        signals.append({
            'timestamp': df.index[i],
            'signal': signal_val,
            'probability': prob
        })
        
        if i % 500 == 0:
            print(f"   Processed {i}/{len(df)} bars...")
            
    signals_df = pd.DataFrame(signals).set_index('timestamp')
    print(f"   Generated {len(signals_df)} signals")
    
    # Backtest
    print("\n4. Running backtest...")
    
    # Align data
    backtest_df = df.loc[signals_df.index]
    
    # Simple backtest calculation
    positions = signals_df['signal'].values
    returns = backtest_df['Close'].pct_change().values[1:]
    positions = positions[:-1]  # Shift to match returns
    
    strategy_returns = positions * returns
    
    # Calculate metrics
    total_return = np.sum(strategy_returns)
    n_trades = np.sum(np.abs(np.diff(positions)) > 0)
    win_trades = np.sum(strategy_returns > 0)
    loss_trades = np.sum(strategy_returns < 0)
    win_rate = win_trades / (win_trades + loss_trades) if (win_trades + loss_trades) > 0 else 0
    
    avg_win = np.mean(strategy_returns[strategy_returns > 0]) if np.any(strategy_returns > 0) else 0
    avg_loss = np.mean(np.abs(strategy_returns[strategy_returns < 0])) if np.any(strategy_returns < 0) else 0
    
    sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252 * 24) if np.std(strategy_returns) > 0 else 0
    
    # Max drawdown
    cumulative = np.cumsum(strategy_returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = rolling_max - cumulative
    max_dd = np.max(drawdown)
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print(f"Total Return:      {total_return:.2%}")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print(f"Max Drawdown:      {max_dd:.2%}")
    print(f"Number of Trades:  {n_trades}")
    print(f"Win Rate:          {win_rate:.1%}")
    print(f"Avg Win:           {avg_win:.4%}")
    print(f"Avg Loss:          {avg_loss:.4%}")
    
    # Signals breakdown
    long_signals = np.sum(positions == 1)
    short_signals = np.sum(positions == -1)
    flat_signals = np.sum(positions == 0)
    
    print(f"\nSignal Distribution:")
    print(f"   Long:  {long_signals} ({long_signals/len(positions)*100:.1f}%)")
    print(f"   Short: {short_signals} ({short_signals/len(positions)*100:.1f}%)")
    print(f"   Flat:  {flat_signals} ({flat_signals/len(positions)*100:.1f}%)")
    
    # Profit factor
    gross_profit = np.sum(strategy_returns[strategy_returns > 0])
    gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    print(f"\nProfit Factor:     {profit_factor:.2f}")
    
    # Save results
    results = {
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'n_trades': int(n_trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'timestamp': datetime.now().isoformat()
    }
    
    import json
    Path("data").mkdir(exist_ok=True)
    with open("data/backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to data/backtest_results.json")
    
    # Verdict
    print("\n" + "=" * 60)
    if sharpe > 1.0 and win_rate > 0.50:
        print("✅ STRATEGY LOOKS PROMISING!")
    elif sharpe > 0.5:
        print("⚠️  STRATEGY NEEDS OPTIMIZATION")
    else:
        print("❌ STRATEGY NEEDS SIGNIFICANT WORK")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    run_ensemble_backtest()
