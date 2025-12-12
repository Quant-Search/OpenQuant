"""Backtest ML Signal Prediction.

Trains an ML model (Gradient Boosting) on the first 70% of data
and backtests on the remaining 30% out-of-sample data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt

from openquant.ml.ml_predictor import MLSignalPredictor

def download_data():
    """Download EUR/USD data."""
    print("📥 Downloading 2 years of EUR/USD data for ML...")
    try:
        import yfinance as yf
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # 1 year safe limit
        
        df = yf.Ticker("EURUSD=X").history(start=start_date, end=end_date, interval="1h")
        
        if not df.empty:
            print(f"   ✅ Downloaded {len(df)} bars")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"   Error: {e}")
    return None

def run_backtest():
    print("=" * 70)
    print("🧠 ML SIGNAL PREDICTION BACKTEST")
    print("=" * 70)
    
    df = download_data()
    if df is None or len(df) < 1000:
        print("❌ Not enough data")
        return
        
    predictor = MLSignalPredictor(model_type='sklearn') # Use reliable sklearn
    
    # Train
    print("\n🎓 Training model on first 70% of data...")
    metrics = predictor.train(df, forward_period=5, threshold=0.002, train_ratio=0.7)
    
    if metrics['test_accuracy'] < 0.51:
        print("\n⚠️  Model accuracy is low (near random).")
        
    print("\n⚡ Running backtest on Out-of-Sample data (last 30%)...")
    
    # Split data same as training
    split_idx = int(len(df) * 0.7)
    test_df = df.iloc[split_idx:].copy()
    
    trades = []
    
    # Generate signals for test set
    # Note: In production, we would predict one by one. 
    # Here efficiently, we rely on the fact that features don't look ahead.
    
    i = 0
    test_len = len(test_df)
    
    while i < test_len - 10:
        # Need context for features
        # In a real loop we'd pass the full history up to i
        # But for speed in this script, we'll re-use the full df but only predict on test indices
        
        # We need to be careful not to look ahead. 
        # The predictor.train/predict logic splits carefully, but let's simulate step-by-step
        # actually, the predict method creates features from input df.
        
        current_idx = split_idx + i
        if current_idx >= len(df):
            break
            
        hist = df.iloc[:current_idx+1]
        
        # This is slow, so let's skip a bit or optimize
        # For this demo, let's just use the model which is already trained
        signal, prob = predictor.predict(hist)
        
        if signal != 0 and prob > 0.55: # Confidence threshold
            entry_price = df['Close'].iloc[current_idx]
            direction = signal
            
            # Simple exit logic for ML: hold for N bars or SL/TP
            # ML target was 5 bars forward returns
            
            trade_pnl = 0
            exit_type = 'time'
            
            sl_dist = entry_price * 0.005 # 0.5% SL
            tp_dist = entry_price * 0.010 # 1.0% TP
            
            for j in range(1, 10): # Hold max 10 bars
                if current_idx + j >= len(df):
                    break
                    
                high = df['High'].iloc[current_idx + j]
                low = df['Low'].iloc[current_idx + j]
                
                if direction == 1:
                    if low <= entry_price - sl_dist:
                        trade_pnl = ((entry_price - sl_dist) - entry_price) / entry_price
                        exit_type = 'sl'
                        break
                    if high >= entry_price + tp_dist:
                        trade_pnl = ((entry_price + tp_dist) - entry_price) / entry_price
                        exit_type = 'tp'
                        break
                else:
                    if high >= entry_price + sl_dist:
                        trade_pnl = (entry_price - (entry_price + sl_dist)) / entry_price
                        exit_type = 'sl'
                        break
                    if low <= entry_price - tp_dist:
                        trade_pnl = (entry_price - (entry_price - tp_dist)) / entry_price
                        exit_type = 'tp'
                        break
                        
            if exit_type == 'time':
                exit_price = df['Close'].iloc[min(current_idx + 10, len(df)-1)]
                if direction == 1:
                    trade_pnl = (exit_price - entry_price) / entry_price
                else:
                    trade_pnl = (entry_price - exit_price) / entry_price
            
            # Apply costs
            trade_pnl -= 0.0002 # 2bps cost
            
            trades.append({
                'pnl': trade_pnl * 100,
                'exit': exit_type,
                'direction': direction
            })
            
            i += 5 # Skip
        else:
            i += 5 # Skip if no signal 
            
        if i % 100 == 0:
            print(f"\r   Progress: {i/test_len*100:.1f}%", end="")
            
    print("\n   Done.")
    
    # Results
    if not trades:
        print("No trades generated.")
        return
        
    total_pnl = sum(t['pnl'] for t in trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total = len(trades)
    wr = wins / total
    
    print("\n" + "=" * 70)
    print("🧠 ML STRATEGY RESULTS (Out-of-Sample)")
    print("=" * 70)
    print(f"Total Return:     {total_pnl:.2f}%")
    print(f"Trades:           {total}")
    print(f"Win Rate:         {wr:.1%}")
    print("---")
    print(f"Most Important Feature: {metrics['top_features'][0][0]}")
    
    # Save results
    results = {
        'total_return': total_pnl,
        'trades': total,
        'win_rate': wr,
        'top_features': metrics['top_features']
    }
    
    with open("data/ml_backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_backtest()
