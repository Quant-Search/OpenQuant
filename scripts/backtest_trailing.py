"""MR with Trailing Stop.

The key insight: MR entries are good but exits are bad.
Solution: Use a TRAILING STOP to let winners run and cut losers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

PAIRS = ["EURUSD=X", "GBPUSD=X", "JPY=X"]
PAIR_NAMES = ["EURUSD", "GBPUSD", "USDJPY"]

def download_pairs():
    data = {}
    try:
        import yfinance as yf
        print("📥 Downloading...")
        for yf_sym, name in zip(PAIRS, PAIR_NAMES):
            try:
                df = yf.Ticker(yf_sym).history(
                    start=datetime.now() - timedelta(days=365),
                    end=datetime.now(), interval="1h"
                )
                if not df.empty:
                    data[name] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    print(f"   ✅ {name}: {len(df)} bars")
            except: pass
    except: pass
    return data

class TrailingMR:
    """MR with trailing stop for better exits."""
    
    def get_signal(self, df: pd.DataFrame) -> dict:
        if len(df) < 25:
            return {'signal': 0}
            
        close = df['Close']
        
        # BB
        sma = close.rolling(20).mean().iloc[-1]
        std = close.rolling(20).std().iloc[-1]
        upper = sma + 2.0 * std
        lower = sma - 2.0 * std
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # ATR for trailing
        high, low = df['High'], df['Low']
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        price = float(close.iloc[-1])
        
        if price <= lower and rsi < 30:
            return {
                'signal': 1,
                'initial_sl': lower - atr * 2,  # WIDE initial SL
                'trail_dist': atr * 1.5,  # Trailing distance
                'target': sma
            }
        elif price >= upper and rsi > 70:
            return {
                'signal': -1,
                'initial_sl': upper + atr * 2,
                'trail_dist': atr * 1.5,
                'target': sma
            }
            
        return {'signal': 0}

def simulate_trade_trailing(df: pd.DataFrame, entry_idx: int, direction: int,
                            initial_sl: float, trail_dist: float, target: float,
                            max_hold: int = 40) -> dict:
    """Simulate trade with trailing stop."""
    entry = df['Close'].iloc[entry_idx]
    current_sl = initial_sl
    best_price = entry
    
    for j in range(1, min(max_hold, len(df) - entry_idx)):
        idx = entry_idx + j
        high, low = df['High'].iloc[idx], df['Low'].iloc[idx]
        
        if direction == 1:  # Long
            # Update best price and trail stop
            if high > best_price:
                best_price = high
                new_sl = high - trail_dist
                if new_sl > current_sl:
                    current_sl = new_sl
                    
            # Check SL
            if low <= current_sl:
                pnl = (current_sl - entry) / entry
                return {'pnl': pnl, 'exit': 'trail' if current_sl > initial_sl else 'sl'}
                
            # Check target
            if high >= target:
                pnl = (target - entry) / entry
                return {'pnl': pnl, 'exit': 'target'}
        else:  # Short
            if low < best_price:
                best_price = low
                new_sl = low + trail_dist
                if new_sl < current_sl:
                    current_sl = new_sl
                    
            if high >= current_sl:
                pnl = (entry - current_sl) / entry
                return {'pnl': pnl, 'exit': 'trail' if current_sl < initial_sl else 'sl'}
                
            if low <= target:
                pnl = (entry - target) / entry
                return {'pnl': pnl, 'exit': 'target'}
                
    # Timeout
    exit_price = df['Close'].iloc[min(entry_idx + max_hold, len(df) - 1)]
    if direction == 1:
        pnl = (exit_price - entry) / entry
    else:
        pnl = (entry - exit_price) / entry
    return {'pnl': pnl, 'exit': 'timeout'}

def run_backtest():
    print("=" * 70)
    print("🎯 MR WITH TRAILING STOP")
    print("=" * 70)
    
    all_data = download_pairs()
    if not all_data:
        return None
        
    strategy = TrailingMR()
    all_trades = []
    
    print("\n⚡ Running...")
    
    for pair_name, df in all_data.items():
        lookback = 25
        i = lookback
        
        while i < len(df) - 50:
            hist = df.iloc[i-lookback:i+1].copy()
            sig = strategy.get_signal(hist)
            
            if sig['signal'] == 0:
                i += 1
                continue
                
            result = simulate_trade_trailing(
                df, i, sig['signal'],
                sig['initial_sl'], sig['trail_dist'], sig['target'],
                max_hold=40
            )
            
            # Apply 2x leverage on 5% = 10% exposure
            pnl = result['pnl'] * 0.10 * 100
            pnl -= 0.01  # Cost
            
            all_trades.append({
                'pair': pair_name, 'pnl': pnl, 'exit': result['exit']
            })
            
            i += 5
            
    # Stats
    total_pnl = sum(t['pnl'] for t in all_trades)
    total = len(all_trades)
    wins = sum(1 for t in all_trades if t['pnl'] > 0)
    wr = wins / total if total > 0 else 0
    
    winning = [t['pnl'] for t in all_trades if t['pnl'] > 0]
    losing = [abs(t['pnl']) for t in all_trades if t['pnl'] < 0]
    pf = sum(winning) / sum(losing) if sum(losing) > 0 else float('inf')
    
    avg_win = np.mean(winning) if winning else 0
    avg_loss = np.mean(losing) if losing else 0
    
    print("\n" + "=" * 70)
    print("🎯 TRAILING STOP RESULTS")
    print("=" * 70)
    print(f"Total Return:     {total_pnl:.2f}%")
    print(f"Trades:           {total}")
    print(f"Win Rate:         {wr:.1%}")
    print(f"Avg Win:          {avg_win:.3f}%")
    print(f"Avg Loss:         {avg_loss:.3f}%")
    print(f"Profit Factor:    {pf:.2f}")
    print(f"---")
    print(f"Target: {sum(1 for t in all_trades if t['exit'] == 'target')}")
    print(f"Trail:  {sum(1 for t in all_trades if t['exit'] == 'trail')}")
    print(f"SL:     {sum(1 for t in all_trades if t['exit'] == 'sl')}")
    print(f"Timeout:{sum(1 for t in all_trades if t['exit'] == 'timeout')}")
    
    Path("data").mkdir(exist_ok=True)
    with open("data/trailing_results.json", "w") as f:
        json.dump({'return': total_pnl, 'trades': total, 'wr': wr, 'pf': pf}, f)
        
    print("\n" + "=" * 70)
    if total_pnl > 5:
        print("🎉 PROFITABLE!")
    elif total_pnl > 0:
        print("✅ Marginally profitable")
    else:
        print("❌ Still losing")
    print("=" * 70)
    
    return {'return': total_pnl}

if __name__ == "__main__":
    run_backtest()
