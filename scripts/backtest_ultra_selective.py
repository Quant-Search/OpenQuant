"""Ultra-Selective MR Strategy.

ONLY takes the highest quality setups:
1. RSI must be at EXTREME levels (<20 or >80)
2. Price must be 1.5x std beyond bands (extreme)
3. Previous bar must show rejection (wick pattern)
4. Use tight SL and let winners run
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

def download_pairs(days: int = 365):
    data = {}
    try:
        import yfinance as yf
        for yf_sym, name in zip(PAIRS, PAIR_NAMES):
            try:
                df = yf.Ticker(yf_sym).history(
                    start=datetime.now() - timedelta(days=days),
                    end=datetime.now(), interval="1h"
                )
                if not df.empty:
                    data[name] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    print(f"   ✅ {name}: {len(df)} bars")
            except: pass
    except: pass
    return data

class UltraSelectiveMR:
    """Ultra-selective MR - ONLY highest quality setups."""
    
    def has_rejection_wick(self, df: pd.DataFrame, direction: int) -> bool:
        """Check for rejection wick (hammer/shooting star)."""
        last = df.iloc[-1]
        body = abs(last['Close'] - last['Open'])
        total = last['High'] - last['Low']
        if total == 0:
            return False
        
        if direction == 1:  # Looking for bullish rejection
            lower_wick = min(last['Open'], last['Close']) - last['Low']
            return lower_wick / total >= 0.5  # Strong lower wick
        else:  # Bearish rejection
            upper_wick = last['High'] - max(last['Open'], last['Close'])
            return upper_wick / total >= 0.5
            
    def get_signal(self, df: pd.DataFrame) -> dict:
        if len(df) < 30:
            return {'signal': 0}
            
        close = df['Close']
        
        # Bollinger Bands (tight)
        sma = close.rolling(20).mean().iloc[-1]
        std = close.rolling(20).std().iloc[-1]
        upper = sma + 2.5 * std  # EXTREME bands
        lower = sma - 2.5 * std
        
        # RSI (EXTREME levels)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # ATR
        high, low = df['High'], df['Low']
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        price = float(close.iloc[-1])
        
        # ULTRA STRICT CONDITIONS
        # BUY: Extreme oversold + rejection wick
        if price < lower and rsi < 20 and self.has_rejection_wick(df, 1):
            sl = price - atr * 0.8  # TIGHT SL
            tp = sma  # Let it run to middle
            return {
                'signal': 1, 'sl': sl, 'tp': tp,
                'reason': f'ULTRA Buy: RSI={rsi:.0f}, extreme'
            }
            
        # SELL: Extreme overbought + rejection wick
        elif price > upper and rsi > 80 and self.has_rejection_wick(df, -1):
            sl = price + atr * 0.8
            tp = sma
            return {
                'signal': -1, 'sl': sl, 'tp': tp,
                'reason': f'ULTRA Sell: RSI={rsi:.0f}, extreme'
            }
            
        return {'signal': 0}

def run_backtest():
    print("=" * 70)
    print("🎯 ULTRA-SELECTIVE MR BACKTEST")
    print("=" * 70)
    print("📥 Downloading...")
    
    all_data = download_pairs(365)
    if not all_data:
        return None
        
    strategy = UltraSelectiveMR()
    
    print("\n⚡ Running...")
    
    all_trades = []
    
    for pair_name, df in all_data.items():
        lookback = 30
        i = lookback
        
        while i < len(df) - 30:
            hist = df.iloc[i-lookback:i+1].copy()
            sig = strategy.get_signal(hist)
            
            if sig['signal'] == 0:
                i += 1
                continue
                
            entry_price = df['Close'].iloc[i]
            direction = sig['signal']
            sl, tp = sig['sl'], sig['tp']
            
            trade_pnl = 0
            exit_type = 'timeout'
            
            # Let it run longer (30 bars)
            for j in range(1, min(40, len(df) - i)):
                idx = i + j
                high, low = df['High'].iloc[idx], df['Low'].iloc[idx]
                
                if direction == 1:
                    if low <= sl:
                        trade_pnl = (sl - entry_price) / entry_price
                        exit_type = 'sl'
                        break
                    if high >= tp:
                        trade_pnl = (tp - entry_price) / entry_price
                        exit_type = 'tp'
                        break
                else:
                    if high >= sl:
                        trade_pnl = (entry_price - sl) / entry_price
                        exit_type = 'sl'
                        break
                    if low <= tp:
                        trade_pnl = (entry_price - tp) / entry_price
                        exit_type = 'tp'
                        break
                        
            if exit_type == 'timeout':
                exit_price = df['Close'].iloc[min(i + 40, len(df) - 1)]
                if direction == 1:
                    trade_pnl = (exit_price - entry_price) / entry_price
                else:
                    trade_pnl = (entry_price - exit_price) / entry_price
                    
            # Apply 5x leverage on 5% = 25% exposure per trade
            trade_pnl = trade_pnl * 0.25 * 100
            trade_pnl -= 0.02  # Cost
            
            all_trades.append({
                'pair': pair_name, 'pnl': trade_pnl, 'exit': exit_type
            })
            
            i += 5  # Skip ahead
            
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
    print("🎯 ULTRA-SELECTIVE RESULTS (5x leverage)")
    print("=" * 70)
    print(f"Total Return:     {total_pnl:.2f}%")
    print(f"Total Trades:     {total}")
    print(f"Win Rate:         {wr:.1%}")
    print(f"Avg Win:          {avg_win:.2f}%")
    print(f"Avg Loss:         {avg_loss:.2f}%")
    print(f"Profit Factor:    {pf:.2f}")
    print(f"---")
    print(f"TP exits: {sum(1 for t in all_trades if t['exit'] == 'tp')}")
    print(f"SL exits: {sum(1 for t in all_trades if t['exit'] == 'sl')}")
    print(f"Timeout:  {sum(1 for t in all_trades if t['exit'] == 'timeout')}")
    
    # Save
    Path("data").mkdir(exist_ok=True)
    with open("data/ultra_selective_results.json", "w") as f:
        json.dump({'return': total_pnl, 'trades': total, 'wr': wr, 'pf': pf}, f)
        
    print("\n" + "=" * 70)
    if total_pnl > 10:
        print("🎉 EXCELLENT!")
    elif total_pnl > 0:
        print("✅ Profitable!")
    else:
        print("❌ Still losing")
    print("=" * 70)
    
    return {'return': total_pnl, 'trades': total}

if __name__ == "__main__":
    run_backtest()
