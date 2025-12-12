"""Backtest on Crypto Assets (BTC, ETH).

Crypto markets are different (more volatile, 24/7).
Tests Mean Reversion and Trend Following on BTC-USD and ETH-USD.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# Crypto often respects momentum better than forex
class CryptoStrategy:
    """Hybrid Strategy for Crypto."""
    
    def __init__(self):
        self.mr_period = 20
        self.mr_std = 2.0
        self.rsi_period = 14
        self.trend_ema = 50
        
    def get_signal(self, df: pd.DataFrame) -> dict:
        if len(df) < 50:
            return {'signal': 0}
            
        close = df['Close']
        price = float(close.iloc[-1])
        
        # Indicators
        sma = close.rolling(self.mr_period).mean().iloc[-1]
        std = close.rolling(self.mr_period).std().iloc[-1]
        upper = sma + self.mr_std * std
        lower = sma - self.mr_std * std
        
        ema50 = close.ewm(span=self.trend_ema).mean().iloc[-1]
        
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Crypto Trend Following (Breakout)
        # If price > EMA50 and RSI not overbought, trend may continue
        if price > ema50 and rsi > 50 and rsi < 75:
            return {
                'signal': 1,
                'sl': price * 0.95, # 5% SL (crypto is volatile)
                'tp': price * 1.15, # 15% TP (let winners run)
                'type': 'trend'
            }
            
        # Crypto Mean Reversion (Dip Buying)
        # Strong Dips in Uptrend are buying opportunities
        if price < lower and price > ema50: # Dip in uptrend
             return {
                'signal': 1,
                'sl': price * 0.95,
                'tp': sma,
                'type': 'dip'
            }

        return {'signal': 0}

def download_crypto():
    data = {}
    try:
        import yfinance as yf
        print("📥 Downloading Crypto Data...")
        pairs = ["BTC-USD", "ETH-USD"]
        for sym in pairs:
            try:
                # Crypto constraints are looser on yfinance usually
                df = yf.Ticker(sym).history(period="1y", interval="1h")
                if not df.empty:
                    data[sym] = df
                    print(f"   ✅ {sym}: {len(df)} bars")
            except Exception as e:
                print(f"   ❌ {sym}: {e}")
    except: pass
    return data

def run_backtest():
    print("=" * 70)
    print("₿ CRYPTO BACKTEST (BTC, ETH)")
    print("=" * 70)
    
    all_data = download_crypto()
    if not all_data:
        print("❌ No data")
        return
        
    strategy = CryptoStrategy()
    all_trades = []
    
    print("\n⚡ Running...")
    
    for pair, df in all_data.items():
        i = 50
        while i < len(df) - 50:
            hist = df.iloc[i-50:i+1].copy()
            sig = strategy.get_signal(hist)
            
            if sig['signal'] == 1: # Only Longs for Crypto Spot usually
                entry = df['Close'].iloc[i]
                sl = sig['sl']
                tp = sig['tp']
                
                # Simulate
                trade_pnl = 0
                exit_type = 'timeout'
                
                # Hold longer for crypto trend
                for j in range(1, min(100, len(df)-i)):
                    idx = i + j
                    low = df['Low'].iloc[idx]
                    high = df['High'].iloc[idx]
                    
                    if low <= sl:
                        trade_pnl = (sl - entry) / entry
                        exit_type = 'sl'
                        break
                    if high >= tp:
                        trade_pnl = (tp - entry) / entry
                        exit_type = 'tp'
                        break
                        
                if exit_type == 'timeout':
                    exit_price = df['Close'].iloc[min(i+100, len(df)-1)]
                    trade_pnl = (exit_price - entry) / entry
                    
                # No leverage, spot trading costs 0.1% usually
                net_pnl = trade_pnl * 100 - 0.2 
                
                all_trades.append({
                    'pair': pair,
                    'pnl': net_pnl,
                    'type': sig['type'],
                    'exit': exit_type
                })
                
                i += 10 # spacing
            else:
                i += 1
                
    # Stats
    if not all_trades:
        print("No trades.")
        return

    total_pnl = sum(t['pnl'] for t in all_trades)
    total = len(all_trades)
    wins = sum(1 for t in all_trades if t['pnl'] > 0)
    wr = wins / total
    
    winning = [t['pnl'] for t in all_trades if t['pnl'] > 0]
    losing = [abs(t['pnl']) for t in all_trades if t['pnl'] < 0]
    pf = sum(winning)/sum(losing) if sum(losing) > 0 else float('inf')
    
    print("\n" + "=" * 70)
    print("₿ CRYPTO RESULTS")
    print("=" * 70)
    print(f"Total Return:     {total_pnl:.2f}%")
    print(f"Total Trades:     {total}")
    print(f"Win Rate:         {wr:.1%}")
    print(f"Profit Factor:    {pf:.2f}")
    
    Path("data").mkdir(exist_ok=True)
    with open("data/crypto_results.json", "w") as f:
        json.dump({'return': total_pnl, 'metrics': {'wr': wr, 'pf': pf}}, f)
        
    if total_pnl > 0:
        print("✅ Crypto Profitable")
    else:
        print("❌ Crypto Loss")
    print("=" * 70)

if __name__ == "__main__":
    run_backtest()
