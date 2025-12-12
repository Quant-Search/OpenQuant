"""Optimized Crypto Strategy.

Crypto markets trend strongly.
Strategy:
1. Trend Following on 4H equivalent (using 1H data)
2. Volatility breakout
3. Trailing stop to catch big runs (100%+)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class CryptoTrend:
    def get_signal(self, df: pd.DataFrame) -> dict:
        if len(df) < 100: return {'signal': 0}
        
        close = df['Close']
        high = df['High'] 
        low = df['Low']
        
        # Donchian Channel (Breakout) - 50 period (approx 2 days)
        dc_high = high.rolling(50).max().shift(1).iloc[-1]
        dc_low = low.rolling(50).min().shift(1).iloc[-1]
        
        price = float(close.iloc[-1])
        
        # Trend Filter (EMA 200)
        ema200 = close.ewm(span=200).mean().iloc[-1]
        
        # Volatility Filter (ATR)
        tr = pd.concat([high-low, abs(high-close.shift(1)), abs(low-close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(24).mean().iloc[-1]
        
        # Buy Breakout in Uptrend
        if price > dc_high and price > ema200:
            return {
                'signal': 1,
                'sl': price - 2 * atr, # Initial SL
                'trail': 2 * atr # Trailing distance
            }
            
        return {'signal': 0}

def run_backtest():
    print("=" * 70)
    print("₿ CRYPTO TREND STRATEGY")
    print("=" * 70)
    
    # Download
    import yfinance as yf
    data = {}
    for sym in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        try:
            df = yf.Ticker(sym).history(period="1y", interval="1h")
            if not df.empty: data[sym] = df
        except: pass
        
    strategy = CryptoTrend()
    trades = []
    
    print("⚡ Running...")
    
    for pair, df in data.items():
        i = 100
        while i < len(df) - 50:
            hist = df.iloc[i-100:i+1]
            sig = strategy.get_signal(hist)
            
            if sig['signal'] == 1:
                entry = df['Close'].iloc[i]
                sl = sig['sl']
                trail_dist = sig['trail']
                
                # Trail stop logic
                curr_sl = sl
                best_price = entry
                exit_price = entry
                
                for j in range(1, len(df)-i):
                    curr = df.iloc[i+j]
                    
                    # Update trailing stop
                    if curr['High'] > best_price:
                        best_price = curr['High']
                        new_sl = best_price - trail_dist
                        if new_sl > curr_sl: curr_sl = new_sl
                        
                    # Check exit
                    if curr['Low'] < curr_sl:
                        exit_price = curr_sl
                        break
                        
                pnl = (exit_price - entry) / entry * 100
                pnl -= 0.1 # fees
                
                trades.append({'pair': pair, 'pnl': pnl})
                
                # Skip forward
                i += j
            else:
                i += 1
                
    if not trades:
         print("No trades")
         return
         
    total_pnl = sum(t['pnl'] for t in trades)
    print(f"\nTotal Return: {total_pnl:.2f}%")
    print(f"Trades: {len(trades)}")
    
    if trades:
        wins = [t['pnl'] for t in trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in trades if t['pnl'] < 0]
        pf = sum(wins)/sum(losses) if losses else 0
        print(f"Profit Factor: {pf:.2f}")

if __name__ == "__main__":
    run_backtest()
