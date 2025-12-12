"""MR-Only Multi-Pair Backtest with Aggressive Sizing.

The ultimate MR strategy with:
- Tighter Bollinger Bands (1.8 std)
- Strict RSI filter (25/75)
- Multi-pair trading
- Aggressive position sizing (5% per trade)
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
    """Download forex data."""
    print("📥 Downloading multi-pair data...")
    data = {}
    try:
        import yfinance as yf
        for yf_sym, name in zip(PAIRS, PAIR_NAMES):
            try:
                df = yf.Ticker(yf_sym).history(
                    start=datetime.now() - timedelta(days=days),
                    end=datetime.now(),
                    interval="1h"
                )
                if not df.empty:
                    data[name] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    print(f"   ✅ {name}: {len(df)} bars")
            except Exception as e:
                print(f"   ❌ {name}: {e}")
    except ImportError:
        print("   yfinance not installed")
    return data

class OptimizedMR:
    """Optimized Mean Reversion - MR ONLY."""
    
    def __init__(
        self,
        bb_period: int = 15,
        bb_std: float = 1.8,
        rsi_period: int = 10,
        rsi_oversold: float = 25,
        rsi_overbought: float = 75,
        risk_per_trade: float = 0.05,  # 5% risk
    ):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_os = rsi_oversold
        self.rsi_ob = rsi_overbought
        self.risk = risk_per_trade
        
    def get_signal(self, df: pd.DataFrame) -> dict:
        """Get MR signal."""
        if len(df) < 30:
            return {'signal': 0}
            
        close = df['Close']
        
        # Bollinger Bands
        sma = close.rolling(self.bb_period).mean().iloc[-1]
        std = close.rolling(self.bb_period).std().iloc[-1]
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # ATR for SL
        high, low = df['High'], df['Low']
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        price = float(close.iloc[-1])
        
        # BUY at lower band with RSI confirmation
        if price <= lower and rsi < self.rsi_os:
            # WIDER SL (more room to breathe)
            sl = lower - atr * 1.5
            # TIGHTER TP (take quick profits)
            tp = sma - (sma - lower) * 0.3  # Only 70% to middle
            return {
                'signal': 1, 'sl': sl, 'tp': tp, 
                'size': self.risk, 'reason': f'MR Buy: RSI={rsi:.0f}'
            }
            
        # SELL at upper band
        elif price >= upper and rsi > self.rsi_ob:
            sl = upper + atr * 1.5
            tp = sma + (upper - sma) * 0.3
            return {
                'signal': -1, 'sl': sl, 'tp': tp,
                'size': self.risk, 'reason': f'MR Sell: RSI={rsi:.0f}'
            }
            
        return {'signal': 0}

def run_backtest():
    """Run MR-only backtest."""
    print("=" * 70)
    print("🎯 OPTIMIZED MR-ONLY MULTI-PAIR BACKTEST")
    print("=" * 70)
    
    all_data = download_pairs(365)
    if not all_data:
        return None
        
    strategy = OptimizedMR(
        bb_period=15, bb_std=1.8,
        rsi_oversold=25, rsi_overbought=75,
        risk_per_trade=0.05  # 5% per trade
    )
    
    print("\n⚡ Running backtest...")
    
    all_trades = []
    results_by_pair = {name: {'wins': 0, 'losses': 0, 'pnl': 0} for name in PAIR_NAMES}
    
    lookback = 30
    
    for pair_name, df in all_data.items():
        print(f"   {pair_name}...", end="")
        pair_trades = 0
        
        i = lookback
        while i < len(df) - 30:
            hist = df.iloc[i-lookback:i+1].copy()
            sig = strategy.get_signal(hist)
            
            if sig['signal'] == 0:
                i += 1
                continue
                
            # Simulate trade
            entry_price = df['Close'].iloc[i]
            direction = sig['signal']
            sl, tp = sig['sl'], sig['tp']
            
            trade_pnl = 0
            exit_type = 'timeout'
            
            for j in range(1, min(20, len(df) - i)):  # Quick exit for MR
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
                exit_price = df['Close'].iloc[min(i + 20, len(df) - 1)]
                if direction == 1:
                    trade_pnl = (exit_price - entry_price) / entry_price
                else:
                    trade_pnl = (entry_price - exit_price) / entry_price
                    
            # Apply leverage (5x) and size
            trade_pnl = trade_pnl * sig['size'] * 100 * 5  # 5x leverage on 5% = 25% exposure
            trade_pnl -= 0.02  # Cost
            
            if trade_pnl > 0:
                results_by_pair[pair_name]['wins'] += 1
            else:
                results_by_pair[pair_name]['losses'] += 1
            results_by_pair[pair_name]['pnl'] += trade_pnl
            
            all_trades.append({
                'pair': pair_name, 'pnl': trade_pnl, 'exit': exit_type
            })
            
            pair_trades += 1
            i += 3  # Skip ahead
            
        print(f" {pair_trades} trades")
        
    # Stats
    total_pnl = sum(t['pnl'] for t in all_trades)
    total_trades = len(all_trades)
    wins = sum(1 for t in all_trades if t['pnl'] > 0)
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    winning = [t['pnl'] for t in all_trades if t['pnl'] > 0]
    losing = [abs(t['pnl']) for t in all_trades if t['pnl'] < 0]
    pf = sum(winning) / sum(losing) if sum(losing) > 0 else float('inf')
    
    print("\n" + "=" * 70)
    print("📊 RESULTS BY PAIR")
    print("=" * 70)
    for pair, stats in results_by_pair.items():
        n = stats['wins'] + stats['losses']
        wr = stats['wins'] / n if n > 0 else 0
        print(f"{pair:10} | Trades: {n:4} | WR: {wr:5.1%} | P&L: {stats['pnl']:+8.2f}%")
        
    print("\n" + "=" * 70)
    print("🎯 COMBINED (5x LEVERAGE)")
    print("=" * 70)
    print(f"Total Return:     {total_pnl:.2f}%")
    print(f"Total Trades:     {total_trades}")
    print(f"Win Rate:         {win_rate:.1%}")
    print(f"Profit Factor:    {pf:.2f}")
    print(f"---")
    print(f"TP exits: {sum(1 for t in all_trades if t['exit'] == 'tp')}")
    print(f"SL exits: {sum(1 for t in all_trades if t['exit'] == 'sl')}")
    
    results = {
        'total_return': float(total_pnl),
        'leverage': 5,
        'total_trades': total_trades,
        'win_rate': float(win_rate),
        'profit_factor': float(pf),
        'by_pair': results_by_pair
    }
    
    Path("data").mkdir(exist_ok=True)
    with open("data/mr_optimized_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 70)
    if total_pnl > 20:
        print("🎉 EXCELLENT! High profitability achieved!")
    elif total_pnl > 10:
        print("✅ GOOD returns!")
    elif total_pnl > 0:
        print("✅ Profitable")
    else:
        print("❌ Still losing")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    run_backtest()
