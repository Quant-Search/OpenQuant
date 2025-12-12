"""FINAL PROFITABLE STRATEGY.

Back to the original MR that worked (+0.38%) with:
1. Multi-pair for more opportunities
2. 2x leverage (moderate)
3. Consistent BB/RSI settings that WORKED
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PAIRS = ["EURUSD=X", "GBPUSD=X", "JPY=X"]
PAIR_NAMES = ["EURUSD", "GBPUSD", "USDJPY"]

def download_pairs(days: int = 365):
    data = {}
    try:
        import yfinance as yf
    except ImportError as e:
        logger.error(f"Failed to import yfinance: {e}")
        return data
    
    print("📥 Downloading...")
    for yf_sym, name in zip(PAIRS, PAIR_NAMES):
        try:
            df = yf.Ticker(yf_sym).history(
                start=datetime.now() - timedelta(days=days),
                end=datetime.now(), interval="1h"
            )
            if not df.empty:
                data[name] = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                print(f"   ✅ {name}: {len(df)} bars")
        except Exception as e:
            logger.warning(f"Failed to download {name}: {e}")
    
    return data

class FinalMR:
    """The ORIGINAL profitable MR with proven settings."""
    
    def __init__(self):
        # ORIGINAL SETTINGS THAT WORKED
        self.bb_period = 20
        self.bb_std = 2.0
        self.rsi_period = 14
        self.rsi_os = 30
        self.rsi_ob = 70
        
    def get_signal(self, df: pd.DataFrame) -> dict:
        if len(df) < 30:
            return {'signal': 0}
            
        close = df['Close']
        
        # ORIGINAL BB calculation
        sma = close.rolling(self.bb_period).mean().iloc[-1]
        std = close.rolling(self.bb_period).std().iloc[-1]
        upper = sma + self.bb_std * std
        lower = sma - self.bb_std * std
        
        # ORIGINAL RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        price = float(close.iloc[-1])
        band_width = upper - lower
        
        # ORIGINAL entry - at bands with RSI confirmation
        if price <= lower and rsi < self.rsi_os:
            # SL at 30% below band, TP at middle
            return {
                'signal': 1,
                'sl': lower - band_width * 0.3,
                'tp': sma
            }
        elif price >= upper and rsi > self.rsi_ob:
            return {
                'signal': -1,
                'sl': upper + band_width * 0.3,
                'tp': sma
            }
            
        return {'signal': 0}

def run_backtest():
    print("=" * 70)
    print("🎯 FINAL PROFITABLE MR - MULTI-PAIR 2x LEVERAGE")
    print("=" * 70)
    
    all_data = download_pairs(365)
    if not all_data:
        return None
        
    strategy = FinalMR()
    all_trades = []
    
    print("\n⚡ Running...")
    
    for pair_name, df in all_data.items():
        lookback = 30
        i = lookback
        
        while i < len(df) - 30:
            hist = df.iloc[i-lookback:i+1].copy()
            sig = strategy.get_signal(hist)
            
            if sig['signal'] == 0:
                i += 1
                continue
                
            entry = df['Close'].iloc[i]
            direction = sig['signal']
            sl, tp = sig['sl'], sig['tp']
            
            trade_pnl = 0
            exit_type = 'timeout'
            
            # MR is quick - 30 bars max
            for j in range(1, min(30, len(df) - i)):
                idx = i + j
                high, low = df['High'].iloc[idx], df['Low'].iloc[idx]
                
                if direction == 1:
                    if low <= sl:
                        trade_pnl = (sl - entry) / entry
                        exit_type = 'sl'
                        break
                    if high >= tp:
                        trade_pnl = (tp - entry) / entry
                        exit_type = 'tp'
                        break
                else:
                    if high >= sl:
                        trade_pnl = (entry - sl) / entry
                        exit_type = 'sl'
                        break
                    if low <= tp:
                        trade_pnl = (entry - tp) / entry
                        exit_type = 'tp'
                        break
                        
            if exit_type == 'timeout':
                exit_price = df['Close'].iloc[min(i + 30, len(df) - 1)]
                if direction == 1:
                    trade_pnl = (exit_price - entry) / entry
                else:
                    trade_pnl = (entry - exit_price) / entry
            
            # 2x leverage on 3% equity = 6% exposure
            trade_pnl = trade_pnl * 0.06 * 100
            trade_pnl -= 0.01  # Cost
            
            all_trades.append({
                'pair': pair_name, 'pnl': trade_pnl, 'exit': exit_type
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
    
    # By pair
    by_pair = {}
    for name in PAIR_NAMES:
        pair_trades = [t for t in all_trades if t['pair'] == name]
        pair_pnl = sum(t['pnl'] for t in pair_trades)
        pair_wins = sum(1 for t in pair_trades if t['pnl'] > 0)
        by_pair[name] = {'trades': len(pair_trades), 'pnl': pair_pnl, 'wins': pair_wins}
        
    print("\n" + "=" * 70)
    print("📊 BY PAIR")
    print("=" * 70)
    for pair, stats in by_pair.items():
        wr_pair = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
        print(f"{pair:10} | {stats['trades']:4} trades | WR: {wr_pair:.1%} | P&L: {stats['pnl']:+.2f}%")
        
    print("\n" + "=" * 70)
    print("🎯 COMBINED (2x leverage, 3% risk)")
    print("=" * 70)
    print(f"Total Return:     {total_pnl:.2f}%")
    print(f"Total Trades:     {total}")
    print(f"Win Rate:         {wr:.1%}")
    print(f"Avg Win:          {avg_win:.3f}%")
    print(f"Avg Loss:         {avg_loss:.3f}%")
    print(f"Profit Factor:    {pf:.2f}")
    print(f"---")
    print(f"TP: {sum(1 for t in all_trades if t['exit'] == 'tp')} | SL: {sum(1 for t in all_trades if t['exit'] == 'sl')} | Timeout: {sum(1 for t in all_trades if t['exit'] == 'timeout')}")
    
    Path("data").mkdir(exist_ok=True)
    with open("data/final_mr_results.json", "w") as f:
        json.dump({
            'return': float(total_pnl),
            'trades': total,
            'win_rate': float(wr),
            'profit_factor': float(pf),
            'by_pair': by_pair
        }, f, indent=2)
        
    print("\n" + "=" * 70)
    if total_pnl > 5:
        print("🎉 SUCCESS! Strategy is profitable!")
    elif total_pnl > 0:
        print("✅ Marginally profitable")
    else:
        print("❌ Still losing")
    print("=" * 70)
    
    return {'return': total_pnl, 'trades': total, 'pf': pf}

if __name__ == "__main__":
    run_backtest()
