"""Backtest on REAL Market Data.

Downloads real forex/crypto data and runs comprehensive backtest.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def download_real_data(symbol: str = "EURUSD", timeframe: str = "1h", days: int = 365):
    """Download real market data using yfinance or MT5."""
    print(f"📥 Downloading {days} days of {symbol} data...")
    
    try:
        # Try yfinance first (works without MT5)
        import yfinance as yf
        
        # Map forex symbols to yfinance format
        yf_symbol = {
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X", 
            "USDJPY": "USDJPY=X",
            "BTCUSD": "BTC-USD",
            "ETHUSD": "ETH-USD"
        }.get(symbol, f"{symbol}=X")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date, end=end_date, interval="1h")
        
        if df.empty:
            # Try daily data if hourly not available
            print("   Hourly data unavailable, trying daily...")
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
        if not df.empty:
            # Rename columns to match our format
            df = df.rename(columns={
                'Open': 'Open',
                'High': 'High',
                'Low': 'Low',
                'Close': 'Close',
                'Volume': 'Volume'
            })
            print(f"   ✅ Downloaded {len(df)} bars from yfinance")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
    except ImportError:
        print("   yfinance not installed, trying alternative...")
    except Exception as e:
        print(f"   yfinance error: {e}")
        
    # Try MT5 if available
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            # Get data from MT5
            tf_map = {"1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4, "1d": mt5.TIMEFRAME_D1}
            tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, days * 24)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df = df.rename(columns={
                    'open': 'Open', 'high': 'High', 
                    'low': 'Low', 'close': 'Close',
                    'tick_volume': 'Volume'
                })
                print(f"   ✅ Downloaded {len(df)} bars from MT5")
                mt5.shutdown()
                return df[['Open', 'High', 'Low', 'Close', 'Volume']]
            mt5.shutdown()
    except Exception as e:
        print(f"   MT5 error: {e}")
        
    # Fallback: use CCXT for crypto
    try:
        import ccxt
        exchange = ccxt.binance({'enableRateLimit': True})
        
        # Download in chunks
        all_data = []
        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        while True:
            ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', since=since, limit=1000)
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            if len(all_data) >= days * 24:
                break
                
        if all_data:
            df = pd.DataFrame(all_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            print(f"   ✅ Downloaded {len(df)} bars from Binance (BTC/USDT)")
            return df
            
    except Exception as e:
        print(f"   CCXT error: {e}")
        
    return None

def run_real_backtest():
    """Run backtest on real market data."""
    print("=" * 70)
    print("🎯 REAL MARKET DATA BACKTEST")
    print("=" * 70)
    print()
    
    # Import modules
    from openquant.strategies.ensemble_strategy import EnsembleStrategy, CombineMethod
    
    # Download real data
    df = download_real_data("EURUSD", "1h", days=365)  # 1 year
    
    if df is None or len(df) < 500:
        print("❌ Failed to download sufficient data. Using longer period...")
        df = download_real_data("BTCUSD", "1h", days=180)  # 6 months crypto
        
    if df is None:
        print("❌ Could not download real data. Please install yfinance: pip install yfinance")
        return None
        
    print(f"\n📊 Data Summary:")
    print(f"   Symbol: EURUSD / BTC")
    print(f"   Period: {df.index[0]} to {df.index[-1]}")
    print(f"   Bars: {len(df)}")
    print(f"   Price Range: {df['Close'].min():.4f} - {df['Close'].max():.4f}")
    
    # Initialize strategy with optimized settings
    print("\n🔧 Initializing Strategy with AGGRESSIVE settings...")
    strategy = EnsembleStrategy(
        combine_method=CombineMethod.VOTING,
        min_agreement=0.35,  # Even lower for more trades
        probability_threshold=0.50,  # Lower threshold
        use_regime_filter=True
    )
    
    # Run backtest
    print("\n⚡ Running walk-forward backtest...")
    lookback = 200
    signals = []
    
    total_bars = len(df)
    report_interval = total_bars // 10
    
    for i in range(lookback, len(df)):
        hist = df.iloc[i-lookback:i+1].copy()
        
        try:
            sig = strategy.generate_signals(hist)
            signal_val = sig['signal'].iloc[-1] if 'signal' in sig.columns else 0
            prob = sig['probability'].iloc[-1] if 'probability' in sig.columns else 0.5
        except Exception:
            signal_val = 0
            prob = 0.5
            
        signals.append({
            'timestamp': df.index[i],
            'signal': signal_val,
            'probability': prob,
            'close': df['Close'].iloc[i]
        })
        
        if i % report_interval == 0:
            progress = (i - lookback) / (total_bars - lookback) * 100
            print(f"   Progress: {progress:.0f}%")
            
    signals_df = pd.DataFrame(signals).set_index('timestamp')
    print(f"\n✅ Generated {len(signals_df)} signals")
    
    # Calculate returns
    positions = signals_df['signal'].values
    prices = signals_df['close'].values
    returns = np.diff(prices) / prices[:-1]
    positions = positions[:-1]
    
    # Strategy returns
    strategy_returns = positions * returns * 100  # In leverage units
    
    # Apply transaction costs (5 bps per trade)
    trades = np.abs(np.diff(np.concatenate([[0], positions])))
    costs = trades * 0.0005
    strategy_returns = strategy_returns - costs * 100
    
    # Calculate metrics
    cumulative = np.cumsum(strategy_returns)
    total_return = cumulative[-1] if len(cumulative) > 0 else 0
    
    # Win rate
    wins = np.sum(strategy_returns > 0)
    losses = np.sum(strategy_returns < 0)
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    
    # Sharpe
    if np.std(strategy_returns) > 0:
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252 * 24)
    else:
        sharpe = 0
        
    # Max drawdown
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = rolling_max - cumulative
    max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Number of trades
    n_trades = np.sum(np.abs(np.diff(positions)) > 0)
    
    # Profit factor
    gross_profit = np.sum(strategy_returns[strategy_returns > 0])
    gross_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Signal distribution
    long_pct = np.sum(positions == 1) / len(positions) * 100
    short_pct = np.sum(positions == -1) / len(positions) * 100
    flat_pct = np.sum(positions == 0) / len(positions) * 100
    
    # Print results
    print("\n" + "=" * 70)
    print("📈 REAL DATA BACKTEST RESULTS")
    print("=" * 70)
    print(f"Period:            {df.index[0].date()} to {df.index[-1].date()}")
    print(f"Bars Tested:       {len(signals_df)}")
    print(f"---")
    print(f"Total Return:      {total_return:.2f}%")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print(f"Max Drawdown:      {max_dd:.2f}%")
    print(f"Win Rate:          {win_rate:.1%}")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"Number of Trades:  {n_trades}")
    print(f"---")
    print(f"Signal Distribution:")
    print(f"   Long:  {long_pct:.1f}%")
    print(f"   Short: {short_pct:.1f}%")
    print(f"   Flat:  {flat_pct:.1f}%")
    
    # Save results
    results = {
        'symbol': 'EURUSD/BTC',
        'period_start': str(df.index[0].date()),
        'period_end': str(df.index[-1].date()),
        'bars': len(signals_df),
        'total_return': float(total_return),
        'sharpe': float(sharpe),
        'max_dd': float(max_dd),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'n_trades': int(n_trades),
        'long_pct': float(long_pct),
        'short_pct': float(short_pct),
        'flat_pct': float(flat_pct),
        'timestamp': datetime.now().isoformat()
    }
    
    Path("data").mkdir(exist_ok=True)
    with open("data/real_backtest_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to data/real_backtest_results.json")
    
    # Verdict
    print("\n" + "=" * 70)
    if sharpe > 1.5 and win_rate > 0.55:
        print("🎉 EXCELLENT! Strategy is PROFITABLE on real data!")
    elif sharpe > 0.8 and win_rate > 0.50:
        print("✅ GOOD! Strategy shows promise, may need tuning")
    elif sharpe > 0.3:
        print("⚠️  MARGINAL - Strategy needs optimization")
    else:
        print("❌ POOR - Strategy needs major rework")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    run_real_backtest()
