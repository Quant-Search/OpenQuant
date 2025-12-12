"""Example: Multi-Timeframe Strategy Usage.

Demonstrates how to use the MultiTimeframeStrategy and MultiTimeframeEnsemble
with signal alignment across multiple timeframes.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from openquant.strategies.mtf_strategy import MultiTimeframeStrategy, MultiTimeframeEnsemble
from openquant.strategies.quant.stat_arb import StatArbStrategy
from openquant.strategies.quant.kalman import KalmanMeanReversionStrategy
from openquant.data.loader import DataLoader


def example_fetch_function(symbol: str, timeframe: str) -> pd.DataFrame:
    """Example fetch function using DataLoader.
    
    In production, this would be connected to your broker's data feed.
    """
    loader = DataLoader()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    try:
        if symbol.startswith('BTC') or symbol.startswith('ETH'):
            df = loader.get_ohlcv(
                source='ccxt:binance',
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
            )
        else:
            df = loader.get_ohlcv(
                source='yfinance',
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
            )
        return df
    except Exception as e:
        print(f"Error fetching {symbol} on {timeframe}: {e}")
        return pd.DataFrame()


def example_simple_mtf_strategy():
    """Example 1: Simple multi-timeframe strategy with a base strategy."""
    print("=" * 60)
    print("Example 1: Simple Multi-Timeframe Strategy")
    print("=" * 60)
    
    base_strategy = StatArbStrategy(entry_z=2.0, exit_z=0.5, lookback=50)
    
    mtf_strategy = MultiTimeframeStrategy(
        base_strategy=base_strategy,
        timeframes=['1h', '4h', '1d'],
        fetch_func=example_fetch_function,
        require_all_timeframes=False,
        min_confirmations=1,
        use_strategy_signals=False,
    )
    
    mtf_strategy.set_symbol('BTC/USDT')
    
    df_1h = example_fetch_function('BTC/USDT', '1h')
    
    if not df_1h.empty:
        signals = mtf_strategy.generate_signals(df_1h)
        
        print(f"\nGenerated {len(signals)} signals")
        print(f"Long signals: {(signals == 1).sum()}")
        print(f"Short signals: {(signals == -1).sum()}")
        print(f"Flat signals: {(signals == 0).sum()}")
        print(f"\nLast 10 signals:\n{signals.tail(10)}")
    else:
        print("No data available")


def example_strict_mtf_strategy():
    """Example 2: Strict multi-timeframe strategy requiring all timeframes."""
    print("\n" + "=" * 60)
    print("Example 2: Strict Multi-Timeframe Strategy (All TFs Required)")
    print("=" * 60)
    
    base_strategy = KalmanMeanReversionStrategy(
        process_noise=1e-5,
        measurement_noise=1e-3,
        threshold=1.5,
        use_gpu=False
    )
    
    mtf_strategy = MultiTimeframeStrategy(
        base_strategy=base_strategy,
        timeframes=['1h', '4h', '1d'],
        fetch_func=example_fetch_function,
        require_all_timeframes=True,
        min_confirmations=2,
        use_strategy_signals=False,
    )
    
    mtf_strategy.set_symbol('ETH/USDT')
    
    df_1h = example_fetch_function('ETH/USDT', '1h')
    
    if not df_1h.empty:
        signals = mtf_strategy.generate_signals(df_1h)
        
        print(f"\nGenerated {len(signals)} signals")
        print(f"Long signals: {(signals == 1).sum()}")
        print(f"Short signals: {(signals == -1).sum()}")
        print(f"Flat signals: {(signals == 0).sum()}")
        
        if (signals != 0).any():
            print(f"\nFirst non-zero signal: {signals[signals != 0].index[0]}")
            print(f"Last non-zero signal: {signals[signals != 0].index[-1]}")
    else:
        print("No data available")


def example_mtf_ensemble():
    """Example 3: Multi-timeframe ensemble with different strategies per timeframe."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Timeframe Ensemble Strategy")
    print("=" * 60)
    
    strategy_1h = StatArbStrategy(entry_z=2.0, exit_z=0.5, lookback=50)
    strategy_4h = KalmanMeanReversionStrategy(threshold=1.5, use_gpu=False)
    strategy_1d = StatArbStrategy(entry_z=1.5, exit_z=0.3, lookback=100)
    
    ensemble = MultiTimeframeEnsemble(
        strategies=[
            ('1h', strategy_1h, 0.5),
            ('4h', strategy_4h, 0.3),
            ('1d', strategy_1d, 0.2),
        ],
        fetch_func=example_fetch_function,
        aggregation='weighted',
        threshold=0.3,
    )
    
    ensemble.set_symbol('BTC/USDT')
    
    df_1h = example_fetch_function('BTC/USDT', '1h')
    
    if not df_1h.empty:
        signals = ensemble.generate_signals(df_1h)
        
        print(f"\nGenerated {len(signals)} signals")
        print(f"Long signals: {(signals == 1).sum()}")
        print(f"Short signals: {(signals == -1).sum()}")
        print(f"Flat signals: {(signals == 0).sum()}")
        
        print(f"\nSignal distribution:")
        print(signals.value_counts().sort_index())
    else:
        print("No data available")


def example_majority_ensemble():
    """Example 4: Ensemble with majority voting."""
    print("\n" + "=" * 60)
    print("Example 4: Multi-Timeframe Ensemble with Majority Voting")
    print("=" * 60)
    
    strategy_1h = StatArbStrategy(entry_z=2.0, exit_z=0.5)
    strategy_4h = StatArbStrategy(entry_z=1.8, exit_z=0.4)
    strategy_1d = StatArbStrategy(entry_z=1.5, exit_z=0.3)
    
    ensemble = MultiTimeframeEnsemble(
        strategies=[
            ('1h', strategy_1h, 1.0),
            ('4h', strategy_4h, 1.0),
            ('1d', strategy_1d, 1.0),
        ],
        fetch_func=example_fetch_function,
        aggregation='majority',
    )
    
    ensemble.set_symbol('ETH/USDT')
    
    df_1h = example_fetch_function('ETH/USDT', '1h')
    
    if not df_1h.empty:
        signals = ensemble.generate_signals(df_1h)
        
        print(f"\nGenerated {len(signals)} signals")
        print(f"Long signals: {(signals == 1).sum()}")
        print(f"Short signals: {(signals == -1).sum()}")
        print(f"Flat signals: {(signals == 0).sum()}")
    else:
        print("No data available")


def example_strategy_based_confirmation():
    """Example 5: MTF strategy using strategy signals for confirmation."""
    print("\n" + "=" * 60)
    print("Example 5: MTF Strategy with Strategy-Based Confirmation")
    print("=" * 60)
    
    base_strategy = KalmanMeanReversionStrategy(threshold=1.2, use_gpu=False)
    
    mtf_strategy = MultiTimeframeStrategy(
        base_strategy=base_strategy,
        timeframes=['1h', '4h', '1d'],
        fetch_func=example_fetch_function,
        require_all_timeframes=False,
        min_confirmations=1,
        use_strategy_signals=True,
    )
    
    mtf_strategy.set_symbol('BTC/USDT')
    
    df_1h = example_fetch_function('BTC/USDT', '1h')
    
    if not df_1h.empty:
        signals = mtf_strategy.generate_signals(df_1h)
        
        print(f"\nGenerated {len(signals)} signals")
        print(f"Long signals: {(signals == 1).sum()}")
        print(f"Short signals: {(signals == -1).sum()}")
        print(f"Flat signals: {(signals == 0).sum()}")
        
        base_signals = base_strategy.generate_signals(df_1h)
        print(f"\nBase strategy signals: {(base_signals != 0).sum()}")
        print(f"MTF filtered signals: {(signals != 0).sum()}")
        print(f"Filter reduction: {((base_signals != 0).sum() - (signals != 0).sum()) / max((base_signals != 0).sum(), 1) * 100:.1f}%")
    else:
        print("No data available")


if __name__ == '__main__':
    try:
        example_simple_mtf_strategy()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_strict_mtf_strategy()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_mtf_ensemble()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_majority_ensemble()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    try:
        example_strategy_based_confirmation()
    except Exception as e:
        print(f"Example 5 failed: {e}")
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
