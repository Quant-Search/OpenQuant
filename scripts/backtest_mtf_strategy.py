"""Backtest Multi-Timeframe Strategy.

Example script showing how to backtest MTF strategies with the existing
backtest engine infrastructure.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from openquant.strategies.mtf_strategy import MultiTimeframeStrategy, MultiTimeframeEnsemble
from openquant.strategies.quant.stat_arb import StatArbStrategy
from openquant.strategies.quant.kalman import KalmanMeanReversionStrategy
from openquant.data.loader import DataLoader
from openquant.backtest.engine import backtest_signals
from openquant.validation.mtf_filter import get_regime_score


class DataCache:
    """Simple cache for multi-timeframe data to avoid redundant fetches."""
    
    def __init__(self):
        self.cache = {}
        self.loader = DataLoader()
        
    def get_ohlcv(self, symbol: str, timeframe: str, days: int = 365) -> pd.DataFrame:
        """Fetch and cache OHLCV data."""
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            if symbol.startswith('BTC') or symbol.startswith('ETH'):
                df = self.loader.get_ohlcv(
                    source='ccxt:binance',
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date,
                )
            else:
                df = self.loader.get_ohlcv(
                    source='yfinance',
                    symbol=symbol,
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date,
                )
            
            self.cache[cache_key] = df
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol} on {timeframe}: {e}")
            return pd.DataFrame()


def print_backtest_metrics(result, strategy_name: str):
    """Print backtest performance metrics."""
    equity = result.equity_curve
    returns = result.returns
    
    total_return = (equity.iloc[-1] - 1.0) * 100
    
    annual_return = (equity.iloc[-1] ** (252 / len(equity)) - 1) * 100 if len(equity) > 0 else 0
    
    volatility = returns.std() * np.sqrt(252) * 100
    
    sharpe = (annual_return / volatility) if volatility > 0 else 0
    
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    winning_trades = (returns > 0).sum()
    losing_trades = (returns < 0).sum()
    total_trades = winning_trades + losing_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name}")
    print(f"{'='*60}")
    print(f"Total Return:       {total_return:>10.2f}%")
    print(f"Annual Return:      {annual_return:>10.2f}%")
    print(f"Volatility:         {volatility:>10.2f}%")
    print(f"Sharpe Ratio:       {sharpe:>10.2f}")
    print(f"Max Drawdown:       {max_drawdown:>10.2f}%")
    print(f"Total Trades:       {total_trades:>10}")
    print(f"Win Rate:           {win_rate:>10.2f}%")
    print(f"{'='*60}\n")


def backtest_simple_mtf():
    """Backtest 1: Simple MTF strategy with trend confirmation."""
    print("\n" + "="*60)
    print("Backtest 1: Simple MTF Strategy")
    print("="*60)
    
    symbol = 'BTC/USDT'
    primary_tf = '1h'
    
    cache = DataCache()
    
    base_strategy = StatArbStrategy(entry_z=2.0, exit_z=0.5, lookback=50)
    
    mtf_strategy = MultiTimeframeStrategy(
        base_strategy=base_strategy,
        timeframes=['1h', '4h', '1d'],
        fetch_func=lambda s, tf: cache.get_ohlcv(s, tf),
        require_all_timeframes=False,
        min_confirmations=1,
        use_strategy_signals=False,
    )
    mtf_strategy.set_symbol(symbol)
    
    df_1h = cache.get_ohlcv(symbol, primary_tf)
    
    if not df_1h.empty:
        base_signals = base_strategy.generate_signals(df_1h)
        mtf_signals = mtf_strategy.generate_signals(df_1h)
        
        print(f"Base Strategy Signals: {(base_signals != 0).sum()}")
        print(f"MTF Filtered Signals:  {(mtf_signals != 0).sum()}")
        print(f"Filter Rate:           {(1 - (mtf_signals != 0).sum() / max((base_signals != 0).sum(), 1)) * 100:.1f}%")
        
        base_result = backtest_signals(df_1h, base_signals, fee_bps=2.0, weight=1.0)
        mtf_result = backtest_signals(df_1h, mtf_signals, fee_bps=2.0, weight=1.0)
        
        print_backtest_metrics(base_result, "Base Strategy (No MTF)")
        print_backtest_metrics(mtf_result, "MTF Strategy")
        
        return base_result, mtf_result
    else:
        print("No data available")
        return None, None


def backtest_strict_mtf():
    """Backtest 2: Strict MTF requiring all timeframes."""
    print("\n" + "="*60)
    print("Backtest 2: Strict MTF Strategy (All TFs Required)")
    print("="*60)
    
    symbol = 'ETH/USDT'
    primary_tf = '1h'
    
    cache = DataCache()
    
    base_strategy = KalmanMeanReversionStrategy(
        process_noise=1e-5,
        measurement_noise=1e-3,
        threshold=1.5,
        use_gpu=False
    )
    
    mtf_strategy = MultiTimeframeStrategy(
        base_strategy=base_strategy,
        timeframes=['1h', '4h', '1d'],
        fetch_func=lambda s, tf: cache.get_ohlcv(s, tf),
        require_all_timeframes=True,
        min_confirmations=2,
        use_strategy_signals=False,
    )
    mtf_strategy.set_symbol(symbol)
    
    df_1h = cache.get_ohlcv(symbol, primary_tf)
    
    if not df_1h.empty:
        base_signals = base_strategy.generate_signals(df_1h)
        mtf_signals = mtf_strategy.generate_signals(df_1h)
        
        print(f"Base Strategy Signals: {(base_signals != 0).sum()}")
        print(f"MTF Filtered Signals:  {(mtf_signals != 0).sum()}")
        print(f"Filter Rate:           {(1 - (mtf_signals != 0).sum() / max((base_signals != 0).sum(), 1)) * 100:.1f}%")
        
        base_result = backtest_signals(df_1h, base_signals, fee_bps=2.0, weight=1.0)
        mtf_result = backtest_signals(df_1h, mtf_signals, fee_bps=2.0, weight=1.0)
        
        print_backtest_metrics(base_result, "Base Strategy (No MTF)")
        print_backtest_metrics(mtf_result, "Strict MTF Strategy")
        
        return base_result, mtf_result
    else:
        print("No data available")
        return None, None


def backtest_ensemble():
    """Backtest 3: Multi-timeframe ensemble."""
    print("\n" + "="*60)
    print("Backtest 3: Multi-Timeframe Ensemble")
    print("="*60)
    
    symbol = 'BTC/USDT'
    primary_tf = '1h'
    
    cache = DataCache()
    
    strategy_1h = StatArbStrategy(entry_z=2.0, exit_z=0.5, lookback=50)
    strategy_4h = KalmanMeanReversionStrategy(threshold=1.5, use_gpu=False)
    strategy_1d = StatArbStrategy(entry_z=1.5, exit_z=0.3, lookback=100)
    
    ensemble = MultiTimeframeEnsemble(
        strategies=[
            ('1h', strategy_1h, 0.5),
            ('4h', strategy_4h, 0.3),
            ('1d', strategy_1d, 0.2),
        ],
        fetch_func=lambda s, tf: cache.get_ohlcv(s, tf),
        aggregation='weighted',
        threshold=0.3,
    )
    ensemble.set_symbol(symbol)
    
    df_1h = cache.get_ohlcv(symbol, primary_tf)
    
    if not df_1h.empty:
        signals_1h = strategy_1h.generate_signals(df_1h)
        ensemble_signals = ensemble.generate_signals(df_1h)
        
        print(f"1h Strategy Signals:   {(signals_1h != 0).sum()}")
        print(f"Ensemble Signals:      {(ensemble_signals != 0).sum()}")
        
        result_1h = backtest_signals(df_1h, signals_1h, fee_bps=2.0, weight=1.0)
        result_ensemble = backtest_signals(df_1h, ensemble_signals, fee_bps=2.0, weight=1.0)
        
        print_backtest_metrics(result_1h, "Single Timeframe (1h)")
        print_backtest_metrics(result_ensemble, "MTF Ensemble")
        
        return result_1h, result_ensemble
    else:
        print("No data available")
        return None, None


def backtest_regime_weighted():
    """Backtest 4: MTF with regime-based position sizing."""
    print("\n" + "="*60)
    print("Backtest 4: MTF with Regime-Based Position Sizing")
    print("="*60)
    
    symbol = 'BTC/USDT'
    primary_tf = '1h'
    
    cache = DataCache()
    
    base_strategy = StatArbStrategy(entry_z=2.0, exit_z=0.5, lookback=50)
    
    mtf_strategy = MultiTimeframeStrategy(
        base_strategy=base_strategy,
        timeframes=['1h', '4h', '1d'],
        fetch_func=lambda s, tf: cache.get_ohlcv(s, tf),
        require_all_timeframes=False,
        min_confirmations=1,
    )
    mtf_strategy.set_symbol(symbol)
    
    df_1h = cache.get_ohlcv(symbol, primary_tf)
    
    if not df_1h.empty:
        mtf_signals = mtf_strategy.generate_signals(df_1h)
        
        regime_weighted_signals = pd.Series(0.0, index=df_1h.index)
        for idx in df_1h.index:
            if mtf_signals.loc[idx] != 0:
                score = get_regime_score(df_1h.loc[:idx], int(mtf_signals.loc[idx]))
                regime_weighted_signals.loc[idx] = mtf_signals.loc[idx] * score
        
        print(f"MTF Signals:           {(mtf_signals != 0).sum()}")
        print(f"Avg Regime Score:      {regime_weighted_signals[regime_weighted_signals != 0].abs().mean():.3f}")
        
        result_mtf = backtest_signals(df_1h, mtf_signals, fee_bps=2.0, weight=1.0)
        result_regime = backtest_signals(df_1h, regime_weighted_signals, fee_bps=2.0, weight=1.0)
        
        print_backtest_metrics(result_mtf, "MTF (Fixed Size)")
        print_backtest_metrics(result_regime, "MTF (Regime Weighted)")
        
        return result_mtf, result_regime
    else:
        print("No data available")
        return None, None


def compare_aggregation_methods():
    """Backtest 5: Compare different ensemble aggregation methods."""
    print("\n" + "="*60)
    print("Backtest 5: Ensemble Aggregation Comparison")
    print("="*60)
    
    symbol = 'ETH/USDT'
    primary_tf = '1h'
    
    cache = DataCache()
    
    strategy_1h = StatArbStrategy(entry_z=2.0, exit_z=0.5)
    strategy_4h = StatArbStrategy(entry_z=1.8, exit_z=0.4)
    strategy_1d = StatArbStrategy(entry_z=1.5, exit_z=0.3)
    
    strategies = [
        ('1h', strategy_1h, 1.0),
        ('4h', strategy_4h, 1.0),
        ('1d', strategy_1d, 1.0),
    ]
    
    df_1h = cache.get_ohlcv(symbol, primary_tf)
    
    if not df_1h.empty:
        results = {}
        
        for agg_method in ['weighted', 'majority', 'unanimous']:
            ensemble = MultiTimeframeEnsemble(
                strategies=strategies,
                fetch_func=lambda s, tf: cache.get_ohlcv(s, tf),
                aggregation=agg_method,
                threshold=0.3,
            )
            ensemble.set_symbol(symbol)
            
            signals = ensemble.generate_signals(df_1h)
            result = backtest_signals(df_1h, signals, fee_bps=2.0, weight=1.0)
            
            results[agg_method] = result
            print_backtest_metrics(result, f"Ensemble ({agg_method.capitalize()})")
        
        return results
    else:
        print("No data available")
        return None


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Multi-Timeframe Strategy Backtest Suite")
    print("="*60)
    
    try:
        backtest_simple_mtf()
    except Exception as e:
        print(f"Backtest 1 failed: {e}")
    
    try:
        backtest_strict_mtf()
    except Exception as e:
        print(f"Backtest 2 failed: {e}")
    
    try:
        backtest_ensemble()
    except Exception as e:
        print(f"Backtest 3 failed: {e}")
    
    try:
        backtest_regime_weighted()
    except Exception as e:
        print(f"Backtest 4 failed: {e}")
    
    try:
        compare_aggregation_methods()
    except Exception as e:
        print(f"Backtest 5 failed: {e}")
    
    print("\n" + "="*60)
    print("Backtest Suite Completed!")
    print("="*60)
