"""Example script for using the Regime Adaptive Strategy.

Demonstrates:
1. Creating and using RegimeAdaptiveStrategy
2. Running regime-specific WFO backtesting
3. Comparing strategies by regime
4. Advanced usage with custom parameters
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from openquant.strategies.regime_adaptive import RegimeAdaptiveStrategy
from openquant.strategies.quant.hurst import HurstExponentStrategy
from openquant.strategies.quant.stat_arb import StatArbStrategy
from openquant.evaluation.wfo import (
    walk_forward_evaluate,
    walk_forward_evaluate_regime_specific,
    compare_strategies_by_regime,
    WFOSpec
)
from openquant.backtest.engine import backtest_signals
from openquant.backtest.metrics import sharpe


def generate_synthetic_data(n_points: int = 1000, regime_type: str = 'mixed') -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with different regime periods.
    
    Args:
        n_points: Number of data points to generate
        regime_type: Type of regime - 'trending', 'ranging', 'volatile', or 'mixed'
    """
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='1H')
    
    price = 100.0
    prices = []
    
    for i in range(n_points):
        if regime_type == 'mixed':
            if i < n_points // 3:
                drift = 0.0002
                volatility = 0.005
            elif i < 2 * n_points // 3:
                drift = 0.001
                volatility = 0.003
            else:
                drift = 0.0
                volatility = 0.01
        elif regime_type == 'trending':
            drift = 0.001
            volatility = 0.003
        elif regime_type == 'ranging':
            drift = 0.0
            volatility = 0.005
        else:
            drift = 0.0
            volatility = 0.015
        
        ret = drift + volatility * np.random.randn()
        price *= (1 + ret)
        prices.append(price)
    
    close_prices = np.array(prices)
    high_prices = close_prices * (1 + np.abs(np.random.randn(n_points) * 0.005))
    low_prices = close_prices * (1 - np.abs(np.random.randn(n_points) * 0.005))
    open_prices = close_prices * (1 + np.random.randn(n_points) * 0.003)
    volume = np.random.uniform(1000, 10000, n_points)
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)
    
    return df


def example_basic_usage():
    """Basic usage of RegimeAdaptiveStrategy."""
    print("=" * 60)
    print("Example 1: Basic RegimeAdaptiveStrategy Usage")
    print("=" * 60)
    
    df = generate_synthetic_data(1000, regime_type='mixed')
    print(f"Generated {len(df)} data points with mixed regimes")
    
    strategy = RegimeAdaptiveStrategy(
        lookback=100,
        hurst_threshold_trend=0.55,
        hurst_threshold_mr=0.45,
        vol_reduce_factor=0.5,
        enable_vol_scaling=True
    )
    
    print("\nGenerating signals...")
    signals = strategy.generate_signals(df)
    
    print(f"\nSignal distribution:")
    print(f"  Long (1):  {(signals == 1).sum():4d} ({(signals == 1).sum()/len(signals)*100:.1f}%)")
    print(f"  Flat (0):  {(signals == 0).sum():4d} ({(signals == 0).sum()/len(signals)*100:.1f}%)")
    print(f"  Short (-1): {(signals == -1).sum():4d} ({(signals == -1).sum()/len(signals)*100:.1f}%)")
    
    print("\nRunning backtest...")
    result = backtest_signals(df, signals, fee_bps=2.0, weight=1.0)
    
    sharpe_ratio = sharpe(result.returns, freq="1h")
    final_equity = result.equity_curve.iloc[-1]
    
    print(f"\nBacktest Results:")
    print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"  Final Equity: ${final_equity:.2f}")
    print(f"  Total Return: {(final_equity - 1.0) * 100:.2f}%")
    
    regime_history = strategy.get_regime_history()
    if not regime_history.empty:
        print(f"\nRegime History (last 5 periods):")
        print(regime_history[['trend_regime', 'volatility_regime', 'hurst_exponent', 'volatility']].tail())
    
    regime_stats = strategy.get_regime_stats()
    if regime_stats:
        print(f"\nRegime Statistics:")
        print(f"  Trend Distribution:")
        for regime, count in regime_stats['trend_regime_distribution'].items():
            print(f"    {regime}: {count} ({count/len(regime_history)*100:.1f}%)")
        print(f"  Volatility Distribution:")
        for regime, count in regime_stats['volatility_regime_distribution'].items():
            print(f"    {regime}: {count} ({count/len(regime_history)*100:.1f}%)")
        print(f"  Mean Hurst Exponent: {regime_stats['mean_hurst_exponent']:.3f}")
        print(f"  Mean Volatility: {regime_stats['mean_volatility']:.4f}")
    
    print()


def example_regime_specific_wfo():
    """Demonstrate regime-specific WFO backtesting."""
    print("=" * 60)
    print("Example 2: Regime-Specific WFO Backtesting")
    print("=" * 60)
    
    df = generate_synthetic_data(2000, regime_type='mixed')
    print(f"Generated {len(df)} data points")
    
    def strategy_factory(lookback=100, vol_reduce_factor=0.5):
        return RegimeAdaptiveStrategy(
            lookback=lookback,
            vol_reduce_factor=vol_reduce_factor,
            enable_vol_scaling=True
        )
    
    param_grid = {
        'lookback': [50, 100, 150],
        'vol_reduce_factor': [0.3, 0.5, 0.7]
    }
    
    wfo_spec = WFOSpec(n_splits=4, train_frac=0.7)
    
    print("\nRunning regime-specific WFO (this may take a moment)...")
    results = walk_forward_evaluate_regime_specific(
        df=df,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        fee_bps=2.0,
        weight=1.0,
        wfo=wfo_spec
    )
    
    print(f"\nOverall Results:")
    print(f"  Mean Test Sharpe: {results['mean_test_sharpe']:.2f}")
    print(f"  Test Sharpes per fold: {[f'{s:.2f}' for s in results['test_sharpes']]}")
    print(f"  Number of folds: {len(results['test_sharpes'])}")
    
    print(f"\nBest Parameters per Split:")
    for i, params in enumerate(results['best_params_per_split']):
        print(f"  Split {i+1}: {params}")
    
    print(f"\nRegime Distribution:")
    for regime, dist in results['regime_distribution'].items():
        print(f"  {regime:15s}: {dist['count']:4d} periods ({dist['percentage']:5.1f}%)")
    
    print(f"\nRegime-Specific Performance:")
    for regime, perf in results['regime_performance'].items():
        if perf['num_periods'] > 0:
            print(f"  {regime:15s}:")
            print(f"    Sharpe:      {perf['sharpe']:7.2f}")
            print(f"    Mean Return: {perf['mean_return']:7.6f}")
            print(f"    Std Return:  {perf['std_return']:7.6f}")
            print(f"    Num Periods: {perf['num_periods']:7d}")
    
    print()


def example_strategy_comparison():
    """Compare multiple strategies by regime."""
    print("=" * 60)
    print("Example 3: Strategy Comparison by Regime")
    print("=" * 60)
    
    df = generate_synthetic_data(1500, regime_type='mixed')
    print(f"Generated {len(df)} data points")
    
    strategies = {
        'RegimeAdaptive': RegimeAdaptiveStrategy(lookback=100),
        'HurstOnly': HurstExponentStrategy(lookback=100),
        'StatArbOnly': StatArbStrategy(lookback=100)
    }
    
    print("\nComparing strategies across regimes (this may take a moment)...")
    comparison_df = compare_strategies_by_regime(
        df=df,
        strategies=strategies,
        fee_bps=2.0,
        weight=1.0
    )
    
    if not comparison_df.empty:
        print("\nStrategy Comparison Results:")
        print(comparison_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        
        print("\nBest Strategy by Regime:")
        for regime in ['overall', 'trending', 'mean_reverting', 'volatile', 'neutral']:
            regime_data = comparison_df[comparison_df['regime'] == regime]
            if not regime_data.empty:
                best_idx = regime_data['sharpe'].idxmax()
                best = regime_data.loc[best_idx]
                print(f"  {regime:15s}: {best['strategy']:15s} (Sharpe: {best['sharpe']:6.2f})")
    
    print()


def example_advanced_usage():
    """Advanced usage with custom parameters and pairs trading."""
    print("=" * 60)
    print("Example 4: Advanced Usage")
    print("=" * 60)
    
    df1 = generate_synthetic_data(1000, regime_type='mixed')
    df2 = generate_synthetic_data(1000, regime_type='mixed')
    print(f"Generated {len(df1)} data points for two correlated assets")
    
    print("\n--- Test 1: Custom sub-strategy parameters ---")
    strategy1 = RegimeAdaptiveStrategy(
        lookback=100,
        hurst_threshold_trend=0.60,
        hurst_threshold_mr=0.40,
        hurst_params={
            'lookback': 80,
            'trend_threshold': 0.6,
            'mr_threshold': 0.4
        },
        stat_arb_params={
            'lookback': 80,
            'entry_z': 2.5,
            'exit_z': 0.5
        }
    )
    
    signals1 = strategy1.generate_signals(df1)
    result1 = backtest_signals(df1, signals1, fee_bps=2.0, weight=1.0)
    sharpe1 = sharpe(result1.returns, freq="1h")
    
    print(f"Custom parameters Sharpe: {sharpe1:.2f}")
    print(f"Signal distribution: Long={((signals1==1).sum())}, Flat={((signals1==0).sum())}, Short={((signals1==-1).sum())}")
    
    print("\n--- Test 2: Volatility scaling disabled ---")
    strategy2 = RegimeAdaptiveStrategy(
        lookback=100,
        enable_vol_scaling=False
    )
    
    signals2 = strategy2.generate_signals(df1)
    result2 = backtest_signals(df1, signals2, fee_bps=2.0, weight=1.0)
    sharpe2 = sharpe(result2.returns, freq="1h")
    
    print(f"No vol scaling Sharpe: {sharpe2:.2f}")
    print(f"Signal distribution: Long={((signals2==1).sum())}, Flat={((signals2==0).sum())}, Short={((signals2==-1).sum())}")
    
    print("\n--- Test 3: Different regime thresholds ---")
    strategy3 = RegimeAdaptiveStrategy(
        lookback=100,
        hurst_threshold_trend=0.65,
        hurst_threshold_mr=0.35,
        vol_reduce_factor=0.3
    )
    
    signals3 = strategy3.generate_signals(df1)
    result3 = backtest_signals(df1, signals3, fee_bps=2.0, weight=1.0)
    sharpe3 = sharpe(result3.returns, freq="1h")
    
    print(f"Wider thresholds Sharpe: {sharpe3:.2f}")
    print(f"Signal distribution: Long={((signals3==1).sum())}, Flat={((signals3==0).sum())}, Short={((signals3==-1).sum())}")
    
    stats3 = strategy3.get_regime_stats()
    if stats3:
        print(f"Regime distribution with wider thresholds:")
        for regime, count in stats3['trend_regime_distribution'].items():
            print(f"  {regime}: {count}")
    
    print()


def example_wfo_integration():
    """Demonstrate standard WFO with RegimeAdaptiveStrategy."""
    print("=" * 60)
    print("Example 5: Standard WFO Integration")
    print("=" * 60)
    
    df = generate_synthetic_data(2000, regime_type='mixed')
    print(f"Generated {len(df)} data points")
    
    def strategy_factory(lookback=100, hurst_threshold_trend=0.55, hurst_threshold_mr=0.45):
        return RegimeAdaptiveStrategy(
            lookback=lookback,
            hurst_threshold_trend=hurst_threshold_trend,
            hurst_threshold_mr=hurst_threshold_mr
        )
    
    param_grid = {
        'lookback': [80, 100, 120],
        'hurst_threshold_trend': [0.50, 0.55, 0.60],
        'hurst_threshold_mr': [0.40, 0.45, 0.50]
    }
    
    wfo_spec = WFOSpec(n_splits=3, train_frac=0.7)
    
    print("\nRunning standard WFO (this may take a moment)...")
    results = walk_forward_evaluate(
        df=df,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        fee_bps=2.0,
        weight=1.0,
        wfo=wfo_spec
    )
    
    print(f"\nStandard WFO Results:")
    print(f"  Mean Test Sharpe: {results['mean_test_sharpe']:.2f}")
    print(f"  Test Sharpes: {[f'{s:.2f}' for s in results['test_sharpes']]}")
    
    print(f"\nBest Parameters per Split:")
    for i, params in enumerate(results['best_params_per_split']):
        print(f"  Split {i+1}:")
        for key, value in params.items():
            print(f"    {key}: {value}")
    
    print()


def example_regime_analysis():
    """Analyze regime detection and transitions."""
    print("=" * 60)
    print("Example 6: Regime Detection Analysis")
    print("=" * 60)
    
    df = generate_synthetic_data(1000, regime_type='mixed')
    print(f"Generated {len(df)} data points")
    
    strategy = RegimeAdaptiveStrategy(lookback=50)
    signals = strategy.generate_signals(df)
    
    regime_history = strategy.get_regime_history()
    
    if not regime_history.empty:
        print("\nRegime Transition Analysis:")
        
        trend_changes = (regime_history['trend_regime'] != regime_history['trend_regime'].shift()).sum()
        vol_changes = (regime_history['volatility_regime'] != regime_history['volatility_regime'].shift()).sum()
        
        print(f"  Number of trend regime transitions: {trend_changes}")
        print(f"  Number of volatility regime transitions: {vol_changes}")
        
        print("\nHurst Exponent Distribution:")
        hurst_values = regime_history['hurst_exponent']
        print(f"  Min:    {hurst_values.min():.3f}")
        print(f"  25%:    {hurst_values.quantile(0.25):.3f}")
        print(f"  Median: {hurst_values.median():.3f}")
        print(f"  75%:    {hurst_values.quantile(0.75):.3f}")
        print(f"  Max:    {hurst_values.max():.3f}")
        
        print("\nVolatility Distribution:")
        vol_values = regime_history['volatility']
        print(f"  Min:    {vol_values.min():.6f}")
        print(f"  25%:    {vol_values.quantile(0.25):.6f}")
        print(f"  Median: {vol_values.median():.6f}")
        print(f"  75%:    {vol_values.quantile(0.75):.6f}")
        print(f"  Max:    {vol_values.max():.6f}")
        
        print("\nCorrelation between Hurst and Volatility:")
        correlation = hurst_values.corr(vol_values)
        print(f"  Correlation coefficient: {correlation:.3f}")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Regime Adaptive Strategy Examples")
    print("=" * 60 + "\n")
    
    np.random.seed(42)
    
    example_basic_usage()
    example_advanced_usage()
    example_regime_analysis()
    example_strategy_comparison()
    example_wfo_integration()
    example_regime_specific_wfo()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
