"""Example script for using the Regime Adaptive Strategy.

Demonstrates:
1. Creating and using RegimeAdaptiveStrategy
2. Running regime-specific WFO backtesting
3. Comparing strategies by regime
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from openquant.strategies.regime_adaptive import RegimeAdaptiveStrategy
from openquant.strategies.quant.hurst import HurstExponentStrategy
from openquant.strategies.quant.stat_arb import StatArbStrategy
from openquant.evaluation.wfo import (
    walk_forward_evaluate_regime_specific,
    compare_strategies_by_regime,
    WFOSpec
)
from openquant.backtest.engine import backtest_signals
from openquant.backtest.metrics import sharpe


def generate_synthetic_data(n_points: int = 1000) -> pd.DataFrame:
    """Generate synthetic OHLCV data with different regime periods."""
    dates = pd.date_range(start='2020-01-01', periods=n_points, freq='1H')
    
    price = 100.0
    prices = []
    
    for i in range(n_points):
        if i < n_points // 3:
            drift = 0.0002
            volatility = 0.005
        elif i < 2 * n_points // 3:
            drift = 0.001
            volatility = 0.003
        else:
            drift = 0.0
            volatility = 0.01
        
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
    
    df = generate_synthetic_data(1000)
    print(f"Generated {len(df)} data points")
    
    strategy = RegimeAdaptiveStrategy(
        lookback=100,
        hurst_threshold_trend=0.55,
        hurst_threshold_mr=0.45,
        vol_reduce_factor=0.5,
        enable_vol_scaling=True
    )
    
    print("\nGenerating signals...")
    signals = strategy.generate_signals(df)
    
    print(f"Signal distribution:")
    print(signals.value_counts())
    
    print("\nRunning backtest...")
    result = backtest_signals(df, signals, fee_bps=2.0, weight=1.0)
    
    sharpe_ratio = sharpe(result.returns, freq="1h")
    final_equity = result.equity_curve.iloc[-1]
    
    print(f"\nBacktest Results:")
    print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"  Final Equity: {final_equity:.2f}")
    print(f"  Total Return: {(final_equity - 1.0) * 100:.2f}%")
    
    regime_history = strategy.get_regime_history()
    if not regime_history.empty:
        print(f"\nRegime History (last 5 periods):")
        print(regime_history.tail())
    
    regime_stats = strategy.get_regime_stats()
    if regime_stats:
        print(f"\nRegime Statistics:")
        print(f"  Trend Distribution: {regime_stats['trend_regime_distribution']}")
        print(f"  Volatility Distribution: {regime_stats['volatility_regime_distribution']}")
        print(f"  Mean Hurst: {regime_stats['mean_hurst_exponent']:.3f}")
        print(f"  Mean Volatility: {regime_stats['mean_volatility']:.4f}")
    
    print()


def example_regime_specific_wfo():
    """Demonstrate regime-specific WFO backtesting."""
    print("=" * 60)
    print("Example 2: Regime-Specific WFO Backtesting")
    print("=" * 60)
    
    df = generate_synthetic_data(2000)
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
    
    print("\nRunning regime-specific WFO...")
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
    print(f"  Test Sharpes: {[f'{s:.2f}' for s in results['test_sharpes']]}")
    
    print(f"\nRegime Distribution:")
    for regime, dist in results['regime_distribution'].items():
        print(f"  {regime}: {dist['count']} periods ({dist['percentage']:.1f}%)")
    
    print(f"\nRegime-Specific Performance:")
    for regime, perf in results['regime_performance'].items():
        print(f"  {regime}:")
        print(f"    Sharpe: {perf['sharpe']:.2f}")
        print(f"    Mean Return: {perf['mean_return']:.6f}")
        print(f"    Std Return: {perf['std_return']:.6f}")
    
    print()


def example_strategy_comparison():
    """Compare multiple strategies by regime."""
    print("=" * 60)
    print("Example 3: Strategy Comparison by Regime")
    print("=" * 60)
    
    df = generate_synthetic_data(1500)
    print(f"Generated {len(df)} data points")
    
    strategies = {
        'RegimeAdaptive': RegimeAdaptiveStrategy(lookback=100),
        'HurstOnly': HurstExponentStrategy(lookback=100),
        'StatArbOnly': StatArbStrategy(lookback=100)
    }
    
    print("\nComparing strategies across regimes...")
    comparison_df = compare_strategies_by_regime(
        df=df,
        strategies=strategies,
        fee_bps=2.0,
        weight=1.0
    )
    
    print("\nStrategy Comparison Results:")
    print(comparison_df.to_string(index=False))
    
    print("\nBest Strategy by Regime:")
    for regime in ['overall', 'trending', 'mean_reverting', 'volatile', 'neutral']:
        regime_data = comparison_df[comparison_df['regime'] == regime]
        if not regime_data.empty:
            best = regime_data.loc[regime_data['sharpe'].idxmax()]
            print(f"  {regime}: {best['strategy']} (Sharpe: {best['sharpe']:.2f})")
    
    print()


def example_custom_regime_classifier():
    """Example with custom regime classifier."""
    print("=" * 60)
    print("Example 4: Custom Regime Classifier")
    print("=" * 60)
    
    df = generate_synthetic_data(1000)
    print(f"Generated {len(df)} data points")
    
    def custom_classifier(df_window: pd.DataFrame) -> str:
        """Custom regime classifier based on simple volatility."""
        returns = df_window['Close'].pct_change().tail(50)
        vol = returns.std()
        
        if vol > 0.015:
            return 'volatile'
        elif vol < 0.005:
            return 'low_vol_trending'
        else:
            return 'normal'
    
    def strategy_factory(lookback=100):
        return RegimeAdaptiveStrategy(lookback=lookback)
    
    param_grid = {'lookback': [80, 100, 120]}
    wfo_spec = WFOSpec(n_splits=3, train_frac=0.7)
    
    print("\nRunning WFO with custom classifier...")
    results = walk_forward_evaluate_regime_specific(
        df=df,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        fee_bps=2.0,
        weight=1.0,
        wfo=wfo_spec,
        regime_classifier=custom_classifier
    )
    
    print(f"\nResults with Custom Classifier:")
    print(f"  Mean Test Sharpe: {results['mean_test_sharpe']:.2f}")
    
    print(f"\nCustom Regime Distribution:")
    for regime, dist in results['regime_distribution'].items():
        if dist['count'] > 0:
            print(f"  {regime}: {dist['count']} periods ({dist['percentage']:.1f}%)")
    
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Regime Adaptive Strategy Examples")
    print("=" * 60 + "\n")
    
    example_basic_usage()
    example_regime_specific_wfo()
    example_strategy_comparison()
    example_custom_regime_classifier()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
