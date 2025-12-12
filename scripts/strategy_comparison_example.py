"""Example: Strategy Backtesting Comparison Report.

Demonstrates how to use the strategy comparison module to:
1. Compare multiple strategy backtest results
2. Generate comprehensive metrics tables
3. Create equity curve and drawdown overlays
4. Perform statistical tests (t-test, Diebold-Mariano)
5. Export results to reports and CSV files
"""
import pandas as pd
import numpy as np
from pathlib import Path

from openquant.backtest.engine import backtest_signals, BacktestResult
from openquant.reporting.strategy_comparison import (
    compare_strategies,
    generate_comparison_report,
    export_comparison_to_csv,
)


def create_sample_data(n: int = 1000) -> pd.DataFrame:
    """Create sample OHLCV data for demonstration."""
    dates = pd.date_range(start='2022-01-01', periods=n, freq='1h')
    
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high_prices = close_prices + np.abs(np.random.randn(n) * 0.3)
    low_prices = close_prices - np.abs(np.random.randn(n) * 0.3)
    open_prices = close_prices + np.random.randn(n) * 0.2
    volume = np.random.uniform(1000, 10000, n)
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)
    
    return df


def strategy_moving_average_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    """Simple moving average crossover strategy."""
    ma_fast = df['Close'].rolling(fast).mean()
    ma_slow = df['Close'].rolling(slow).mean()
    
    signals = pd.Series(0, index=df.index)
    signals[ma_fast > ma_slow] = 1
    signals[ma_fast < ma_slow] = -1
    
    return signals


def strategy_rsi(df: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """RSI-based strategy."""
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    signals = pd.Series(0, index=df.index)
    signals[rsi < oversold] = 1
    signals[rsi > overbought] = -1
    
    return signals


def strategy_momentum(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Momentum strategy."""
    momentum = df['Close'].pct_change(lookback)
    
    signals = pd.Series(0, index=df.index)
    signals[momentum > 0.02] = 1
    signals[momentum < -0.02] = -1
    
    return signals


def strategy_mean_reversion(df: pd.DataFrame, window: int = 20, threshold: float = 2.0) -> pd.Series:
    """Mean reversion strategy based on Bollinger Bands."""
    ma = df['Close'].rolling(window).mean()
    std = df['Close'].rolling(window).std()
    
    upper_band = ma + threshold * std
    lower_band = ma - threshold * std
    
    signals = pd.Series(0, index=df.index)
    signals[df['Close'] < lower_band] = 1
    signals[df['Close'] > upper_band] = -1
    
    return signals


def main():
    """Main example demonstrating strategy comparison."""
    print("=" * 80)
    print("STRATEGY COMPARISON EXAMPLE")
    print("=" * 80)
    print()
    
    print("1. Generating sample OHLCV data...")
    df = create_sample_data(n=2000)
    print(f"   Created {len(df)} bars of data")
    print()
    
    print("2. Running backtests for multiple strategies...")
    strategies = {
        "MA_Crossover_10_30": strategy_moving_average_crossover(df, 10, 30),
        "MA_Crossover_20_50": strategy_moving_average_crossover(df, 20, 50),
        "RSI_14": strategy_rsi(df, 14, 30, 70),
        "Momentum_20": strategy_momentum(df, 20),
        "Mean_Reversion": strategy_mean_reversion(df, 20, 2.0),
    }
    
    results = {}
    for name, signals in strategies.items():
        print(f"   Backtesting: {name}...")
        result = backtest_signals(
            df=df,
            signals=signals,
            fee_bps=1.0,
            weight=1.0
        )
        results[name] = result
    
    print(f"   Completed {len(results)} backtests")
    print()
    
    print("3. Comparing strategies...")
    comparison = compare_strategies(
        results=results,
        freq="1h",
        ranking_method="composite"
    )
    print("   Comparison complete")
    print()
    
    print("4. Strategy Rankings:")
    print(comparison.ranked_strategies)
    print()
    
    print("5. Performance Metrics Summary:")
    key_metrics = [
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown (%)",
        "Total Return (%)",
        "Win Rate (%)"
    ]
    print(comparison.metrics_table.loc[key_metrics])
    print()
    
    print("6. Statistical Test Summary:")
    for test_name, tests in comparison.statistical_tests.items():
        print(f"\n   {test_name}:")
        t_test = tests["t_test"]
        dm_test = tests["diebold_mariano"]
        
        print(f"      T-test p-value: {t_test['p_value']:.4f} "
              f"({'Significant' if t_test['significant'] else 'Not significant'})")
        print(f"      DM-test p-value: {dm_test['p_value']:.4f} "
              f"({'Significant' if dm_test['significant'] else 'Not significant'})")
        
        if dm_test['significant']:
            print(f"      Better strategy: {dm_test['better_strategy']}")
    print()
    
    print("7. Generating reports and exports...")
    output_dir = Path("data/strategy_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_text = generate_comparison_report(
        comparison=comparison,
        output_path=output_dir / "comparison_report.txt",
        include_plots=True
    )
    
    export_comparison_to_csv(
        comparison=comparison,
        output_dir=output_dir
    )
    
    print(f"   Reports saved to: {output_dir}/")
    print("   - comparison_report.txt")
    print("   - metrics_comparison.csv")
    print("   - equity_curves.csv")
    print("   - drawdown_curves.csv")
    print("   - correlation_matrix.csv")
    print("   - strategy_rankings.csv")
    print("   - statistical_tests.csv")
    print("   - strategy_comparison.png")
    print("   - correlation_matrix.png")
    print()
    
    print("=" * 80)
    print("EXAMPLE COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("- Review the generated reports and CSV files")
    print("- Adjust strategy parameters and re-run comparison")
    print("- Use statistical tests to select best strategy")
    print("- Consider strategy ensemble or rotation based on results")


if __name__ == "__main__":
    main()
