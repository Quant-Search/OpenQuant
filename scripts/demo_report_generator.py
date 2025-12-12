"""Demo script showing how to use the ProfitabilityReportGenerator.

This demonstrates generating comprehensive backtest reports with:
- Equity curves with Monte Carlo confidence intervals
- Monthly/yearly returns heatmaps
- Drawdown analysis with underwater charts
- Regime-specific performance
- Top 10 best/worst trades
- Performance attribution
- Stress test results
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from openquant.backtest.engine import backtest_signals, BacktestResult
from openquant.reporting.profitability_report import ProfitabilityReportGenerator


def create_sample_backtest_data():
    """Create sample backtest data for demonstration."""
    
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='1D')
    n = len(dates)
    
    np.random.seed(42)
    returns = np.random.randn(n) * 0.02
    close_prices = 100 * (1 + pd.Series(returns)).cumprod()
    
    high_prices = close_prices * (1 + np.abs(np.random.randn(n)) * 0.01)
    low_prices = close_prices * (1 - np.abs(np.random.randn(n)) * 0.01)
    open_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    volume = np.random.randint(1000000, 10000000, n)
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)
    
    sma_short = df['Close'].rolling(20).mean()
    sma_long = df['Close'].rolling(50).mean()
    
    signals = pd.Series(0, index=dates)
    signals[sma_short > sma_long] = 1
    signals[sma_short <= sma_long] = 0
    
    volatility = df['Close'].pct_change().rolling(20).std()
    regime_labels = pd.Series('NORMAL', index=dates)
    regime_labels[volatility > volatility.quantile(0.75)] = 'HIGH_VOL'
    regime_labels[volatility < volatility.quantile(0.25)] = 'LOW_VOL'
    
    return df, signals, regime_labels


def main():
    """Run demo report generation."""
    print("=" * 70)
    print("OpenQuant Profitability Report Generator Demo")
    print("=" * 70)
    print()
    
    print("📊 Creating sample backtest data...")
    df, signals, regime_labels = create_sample_backtest_data()
    print(f"   ✅ Generated {len(df)} days of data")
    print(f"   ✅ Signals: {(signals == 1).sum()} long, {(signals == 0).sum()} flat")
    print()
    
    print("🔄 Running backtest...")
    result = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=1.0,
        weight=1.0
    )
    
    total_return = (result.equity_curve.iloc[-1] - 1) * 100
    print(f"   ✅ Backtest complete")
    print(f"   ✅ Total Return: {total_return:.2f}%")
    print()
    
    print("📄 Generating comprehensive reports...")
    generator = ProfitabilityReportGenerator(output_dir=Path("reports"))
    
    report_files = generator.generate_report(
        result=result,
        df=df,
        strategy_name="Demo_SMA_Crossover",
        freq="1d",
        regime_labels=regime_labels,
        format="both",
        monte_carlo_runs=500
    )
    
    print()
    print("✅ Report generation complete!")
    print()
    
    for format_type, file_path in report_files.items():
        print(f"   📁 {format_type.upper()}: {file_path}")
    
    print()
    print("=" * 70)
    print("Report Features:")
    print("=" * 70)
    print("✓ Equity curve with Monte Carlo confidence bands")
    print("✓ Monthly/yearly returns heatmaps")
    print("✓ Drawdown timeline and underwater chart")
    print("✓ Regime-specific performance analysis")
    print("✓ Top 10 best and worst trades")
    print("✓ Performance attribution (long vs short)")
    print("✓ Stress test scenarios")
    print("✓ Comprehensive metrics (Sharpe, Sortino, Calmar, etc.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
