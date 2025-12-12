"""Generate comprehensive backtest report from existing backtest results.

This script demonstrates how to integrate the ProfitabilityReportGenerator
with your existing backtesting workflow. It can be used as a template for
generating reports from any strategy backtest.

Usage:
    python scripts/generate_backtest_report.py --strategy stat_arb --symbol BTC/USD
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from openquant.backtest.engine import backtest_signals, BacktestResult
from openquant.reporting.profitability_report import ProfitabilityReportGenerator
from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)


def load_or_generate_data(symbol: str = "BTC/USD", period: str = "1y"):
    """Load historical data or generate synthetic data for demo."""
    try:
        import yfinance as yf
        
        ticker_map = {
            "BTC/USD": "BTC-USD",
            "ETH/USD": "ETH-USD",
            "EUR/USD": "EURUSD=X",
            "GBP/USD": "GBPUSD=X"
        }
        
        ticker = ticker_map.get(symbol, symbol.replace("/", "-"))
        LOGGER.info(f"Downloading data for {ticker}...")
        
        df = yf.Ticker(ticker).history(period=period, interval="1d")
        
        if df.empty:
            raise ValueError(f"No data returned for {symbol}")
        
        LOGGER.info(f"Downloaded {len(df)} bars for {symbol}")
        return df
        
    except Exception as e:
        LOGGER.warning(f"Failed to download data: {e}. Using synthetic data.")
        return generate_synthetic_data()


def generate_synthetic_data(days: int = 500):
    """Generate synthetic OHLCV data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=days, freq='1D')
    
    np.random.seed(42)
    returns = np.random.randn(days) * 0.015 + 0.0002
    close_prices = 100 * (1 + pd.Series(returns)).cumprod()
    
    high_prices = close_prices * (1 + np.abs(np.random.randn(days)) * 0.01)
    low_prices = close_prices * (1 - np.abs(np.random.randn(days)) * 0.01)
    open_prices = close_prices.shift(1).fillna(close_prices.iloc[0])
    volume = np.random.randint(1000000, 10000000, days)
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)
    
    return df


def simple_strategy_signals(df: pd.DataFrame, strategy: str = "sma_cross") -> pd.Series:
    """Generate trading signals based on simple strategies."""
    
    if strategy == "sma_cross":
        sma_fast = df['Close'].rolling(20).mean()
        sma_slow = df['Close'].rolling(50).mean()
        signals = pd.Series(0, index=df.index)
        signals[sma_fast > sma_slow] = 1
        return signals
    
    elif strategy == "rsi":
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=df.index)
        signals[(rsi < 30)] = 1
        signals[(rsi > 70)] = 0
        return signals
    
    elif strategy == "bollinger":
        sma = df['Close'].rolling(20).mean()
        std = df['Close'].rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        signals = pd.Series(0, index=df.index)
        signals[df['Close'] < lower] = 1
        signals[df['Close'] > upper] = 0
        return signals
    
    elif strategy == "stat_arb":
        returns = df['Close'].pct_change()
        zscore = (returns - returns.rolling(20).mean()) / returns.rolling(20).std()
        
        signals = pd.Series(0, index=df.index)
        signals[zscore < -2] = 1
        signals[zscore > 2] = -1
        signals[(zscore > -0.5) & (zscore < 0.5)] = 0
        return signals
    
    else:
        LOGGER.warning(f"Unknown strategy {strategy}, using buy-and-hold")
        return pd.Series(1, index=df.index)


def detect_market_regime(df: pd.DataFrame) -> pd.Series:
    """Detect market regimes based on volatility and trend."""
    
    returns = df['Close'].pct_change()
    volatility = returns.rolling(20).std()
    
    sma_50 = df['Close'].rolling(50).mean()
    sma_200 = df['Close'].rolling(200).mean()
    
    regime = pd.Series('NEUTRAL', index=df.index)
    
    high_vol = volatility > volatility.quantile(0.75)
    low_vol = volatility < volatility.quantile(0.25)
    
    uptrend = (df['Close'] > sma_50) & (sma_50 > sma_200)
    downtrend = (df['Close'] < sma_50) & (sma_50 < sma_200)
    
    regime[uptrend & low_vol] = 'BULL_QUIET'
    regime[uptrend & high_vol] = 'BULL_VOLATILE'
    regime[downtrend & low_vol] = 'BEAR_QUIET'
    regime[downtrend & high_vol] = 'BEAR_VOLATILE'
    
    return regime


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive backtest report"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTC/USD",
        help="Trading symbol (e.g., BTC/USD, ETH/USD)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="sma_cross",
        choices=["sma_cross", "rsi", "bollinger", "stat_arb"],
        help="Strategy to backtest"
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1y",
        help="Historical data period (e.g., 1y, 2y, 6mo)"
    )
    parser.add_argument(
        "--format",
        type=str,
        default="both",
        choices=["pdf", "html", "both"],
        help="Report format"
    )
    parser.add_argument(
        "--fee-bps",
        type=float,
        default=1.0,
        help="Trading fee in basis points"
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="Position size as fraction of capital"
    )
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=500,
        help="Number of Monte Carlo simulations"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("OpenQuant Backtest Report Generator")
    print("=" * 70)
    print(f"Symbol:     {args.symbol}")
    print(f"Strategy:   {args.strategy}")
    print(f"Period:     {args.period}")
    print(f"Format:     {args.format}")
    print("=" * 70)
    print()
    
    LOGGER.info("Loading market data...")
    df = load_or_generate_data(args.symbol, args.period)
    print(f"✅ Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    print()
    
    LOGGER.info("Generating trading signals...")
    signals = simple_strategy_signals(df, args.strategy)
    long_signals = (signals == 1).sum()
    flat_signals = (signals == 0).sum()
    short_signals = (signals == -1).sum()
    print(f"✅ Signals: {long_signals} long, {short_signals} short, {flat_signals} flat")
    print()
    
    LOGGER.info("Detecting market regimes...")
    regime_labels = detect_market_regime(df)
    regime_counts = regime_labels.value_counts()
    print("✅ Market regimes detected:")
    for regime, count in regime_counts.items():
        print(f"   - {regime}: {count} days ({count/len(df)*100:.1f}%)")
    print()
    
    LOGGER.info("Running backtest...")
    result = backtest_signals(
        df=df,
        signals=signals,
        fee_bps=args.fee_bps,
        weight=args.weight,
        slippage_bps=0.5
    )
    
    final_equity = result.equity_curve.iloc[-1]
    total_return = (final_equity - 1) * 100
    
    from openquant.backtest.metrics import sharpe, max_drawdown
    sharpe_ratio = sharpe(result.returns, "1d")
    max_dd = max_drawdown(result.equity_curve) * 100
    
    print("✅ Backtest complete:")
    print(f"   - Total Return:    {total_return:>8.2f}%")
    print(f"   - Sharpe Ratio:    {sharpe_ratio:>8.2f}")
    print(f"   - Max Drawdown:    {max_dd:>8.2f}%")
    print(f"   - Final Equity:    {final_equity:>8.2f}")
    print()
    
    LOGGER.info("Generating comprehensive report...")
    generator = ProfitabilityReportGenerator(output_dir=Path("reports"))
    
    strategy_name = f"{args.strategy}_{args.symbol.replace('/', '_')}"
    
    report_files = generator.generate_report(
        result=result,
        df=df,
        strategy_name=strategy_name,
        freq="1d",
        regime_labels=regime_labels,
        format=args.format,
        monte_carlo_runs=args.mc_runs
    )
    
    print("=" * 70)
    print("✅ Report Generation Complete!")
    print("=" * 70)
    
    for format_type, file_path in report_files.items():
        print(f"📁 {format_type.upper()}: {file_path}")
    
    print()
    print("Report includes:")
    print("  • Equity curve with Monte Carlo confidence bands")
    print("  • Monthly/yearly returns heatmaps")
    print("  • Drawdown timeline and underwater chart")
    print("  • Regime-specific performance breakdown")
    print("  • Top 10 best/worst trades analysis")
    print("  • Performance attribution (long vs short)")
    print("  • Stress test scenarios")
    print("  • Comprehensive risk metrics")
    print("=" * 70)


if __name__ == "__main__":
    main()
