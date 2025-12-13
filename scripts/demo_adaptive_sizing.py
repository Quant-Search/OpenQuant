"""Demo script showing adaptive position sizing in backtest_signals."""
import pandas as pd
import numpy as np
from openquant.backtest.engine import backtest_signals

# Create sample data
np.random.seed(42)
idx = pd.date_range("2020-01-01", periods=500, freq="1h", tz="UTC")
close = pd.Series(100 + np.cumsum(np.random.randn(500) * 0.5), index=idx)
df = pd.DataFrame({
    "Close": close,
    "High": close * 1.01,
    "Low": close * 0.99,
    "Open": close,
    "Volume": 1000
})

# Generate sample signals (simple momentum)
returns = df["Close"].pct_change()
signals = pd.Series(0, index=idx)
signals[returns > 0.005] = 1
signals[returns < -0.005] = -1
signals = signals.shift(1).fillna(0)

print("=" * 60)
print("Adaptive Position Sizing Demo")
print("=" * 60)

# Test 1: Fixed sizing (baseline)
print("\n1. Fixed Sizing (baseline):")
result_fixed = backtest_signals(
    df, signals, 
    fee_bps=1.0, 
    weight=1.0,
    sizing_method="fixed"
)
print(f"   Final Equity: {result_fixed.equity_curve.iloc[-1]:.4f}")
print(f"   Total Return: {(result_fixed.equity_curve.iloc[-1] - 1) * 100:.2f}%")
print(f"   Total Trades: {result_fixed.trades.sum():.0f}")

# Test 2: Kelly Criterion sizing
print("\n2. Kelly Criterion Sizing:")
result_kelly = backtest_signals(
    df, signals, 
    fee_bps=1.0, 
    weight=1.0,
    sizing_method="kelly"
)
print(f"   Final Equity: {result_kelly.equity_curve.iloc[-1]:.4f}")
print(f"   Total Return: {(result_kelly.equity_curve.iloc[-1] - 1) * 100:.2f}%")
print(f"   Total Trades: {result_kelly.trades.sum():.0f}")

# Test 3: Volatility-based sizing
print("\n3. Volatility-Based Sizing:")
result_vol = backtest_signals(
    df, signals, 
    fee_bps=1.0, 
    weight=1.0,
    sizing_method="volatility"
)
print(f"   Final Equity: {result_vol.equity_curve.iloc[-1]:.4f}")
print(f"   Total Return: {(result_vol.equity_curve.iloc[-1] - 1) * 100:.2f}%")
print(f"   Total Trades: {result_vol.trades.sum():.0f}")

# Test 4: Adaptive sizing (full)
print("\n4. Adaptive Sizing (full):")
result_adaptive = backtest_signals(
    df, signals, 
    fee_bps=1.0, 
    weight=1.0,
    sizing_method="adaptive"
)
print(f"   Final Equity: {result_adaptive.equity_curve.iloc[-1]:.4f}")
print(f"   Total Return: {(result_adaptive.equity_curve.iloc[-1] - 1) * 100:.2f}%")
print(f"   Total Trades: {result_adaptive.trades.sum():.0f}")

print("\n" + "=" * 60)
print("Demo completed successfully!")
print("=" * 60)
