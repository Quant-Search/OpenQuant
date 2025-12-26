#!/usr/bin/env python3
"""
Test script for MVP robot imports and functionality.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 60)
print("OpenQuant MVP - Import Test")
print("=" * 60)

# Test 1: Config module
print("\n[1] Testing config module...")
try:
    from robot.config import (
        Config, 
        save_credentials, 
        load_saved_credentials,
        _find_mt5_terminal
    )
    print("    Config import: OK")
    
    # Test MT5 detection
    detected_path = _find_mt5_terminal()
    if detected_path:
        print(f"    MT5 Auto-detected: {detected_path}")
    else:
        print("    MT5 Auto-detected: Not found (OK if MT5 not installed)")
    
    # Test status
    status = Config.get_mt5_status()
    print(f"    MT5 Configured: {status['fully_configured']}")
    
except Exception as e:
    print(f"    ERROR: {e}")
    sys.exit(1)

# Test 2: DataFetcher module
print("\n[2] Testing data_fetcher module...")
try:
    from robot.data_fetcher import DataFetcher
    fetcher = DataFetcher(use_mt5=False)  # Don't require MT5
    print("    DataFetcher: OK")
except Exception as e:
    print(f"    ERROR: {e}")
    sys.exit(1)

# Test 3: Strategy module
print("\n[3] Testing strategy module...")
try:
    from robot.strategy import KalmanStrategy
    strategy = KalmanStrategy(
        process_noise=1e-5,
        measurement_noise=1e-3,
        threshold=1.5
    )
    print("    KalmanStrategy: OK")
except Exception as e:
    print(f"    ERROR: {e}")
    sys.exit(1)

# Test 4: Trader module
print("\n[4] Testing trader module...")
try:
    from robot.trader import Trader
    trader = Trader(mode="paper")
    print(f"    Trader (paper): OK (balance=${trader._paper_cash:,.2f})")
except Exception as e:
    print(f"    ERROR: {e}")
    sys.exit(1)

# Test 5: Risk manager module
print("\n[5] Testing risk_manager module...")
try:
    from robot.risk_manager import RiskManager
    print("    RiskManager: OK")
except Exception as e:
    print(f"    ERROR: {e}")
    sys.exit(1)

# Test 6: Fetch sample data
print("\n[6] Testing data fetch (yfinance fallback)...")
try:
    import pandas as pd
    fetcher = DataFetcher(use_mt5=False)
    df = fetcher.fetch("EURUSD", "1d", bars=50)
    if not df.empty:
        print(f"    Fetched {len(df)} bars for EURUSD")
        print(f"    Latest close: {df['Close'].iloc[-1]:.5f}")
    else:
        print("    Warning: No data fetched (may be a connection issue)")
except Exception as e:
    print(f"    ERROR: {e}")

# Test 7: Generate signals
print("\n[7] Testing signal generation...")
try:
    if not df.empty and len(df) >= 50:
        signals = strategy.generate_signals(df)
        latest_signal = int(signals.iloc[-1])
        signal_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}[latest_signal]
        print(f"    Latest signal: {signal_str}")
    else:
        print("    Skipped (insufficient data)")
except Exception as e:
    print(f"    ERROR: {e}")

print("\n" + "=" * 60)
print("All MVP tests completed!")
print("=" * 60)
print("\nTo run the dashboard:")
print("  streamlit run robot/dashboard.py")
print("\nTo run the robot in paper mode:")
print("  python mvp_robot.py --mode paper")

