"""Environment checker for Profitability Testing Framework.

Validates that all required dependencies and configurations are in place
before running profitability tests.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_python_version():
    """Check Python version."""
    print("Checking Python version...", end=" ")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} (requires 3.10+)")
        return False


def check_required_packages():
    """Check required packages."""
    print("\nChecking required packages...")
    
    packages = {
        "pandas": "pandas",
        "numpy": "numpy",
        "scipy": "scipy",
        "statsmodels": "statsmodels",
        "ccxt": "ccxt (for crypto data)",
        "yfinance": "yfinance (for stock data)",
    }
    
    all_ok = True
    for module, display_name in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ✗ {display_name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok


def check_openquant_modules():
    """Check OpenQuant modules."""
    print("\nChecking OpenQuant modules...")
    
    modules = [
        ("openquant.backtest.engine", "Backtest Engine"),
        ("openquant.backtest.metrics", "Performance Metrics"),
        ("openquant.data.loader", "Data Loader"),
        ("openquant.strategies.registry", "Strategy Registry"),
        ("openquant.strategies.quant.stat_arb", "Statistical Arbitrage"),
        ("openquant.strategies.quant.kalman", "Kalman Filter"),
        ("openquant.strategies.quant.hurst", "Hurst Exponent"),
        ("openquant.utils.logging", "Logging Utility"),
    ]
    
    all_ok = True
    for module_path, display_name in modules:
        try:
            __import__(module_path)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name} - IMPORT ERROR: {e}")
            all_ok = False
    
    return all_ok


def check_data_access():
    """Check data source access."""
    print("\nChecking data source access...")
    
    from openquant.data.loader import DataLoader
    from datetime import datetime, timedelta
    
    loader = DataLoader()
    
    sources = [
        ("ccxt:binance", "BTC/USDT", "Binance (Crypto)"),
    ]
    
    all_ok = True
    for source, symbol, display_name in sources:
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=7)
            
            df = loader.get_ohlcv(
                source=source,
                symbol=symbol,
                timeframe="1d",
                start=start,
                end=end,
                limit=10
            )
            
            if df.empty:
                print(f"  ⚠ {display_name} - No data returned (may be rate limited)")
                all_ok = False
            else:
                print(f"  ✓ {display_name} - {len(df)} bars retrieved")
        except Exception as e:
            print(f"  ✗ {display_name} - ERROR: {e}")
            all_ok = False
    
    return all_ok


def check_strategy_registry():
    """Check strategy registry."""
    print("\nChecking strategy registry...")
    
    try:
        from openquant.strategies.registry import REGISTRY, get_strategy
        
        print(f"  Available strategies: {len(REGISTRY)}")
        for name in REGISTRY.keys():
            try:
                strategy = get_strategy(name)
                print(f"    ✓ {name}")
            except Exception as e:
                print(f"    ✗ {name} - ERROR: {e}")
        
        return True
    except Exception as e:
        print(f"  ✗ Registry error: {e}")
        return False


def check_directories():
    """Check required directories."""
    print("\nChecking directories...")
    
    dirs = ["reports", "logs", "data"]
    all_ok = True
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✓ {dir_name}/ exists")
        else:
            print(f"  ⚠ {dir_name}/ missing (will be created on first run)")
    
    return all_ok


def test_basic_functionality():
    """Test basic framework functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from test_profitability import ProfitabilityTester
        
        print("  ✓ ProfitabilityTester imported successfully")
        
        tester = ProfitabilityTester(
            strategy_name="stat_arb",
            symbols=["BTC/USDT"],
            timeframe="1h",
            monte_carlo_runs=10
        )
        
        print("  ✓ ProfitabilityTester instantiated successfully")
        
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n = 1000
        price = 1.0 + np.cumsum(np.random.randn(n) * 0.01)
        df = pd.DataFrame({
            'Open': price,
            'High': price * 1.01,
            'Low': price * 0.99,
            'Close': price,
            'Volume': np.random.randint(1000, 10000, n)
        }, index=pd.date_range('2024-01-01', periods=n, freq='1h'))
        
        from openquant.strategies.quant.stat_arb import StatArbStrategy
        strategy = StatArbStrategy()
        signals = strategy.generate_signals(df)
        
        print(f"  ✓ Generated {len(signals)} signals")
        
        from openquant.backtest.engine import backtest_signals
        result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)
        
        print(f"  ✓ Backtest completed (equity: {result.equity_curve.iloc[-1]:.4f})")
        
        metrics = tester.calculate_metrics(result, freq="1h")
        print(f"  ✓ Metrics calculated (Sharpe: {metrics.sharpe_ratio:.2f})")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks."""
    print("=" * 80)
    print("PROFITABILITY TESTING FRAMEWORK - ENVIRONMENT CHECK")
    print("=" * 80)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("OpenQuant Modules", check_openquant_modules),
        ("Data Access", check_data_access),
        ("Strategy Registry", check_strategy_registry),
        ("Directories", check_directories),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n✗ {name} check failed with exception: {e}")
            results[name] = False
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<30} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready to run profitability tests!")
        print("\nNext steps:")
        print("  python scripts/test_profitability.py --strategy stat_arb --symbol BTC/USDT")
        print("  python scripts/example_profitability_test.py --example 1")
        return 0
    else:
        print("✗ SOME CHECKS FAILED - Please fix the issues above")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Check internet connection for data access")
        print("  - Verify .env file is configured (if using API keys)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
