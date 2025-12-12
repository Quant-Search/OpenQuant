"""Pre-Production Validation Script.

Run this before going live to validate strategy robustness.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime

def validate_strategy():
    """Run comprehensive validation checks."""
    print("=" * 60)
    print("OpenQuant Pre-Production Validation")
    print("=" * 60)
    print()
    
    results = {
        "passed": 0,
        "failed": 0,
        "warnings": 0
    }
    
    # 1. Check imports
    print("1. Checking module imports...")
    try:
        from openquant.strategies.ml_strategy import MLStrategy
        from openquant.strategies.ensemble_strategy import EnsembleStrategy
        from openquant.quant.regime_detector import RegimeDetector
        from openquant.validation.overfitting_guard import OverfittingGuard
        from openquant.risk.adaptive_sizing import AdaptiveSizer
        from openquant.trading.trade_filter import TradeFilter
        from openquant.reporting.performance_tracker import PERFORMANCE_TRACKER
        from openquant.reporting.intelligent_alerts import IntelligentAlerts
        print("   ✅ All core modules imported successfully")
        results["passed"] += 1
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        results["failed"] += 1
        return results
        
    # 2. Check MT5 availability
    print("\n2. Checking MT5 connection...")
    try:
        from openquant.broker.mt5_broker import MT5_AVAILABLE
        if MT5_AVAILABLE:
            print("   ✅ MT5 is available")
            results["passed"] += 1
        else:
            print("   ⚠️  MT5 not available (install MetaTrader5 on Windows)")
            results["warnings"] += 1
    except Exception as e:
        print(f"   ⚠️  MT5 check failed: {e}")
        results["warnings"] += 1
        
    # 3. Check data directory
    print("\n3. Checking data directory...")
    data_dir = Path("data")
    if data_dir.exists():
        print(f"   ✅ Data directory exists")
        results["passed"] += 1
        
        # Check for key files
        key_files = ["paper_state.json", "results.duckdb"]
        for f in key_files:
            if (data_dir / f).exists():
                print(f"   ✅ {f} found")
            else:
                print(f"   ⚠️  {f} not found (will be created on first run)")
                results["warnings"] += 1
    else:
        print("   ⚠️  Data directory not found (will be created)")
        data_dir.mkdir(exist_ok=True)
        results["warnings"] += 1
        
    # 4. Test strategy generation
    print("\n4. Testing strategy signal generation...")
    try:
        # Create dummy data
        dates = pd.date_range(start='2024-01-01', periods=500, freq='1h')
        np.random.seed(42)
        df = pd.DataFrame({
            'Open': 100 + np.cumsum(np.random.randn(500) * 0.5),
            'High': 100 + np.cumsum(np.random.randn(500) * 0.5) + 0.5,
            'Low': 100 + np.cumsum(np.random.randn(500) * 0.5) - 0.5,
            'Close': 100 + np.cumsum(np.random.randn(500) * 0.5),
            'Volume': np.random.randint(1000, 10000, 500)
        }, index=dates)
        
        # Test regime detector
        detector = RegimeDetector(lookback=100)
        regime = detector.detect_regime(df)
        print(f"   ✅ Regime detection works (Hurst: {regime['hurst_exponent']:.2f})")
        results["passed"] += 1
        
        # Test ensemble strategy
        ensemble = EnsembleStrategy()
        signals = ensemble.generate_signals(df)
        print(f"   ✅ Ensemble strategy generated {len(signals)} signal rows")
        results["passed"] += 1
        
    except Exception as e:
        print(f"   ❌ Strategy test failed: {e}")
        results["failed"] += 1
        
    # 5. Test adaptive sizer
    print("\n5. Testing adaptive position sizing...")
    try:
        sizer = AdaptiveSizer(
            method="kelly",
            max_drawdown=0.50,
            aggressive_mode=True
        )
        
        # Simulate some trades
        sizer.update(100, 1000)  # Win
        sizer.update(-50, 950)  # Loss
        sizer.update(75, 1025)  # Win
        
        size = sizer.get_size(volatility=0.25, probability=0.65)
        stats = sizer.get_stats()
        
        print(f"   ✅ Adaptive sizing works (recommended size: {size:.2f})")
        print(f"   ✅ Win rate: {stats['win_rate']:.1%}, Expectancy: ${stats['expectancy']:.2f}")
        results["passed"] += 1
        
    except Exception as e:
        print(f"   ❌ Sizing test failed: {e}")
        results["failed"] += 1
        
    # 6. Test trade filter
    print("\n6. Testing trade filter...")
    try:
        from openquant.trading.trade_filter import TradeFilter, TradeSignal, FilterResult
        
        filter_obj = TradeFilter()
        signal = TradeSignal(
            symbol="EURUSD",
            side="LONG",
            probability=0.65,
            features={}
        )
        
        result, reason = filter_obj.check_trade(signal, df)
        print(f"   ✅ Trade filter works (result: {result.value})")
        results["passed"] += 1
        
    except Exception as e:
        print(f"   ❌ Filter test failed: {e}")
        results["failed"] += 1
        
    # 7. Test overfitting guard
    print("\n7. Testing overfitting guard...")
    try:
        guard = OverfittingGuard()
        
        # Simulate returns
        good_returns = pd.Series(np.random.randn(100) * 0.02 + 0.001)
        
        check = guard.check_strategy(
            returns=good_returns,
            is_sharpe=1.5,
            oos_sharpe=1.2,
            n_trials=20
        )
        
        print(f"   ✅ Overfitting guard works (is_safe: {check.is_safe})")
        print(f"      Reason: {check.reason}")
        if check.metrics:
            print(f"      Sharpe: {check.metrics.get('sharpe', 0):.2f}")
        results["passed"] += 1
        
    except Exception as e:
        print(f"   ❌ Overfitting guard test failed: {e}")
        results["failed"] += 1
        
    # 8. Test symbol selector
    print("\n8. Testing symbol selector...")
    try:
        from openquant.trading.symbol_selector import SymbolSelector
        
        selector = SymbolSelector()
        metrics = selector.analyze_symbol("EURUSD", df, spread_bps=2.0)
        
        print(f"   ✅ Symbol selector works")
        print(f"      ADX: {metrics.adx:.1f}, Volatility: {metrics.volatility:.2%}, Score: {metrics.score:.1f}")
        results["passed"] += 1
        
    except Exception as e:
        print(f"   ❌ Symbol selector test failed: {e}")
        results["failed"] += 1
        
    # 9. Test smart executor
    print("\n9. Testing smart executor...")
    try:
        from openquant.trading.smart_executor import SmartExecutor, OrderType, ExecutionConfig
        
        config = ExecutionConfig(order_type=OrderType.LIMIT)
        # Just test config creation (can't test execution without broker)
        
        print(f"   ✅ Smart executor config works (type: {config.order_type.value})")
        print(f"      TWAP slices: {config.twap_slices}, Retries: {config.max_retries}")
        results["passed"] += 1
        
    except Exception as e:
        print(f"   ❌ Smart executor test failed: {e}")
        results["failed"] += 1
        
    # 10. Test session optimizer
    print("\n10. Testing session optimizer...")
    try:
        from openquant.trading.session_optimizer import SessionOptimizer
        
        optimizer = SessionOptimizer()
        session, quality = optimizer.get_current_session()
        should_trade, reason = optimizer.should_trade_now()
        schedule = optimizer.get_recommended_schedule()
        
        print(f"   ✅ Session optimizer works")
        print(f"      Current: {session} ({quality:.0%})")
        print(f"      Should trade: {should_trade}")
        print(f"      Best hours: {schedule['best_hours_utc']}")
        results["passed"] += 1
        
    except Exception as e:
        print(f"   ❌ Session optimizer test failed: {e}")
        results["failed"] += 1
        
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Passed:   {results['passed']}")
    print(f"⚠️  Warnings: {results['warnings']}")
    print(f"❌ Failed:   {results['failed']}")
    print()
    
    if results["failed"] == 0:
        print("🎉 All critical checks passed! Robot is ready for paper trading.")
    else:
        print("⚠️  Some checks failed. Review errors before going live.")
        
    return results

if __name__ == "__main__":
    validate_strategy()
