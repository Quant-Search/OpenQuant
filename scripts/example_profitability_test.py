"""Example usage of the Profitability Testing Framework.

This script demonstrates different ways to use the profitability testing framework
for strategy validation before production deployment.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from test_profitability import ProfitabilityTester


def example_basic_test():
    """Basic profitability test with default settings."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Profitability Test")
    print("=" * 80 + "\n")
    
    tester = ProfitabilityTester(
        strategy_name="stat_arb",
        symbols=["BTC/USDT"],
        timeframe="1h"
    )
    
    report = tester.run_profitability_test()
    tester.save_report(report)
    
    return report


def example_custom_targets():
    """Test with custom return and drawdown targets."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Custom Targets (60% return, 20% max drawdown)")
    print("=" * 80 + "\n")
    
    tester = ProfitabilityTester(
        strategy_name="kalman",
        symbols=["ETH/USDT"],
        timeframe="4h",
        return_target=0.60,
        max_drawdown_constraint=0.20
    )
    
    report = tester.run_profitability_test()
    tester.save_report(report)
    
    return report


def example_high_confidence_test():
    """Test with more Monte Carlo runs for higher confidence."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: High Confidence Test (1000 Monte Carlo runs)")
    print("=" * 80 + "\n")
    
    tester = ProfitabilityTester(
        strategy_name="hurst",
        symbols=["BTC/USDT"],
        timeframe="1h",
        monte_carlo_runs=1000,
        min_years=3.5
    )
    
    report = tester.run_profitability_test()
    tester.save_report(report)
    
    return report


def example_multi_strategy_test():
    """Test multiple strategies and compare."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Multi-Strategy Comparison")
    print("=" * 80 + "\n")
    
    strategies = ["kalman", "hurst", "stat_arb"]
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy}...")
        print("-" * 80)
        
        tester = ProfitabilityTester(
            strategy_name=strategy,
            symbols=["BTC/USDT"],
            timeframe="1h",
            monte_carlo_runs=500
        )
        
        try:
            report = tester.run_profitability_test()
            tester.save_report(report)
            results[strategy] = report
        except Exception as e:
            print(f"Failed to test {strategy}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<15} {'Return':<10} {'Sharpe':<10} {'Max DD':<10} {'Confidence':<12} {'Result'}")
    print("-" * 80)
    
    for strategy, report in results.items():
        oos = report.out_of_sample_metrics
        rec = "GO" if report.confidence_score >= 60 else "NO GO"
        print(f"{strategy:<15} {oos.total_return:>8.1%} {oos.sharpe_ratio:>9.2f} "
              f"{abs(oos.max_drawdown):>9.1%} {report.confidence_score:>10.1f}/100 {rec}")
    
    return results


def example_production_validation():
    """Example of full production validation workflow."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Production Validation Workflow")
    print("=" * 80 + "\n")
    
    strategy_name = "stat_arb"
    symbol = "BTC/USDT"
    
    print("Step 1: Quick test with minimal data")
    print("-" * 80)
    quick_tester = ProfitabilityTester(
        strategy_name=strategy_name,
        symbols=[symbol],
        timeframe="4h",
        monte_carlo_runs=100,
        min_years=2.0
    )
    
    try:
        quick_report = quick_tester.run_profitability_test()
    except Exception as e:
        print(f"Quick test failed: {e}")
        print("Strategy is not ready for further testing.")
        return None
    
    if quick_report.confidence_score < 40:
        print("\nQuick test failed. Strategy needs fundamental improvements.")
        return quick_report
    
    print("\n\nStep 2: Full validation with comprehensive testing")
    print("-" * 80)
    full_tester = ProfitabilityTester(
        strategy_name=strategy_name,
        symbols=[symbol],
        timeframe="1h",
        monte_carlo_runs=500,
        min_years=3.0,
        return_target=0.50,
        max_drawdown_constraint=0.25
    )
    
    try:
        full_report = full_tester.run_profitability_test()
        full_tester.save_report(full_report)
    except Exception as e:
        print(f"Full test failed: {e}")
        return quick_report
    
    print("\n\nStep 3: Decision")
    print("-" * 80)
    if full_report.confidence_score >= 70:
        print("✓ Strategy APPROVED for production deployment")
        print("  Next steps: Deploy to paper trading, then live with small allocation")
    elif full_report.confidence_score >= 60:
        print("⚠ Strategy CONDITIONALLY APPROVED")
        print("  Next steps: Extended paper trading period recommended")
    elif full_report.confidence_score >= 50:
        print("⚠ Strategy needs paper trading validation")
        print("  Next steps: Run in paper trading for 1-2 months")
    else:
        print("✗ Strategy NOT APPROVED")
        print("  Next steps: Optimize parameters or redesign strategy")
    
    return full_report


def main():
    """Run all examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Profitability Testing Examples")
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific example (1-5)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all examples"
    )
    
    args = parser.parse_args()
    
    examples = {
        1: ("Basic Test", example_basic_test),
        2: ("Custom Targets", example_custom_targets),
        3: ("High Confidence", example_high_confidence_test),
        4: ("Multi-Strategy", example_multi_strategy_test),
        5: ("Production Validation", example_production_validation)
    }
    
    if args.all:
        for num, (name, func) in examples.items():
            print(f"\n\nRunning Example {num}: {name}")
            try:
                func()
            except Exception as e:
                print(f"Example {num} failed: {e}")
                continue
    elif args.example:
        name, func = examples[args.example]
        print(f"\nRunning Example {args.example}: {name}")
        func()
    else:
        print("\nProfitability Testing Framework - Examples\n")
        print("Available examples:")
        for num, (name, _) in examples.items():
            print(f"  {num}. {name}")
        print("\nUsage:")
        print("  python example_profitability_test.py --example 1")
        print("  python example_profitability_test.py --all")


if __name__ == "__main__":
    main()
