"""Example: Performance Attribution Integration.

Demonstrates how to integrate performance attribution into a trading workflow.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.analysis.attribution import PerformanceAttributor, quick_attribution
from openquant.storage.audit_trail import AuditTrail
from openquant.analysis.tca import TCAMonitor


def example_basic_attribution():
    """Basic attribution analysis example."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Attribution Analysis")
    print("="*60)
    
    result = quick_attribution(days=30)
    
    print("\nAttribution complete!")
    print(f"Total return decomposed into {len(result.details)} components")


def example_strategy_comparison():
    """Compare multiple strategies."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Strategy Comparison")
    print("="*60)
    
    attributor = PerformanceAttributor()
    comparison = attributor.compare_strategies(days=30)
    
    if comparison:
        print("\nStrategy Performance:")
        for strategy, metrics in comparison.items():
            print(f"\n{strategy}:")
            print(f"  Return: {metrics['total_return']:+.2%}")
            print(f"  Win Rate: {metrics['win_rate']:.1%}")
            print(f"  Trades: {metrics['num_trades']}")
    else:
        print("\nNo strategies found in the last 30 days.")


def example_trade_level_analysis():
    """Analyze individual trades."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Trade-Level Analysis")
    print("="*60)
    
    attributor = PerformanceAttributor()
    trades = attributor.get_trade_level_attribution(days=7)
    
    if trades:
        print(f"\nAnalyzing {len(trades)} recent trades...")
        
        avg_timing = sum(t.timing_quality for t in trades) / len(trades)
        avg_holding = sum(t.holding_period_hours for t in trades) / len(trades)
        
        print(f"Average Timing Quality: {avg_timing:.1%}")
        print(f"Average Holding Period: {avg_holding:.1f} hours")
        
        profitable = [t for t in trades if t.pnl > 0]
        print(f"Profitable Trades: {len(profitable)} / {len(trades)}")
    else:
        print("\nNo trades found in the last 7 days.")


def example_full_report():
    """Generate comprehensive report."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Comprehensive Report Generation")
    print("="*60)
    
    attributor = PerformanceAttributor()
    report = attributor.generate_report(
        days=30,
        output_path="data/example_attribution_report.json"
    )
    
    print("\nReport generated with the following sections:")
    print(f"  - Period: {report['period']['start']} to {report['period']['end']}")
    print(f"  - Total Trades: {report['summary']['total_trades']}")
    print(f"  - Strategies Analyzed: {len(report['strategy_comparison'])}")
    print(f"  - Instruments Analyzed: {len(report['instrument_comparison'])}")
    
    overall = report['overall_attribution']
    print(f"\nOverall Performance:")
    print(f"  Total Return: {overall['total_return']:+.2%}")
    print(f"  Timing Effect: {overall['timing_effect']:+.2%}")
    print(f"  Selection Effect: {overall['selection_effect']:+.2%}")
    print(f"  Sizing Effect: {overall['sizing_effect']:+.2%}")
    print(f"  Cost Drag: {overall['cost_drag']:+.2%}")


def example_execution_quality():
    """Analyze execution quality metrics."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Execution Quality Analysis")
    print("="*60)
    
    audit = AuditTrail()
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=7)
    
    exec_metrics = audit.get_execution_quality_metrics(start_time, end_time)
    print("\nExecution Metrics:")
    print(f"  Decisions: {exec_metrics['total_decisions']}")
    print(f"  Executions: {exec_metrics['total_executions']}")
    print(f"  Execution Rate: {exec_metrics['execution_rate']:.1%}")
    
    lag_stats = audit.get_signal_to_execution_lag(start_time, end_time)
    if lag_stats['count'] > 0:
        print(f"\nSignal-to-Execution Lag:")
        print(f"  Average: {lag_stats['avg_lag_seconds']:.2f}s")
        print(f"  Median: {lag_stats['median_lag_seconds']:.2f}s")
        print(f"  Range: {lag_stats['min_lag_seconds']:.2f}s - {lag_stats['max_lag_seconds']:.2f}s")
    else:
        print("\nNo signal-to-execution lag data available.")


def example_custom_integration():
    """Custom integration with audit trail and TCA."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Custom Integration")
    print("="*60)
    
    audit = AuditTrail(db_path="data/custom_audit.duckdb")
    tca = TCAMonitor(db_path="data/custom_tca.duckdb")
    
    attributor = PerformanceAttributor(
        audit_trail=audit,
        tca_monitor=tca
    )
    
    print("\nCustom attributor initialized with:")
    print(f"  - Custom audit trail: data/custom_audit.duckdb")
    print(f"  - Custom TCA monitor: data/custom_tca.duckdb")
    
    result = attributor.analyze(days=7)
    print(f"\nAnalysis complete: {result.details.get('num_trades', 0)} trades analyzed")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("PERFORMANCE ATTRIBUTION EXAMPLES")
    print("="*70)
    print("\nThese examples demonstrate the performance attribution module.")
    print("Note: Examples require trading data in the audit trail.")
    
    try:
        example_basic_attribution()
    except Exception as e:
        print(f"\nExample 1 error: {e}")
    
    try:
        example_strategy_comparison()
    except Exception as e:
        print(f"\nExample 2 error: {e}")
    
    try:
        example_trade_level_analysis()
    except Exception as e:
        print(f"\nExample 3 error: {e}")
    
    try:
        example_full_report()
    except Exception as e:
        print(f"\nExample 4 error: {e}")
    
    try:
        example_execution_quality()
    except Exception as e:
        print(f"\nExample 5 error: {e}")
    
    try:
        example_custom_integration()
    except Exception as e:
        print(f"\nExample 6 error: {e}")
    
    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70)
    print("\nFor more details, see:")
    print("  - openquant/analysis/README_ATTRIBUTION.md")
    print("  - scripts/run_attribution_analysis.py")
    print()


if __name__ == "__main__":
    main()
