"""Performance Attribution Analysis Script.

Demonstrates comprehensive performance attribution capabilities.

Usage:
    python scripts/run_attribution_analysis.py --days 30
    python scripts/run_attribution_analysis.py --strategy kalman --days 7
    python scripts/run_attribution_analysis.py --report
"""
import argparse
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.analysis.attribution import PerformanceAttributor, quick_attribution
from openquant.storage.audit_trail import AuditTrail
from openquant.analysis.tca import TCAMonitor
from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Performance Attribution Analysis")
    parser.add_argument("--days", type=int, default=30, help="Number of days to analyze")
    parser.add_argument("--strategy", type=str, help="Filter by strategy name")
    parser.add_argument("--symbol", type=str, help="Filter by symbol")
    parser.add_argument("--report", action="store_true", help="Generate full report")
    parser.add_argument("--compare-strategies", action="store_true", help="Compare strategies")
    parser.add_argument("--compare-instruments", action="store_true", help="Compare instruments")
    parser.add_argument("--trade-level", action="store_true", help="Show trade-level attribution")
    parser.add_argument("--output", type=str, default="data/attribution_report.json", 
                        help="Output path for report")
    
    args = parser.parse_args()
    
    LOGGER.info("Initializing Performance Attributor...")
    attributor = PerformanceAttributor()
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=args.days)
    
    if args.report:
        LOGGER.info("Generating comprehensive attribution report...")
        report = attributor.generate_report(
            start_time=start_time,
            end_time=end_time,
            days=args.days,
            output_path=args.output
        )
        print("\n" + "="*60)
        print("COMPREHENSIVE ATTRIBUTION REPORT")
        print("="*60)
        print(f"\nReport saved to: {args.output}")
        print(f"\nPeriod: {start_time.date()} to {end_time.date()}")
        print(f"Total Trades: {report['summary']['total_trades']}")
        print(f"Profitable Trades: {report['summary']['profitable_trades']}")
        print(f"Average Timing Quality: {report['summary']['average_timing_quality']:.2%}")
        print(f"Average Holding Period: {report['summary']['average_holding_hours']:.1f} hours")
        
        overall = report['overall_attribution']
        print(f"\nTotal Return: {overall['total_return']:+.2%}")
        print(f"  Timing Effect: {overall['timing_effect']:+.2%}")
        print(f"  Selection Effect: {overall['selection_effect']:+.2%}")
        print(f"  Sizing Effect: {overall['sizing_effect']:+.2%}")
        print(f"  Cost Drag: {overall['cost_drag']:+.2%}")
        print(f"  Residual: {overall['residual']:+.2%}")
        
    elif args.compare_strategies:
        LOGGER.info("Comparing strategy performance...")
        comparison = attributor.compare_strategies(
            start_time=start_time,
            end_time=end_time,
            days=args.days
        )
        
        print("\n" + "="*60)
        print("STRATEGY PERFORMANCE COMPARISON")
        print("="*60)
        
        for strategy, metrics in comparison.items():
            print(f"\n{strategy.upper()}")
            print("-" * 40)
            print(f"  Total Return: {metrics['total_return']:+.2%}")
            print(f"  Timing Effect: {metrics['timing_effect']:+.2%}")
            print(f"  Sizing Effect: {metrics['sizing_effect']:+.2%}")
            print(f"  Cost Drag: {metrics['cost_drag']:+.2%}")
            print(f"  Trades: {metrics['num_trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.1%}")
            print(f"  Avg P&L: ${metrics['avg_pnl']:,.2f}")
            print(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
            
    elif args.compare_instruments:
        LOGGER.info("Comparing instrument performance...")
        comparison = attributor.compare_instruments(
            start_time=start_time,
            end_time=end_time,
            days=args.days
        )
        
        print("\n" + "="*60)
        print("INSTRUMENT PERFORMANCE COMPARISON")
        print("="*60)
        
        for symbol, metrics in sorted(comparison.items(), 
                                     key=lambda x: x[1]['total_pnl'], 
                                     reverse=True):
            print(f"\n{symbol}")
            print("-" * 40)
            print(f"  Total Return: {metrics['total_return']:+.2%}")
            print(f"  Timing Effect: {metrics['timing_effect']:+.2%}")
            print(f"  Trades: {metrics['num_trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.1%}")
            print(f"  Avg Holding: {metrics['avg_holding_hours']:.1f} hours")
            print(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
            
    elif args.trade_level:
        LOGGER.info("Getting trade-level attribution...")
        trades = attributor.get_trade_level_attribution(
            start_time=start_time,
            end_time=end_time,
            days=args.days
        )
        
        print("\n" + "="*60)
        print("TRADE-LEVEL ATTRIBUTION")
        print("="*60)
        print(f"\nShowing {min(20, len(trades))} most recent trades")
        
        for i, trade in enumerate(trades[:20], 1):
            print(f"\n{i}. {trade.symbol} ({trade.strategy})")
            print(f"   Entry: {trade.entry_time.strftime('%Y-%m-%d %H:%M')} @ ${trade.entry_price:.4f}")
            print(f"   Exit:  {trade.exit_time.strftime('%Y-%m-%d %H:%M')} @ ${trade.exit_price:.4f}")
            print(f"   P&L: {trade.pnl_pct:+.2%} (${trade.pnl:+,.2f})")
            print(f"   Timing Quality: {trade.timing_quality:.1%}")
            print(f"   Holding Period: {trade.holding_period_hours:.1f} hours")
            print(f"   Cost Impact: {trade.cost_impact:+.2%}")
            
    else:
        LOGGER.info(f"Running attribution analysis for last {args.days} days...")
        result = attributor.analyze(
            start_time=start_time,
            end_time=end_time,
            days=args.days
        )
        
        print("\n" + result.summary())
        print("\n" + "="*60)
        print("DETAILS")
        print("="*60)
        print(f"Number of trades: {result.details['num_trades']}")
        print(f"Symbols traded: {', '.join(result.details['symbols_traded'])}")
        print(f"Strategies used: {', '.join(result.details['strategies_used'])}")
        print(f"Total volume: ${result.details['total_volume']:,.2f}")
        
    LOGGER.info("Attribution analysis complete!")
    
    audit = AuditTrail()
    exec_quality = audit.get_execution_quality_metrics(start_time, end_time)
    print("\n" + "="*60)
    print("EXECUTION QUALITY METRICS")
    print("="*60)
    print(f"Total Decisions: {exec_quality['total_decisions']}")
    print(f"Total Executions: {exec_quality['total_executions']}")
    print(f"Execution Rate: {exec_quality['execution_rate']:.1%}")
    
    lag_stats = audit.get_signal_to_execution_lag(start_time, end_time)
    print("\n" + "="*60)
    print("SIGNAL-TO-EXECUTION LAG")
    print("="*60)
    print(f"Count: {lag_stats['count']}")
    if lag_stats['count'] > 0:
        print(f"Average: {lag_stats['avg_lag_seconds']:.2f}s")
        print(f"Median: {lag_stats['median_lag_seconds']:.2f}s")
        print(f"Min: {lag_stats['min_lag_seconds']:.2f}s")
        print(f"Max: {lag_stats['max_lag_seconds']:.2f}s")


if __name__ == "__main__":
    main()
