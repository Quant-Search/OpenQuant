"""Example script for monitoring order execution quality.

Demonstrates how to use ExecutionQualityMonitor to track fill rates,
rejection rates, slippage, and get alerts for execution degradation.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.analysis.execution_quality import ExecutionQualityMonitor
from openquant.analysis.tca import TCAMonitor
import json


def main():
    """Run execution quality monitoring."""
    print("=" * 80)
    print("Execution Quality Monitor")
    print("=" * 80)
    print()
    
    # Initialize monitor with custom thresholds
    monitor = ExecutionQualityMonitor(
        db_path="data/tca.duckdb",
        fill_rate_threshold=0.95,           # 95% minimum fill rate
        rejection_rate_threshold=0.05,      # 5% maximum rejection rate
        slippage_threshold_bps=10.0,        # 10 bps maximum average slippage
        slippage_std_threshold_bps=20.0,    # 20 bps maximum slippage volatility
        lookback_hours=24                   # Look back 24 hours
    )
    
    # Run full monitoring cycle
    print("Running execution quality monitoring...")
    metrics, alerts = monitor.monitor(save_snapshot=True)
    
    # Display current metrics
    print("\n" + "=" * 80)
    print("CURRENT EXECUTION METRICS (Last 24 hours)")
    print("=" * 80)
    print(f"Total Orders:        {metrics.total_orders}")
    print(f"Filled Orders:       {metrics.filled_orders}")
    print(f"Rejected Orders:     {metrics.rejected_orders}")
    print(f"Partial Fills:       {metrics.partial_fills}")
    print()
    print(f"Fill Rate:           {metrics.fill_rate:.2%}")
    print(f"Rejection Rate:      {metrics.rejection_rate:.2%}")
    print(f"Partial Fill Rate:   {metrics.partial_fill_rate:.2%}")
    print()
    print(f"Avg Slippage:        {metrics.avg_slippage_bps:.2f} bps")
    print(f"Median Slippage:     {metrics.median_slippage_bps:.2f} bps")
    print(f"Slippage Std Dev:    {metrics.slippage_std_bps:.2f} bps")
    print(f"Best Slippage:       {metrics.best_slippage_bps:.2f} bps")
    print(f"Worst Slippage:      {metrics.worst_slippage_bps:.2f} bps")
    print()
    print(f"Avg Fill Time:       {metrics.avg_fill_time_ms:.2f} ms")
    print(f"Total Fees:          ${metrics.total_fees:.2f}")
    print(f"Avg Fee:             {metrics.avg_fee_bps:.2f} bps")
    
    # Display alerts
    print("\n" + "=" * 80)
    print(f"EXECUTION QUALITY ALERTS ({len(alerts)} found)")
    print("=" * 80)
    if alerts:
        for i, alert in enumerate(alerts, 1):
            print(f"\nAlert #{i}:")
            print(f"  Severity:    {alert.severity.upper()}")
            print(f"  Type:        {alert.alert_type}")
            print(f"  Metric:      {alert.metric}")
            print(f"  Current:     {alert.current_value:.4f}")
            print(f"  Threshold:   {alert.threshold_value:.4f}")
            print(f"  Message:     {alert.message}")
            if alert.details:
                print(f"  Details:     {alert.details}")
    else:
        print("No alerts - execution quality within acceptable parameters.")
    
    # Get slippage distribution
    print("\n" + "=" * 80)
    print("SLIPPAGE DISTRIBUTION")
    print("=" * 80)
    slippage_dist = monitor.get_slippage_distribution()
    if slippage_dist["count"] > 0:
        print(f"Sample Size:  {slippage_dist['count']} orders")
        print(f"Mean:         {slippage_dist['mean']:.2f} bps")
        print(f"Median:       {slippage_dist['median']:.2f} bps")
        print(f"Std Dev:      {slippage_dist['std']:.2f} bps")
        print(f"Min:          {slippage_dist['min']:.2f} bps")
        print(f"Max:          {slippage_dist['max']:.2f} bps")
        print()
        print("Percentiles:")
        for pct, val in slippage_dist['percentiles'].items():
            print(f"  {pct.upper():6s}: {val:.2f} bps")
    else:
        print("No slippage data available.")
    
    # Compare to benchmark
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON (vs. last week)")
    print("=" * 80)
    benchmark = monitor.compare_to_benchmark(metrics, benchmark_lookback_hours=168)
    if benchmark.get("has_benchmark"):
        print("Benchmark Period: Last 7 days")
        print()
        print(f"Fill Rate:           {benchmark['current']['fill_rate']:.2%} "
              f"(Δ {benchmark['deltas']['fill_rate']:+.2%}, "
              f"{benchmark['percent_changes']['fill_rate']:+.1f}%)")
        print(f"Rejection Rate:      {benchmark['current']['rejection_rate']:.2%} "
              f"(Δ {benchmark['deltas']['rejection_rate']:+.2%}, "
              f"{benchmark['percent_changes']['rejection_rate']:+.1f}%)")
        print(f"Avg Slippage:        {benchmark['current']['avg_slippage_bps']:.2f} bps "
              f"(Δ {benchmark['deltas']['avg_slippage_bps']:+.2f} bps, "
              f"{benchmark['percent_changes']['avg_slippage']:+.1f}%)")
        print(f"Slippage Std Dev:    {benchmark['current']['slippage_std_bps']:.2f} bps "
              f"(Δ {benchmark['deltas']['slippage_std_bps']:+.2f} bps)")
    else:
        print(f"Insufficient historical data: {benchmark.get('message', 'Unknown reason')}")
    
    # Get recent alerts from database
    print("\n" + "=" * 80)
    print("RECENT ALERTS (Last 24 hours)")
    print("=" * 80)
    recent_alerts = monitor.get_recent_alerts(lookback_hours=24)
    if recent_alerts:
        critical_count = sum(1 for a in recent_alerts if a.get("severity") == "critical")
        warning_count = sum(1 for a in recent_alerts if a.get("severity") == "warning")
        print(f"Total Alerts:    {len(recent_alerts)}")
        print(f"Critical:        {critical_count}")
        print(f"Warnings:        {warning_count}")
        print()
        for i, alert in enumerate(recent_alerts[:5], 1):  # Show top 5
            print(f"{i}. [{alert['severity'].upper()}] {alert['alert_type']}: {alert['message']}")
    else:
        print("No recent alerts in database.")
    
    # Generate comprehensive report
    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE REPORT")
    print("=" * 80)
    report = monitor.get_summary_report()
    report_path = "data/execution_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved to: {report_path}")
    
    # Show TCA stats for comparison
    print("\n" + "=" * 80)
    print("TCA MONITOR COMPARISON")
    print("=" * 80)
    tca = TCAMonitor(db_path="data/tca.duckdb")
    tca_stats = tca.get_stats()
    if tca_stats.get("count", 0) > 0:
        print(f"Total Filled Orders: {tca_stats['count']}")
        print(f"Avg Slippage:        {tca_stats['avg_slippage_bps']:.2f} bps")
        print(f"Total Fees:          ${tca_stats['total_fees']:.2f}")
        print(f"Worst Slippage:      {tca_stats['worst_slippage']:.2f} bps")
    else:
        print("No TCA data available.")
    
    print("\n" + "=" * 80)
    print("Monitoring complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
