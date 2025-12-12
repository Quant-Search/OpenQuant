"""Example of integrating ExecutionQualityMonitor with trading workflow.

This demonstrates:
1. Using TCAMonitor to track order execution
2. Using ExecutionQualityMonitor to analyze and alert on quality metrics
3. Periodic monitoring in a trading loop
"""
import sys
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.analysis.tca import TCAMonitor
from openquant.analysis.execution_quality import ExecutionQualityMonitor


def simulate_trading_session():
    """Simulate a trading session with order execution tracking."""
    print("=" * 80)
    print("SIMULATED TRADING SESSION WITH EXECUTION QUALITY MONITORING")
    print("=" * 80)
    print()
    
    # Initialize monitors
    tca = TCAMonitor(db_path="data/tca.duckdb")
    eq_monitor = ExecutionQualityMonitor(
        db_path="data/tca.duckdb",
        fill_rate_threshold=0.95,
        rejection_rate_threshold=0.05,
        slippage_threshold_bps=10.0,
        lookback_hours=1  # Short lookback for demo
    )
    
    print("Monitors initialized")
    print("  - TCAMonitor: Tracking order executions")
    print("  - ExecutionQualityMonitor: Analyzing quality metrics")
    print()
    
    # Simulate some orders (in real system, these would come from broker)
    print("Simulating order executions...")
    print("-" * 80)
    
    orders = [
        # (order_id, symbol, side, quantity, arrival_price, fill_price, fill_qty, fee, status)
        ("ORD001", "BTC/USD", "buy", 0.1, 50000.0, 50005.0, 0.1, 2.5, "FILLED"),
        ("ORD002", "ETH/USD", "buy", 1.0, 3000.0, 3001.0, 1.0, 1.5, "FILLED"),
        ("ORD003", "BTC/USD", "sell", 0.05, 50010.0, 50005.0, 0.05, 1.25, "FILLED"),
        ("ORD004", "SOL/USD", "buy", 10.0, 100.0, 100.2, 10.0, 0.5, "FILLED"),
        ("ORD005", "AAPL", "buy", 100.0, 150.0, 150.05, 100.0, 7.5, "FILLED"),
        ("ORD006", "BTC/USD", "buy", 0.2, 50020.0, None, 0.0, 0.0, "REJECTED"),
        ("ORD007", "ETH/USD", "sell", 2.0, 3005.0, 3004.0, 2.0, 3.0, "FILLED"),
        ("ORD008", "MSFT", "buy", 50.0, 300.0, 300.1, 50.0, 7.5, "FILLED"),
    ]
    
    for order_data in orders:
        order_id, symbol, side, qty, arrival, fill, fill_qty, fee, status = order_data
        
        # Log order submission
        tca.log_order(order_id, symbol, side, qty, arrival)
        print(f"  Submitted: {order_id} {side.upper()} {qty} {symbol} @ {arrival}")
        
        # Simulate execution delay
        time.sleep(0.1)
        
        # Update with fill if executed
        if status == "FILLED" and fill is not None:
            tca.update_fill(order_id, fill, fill_qty, fee)
            slippage_bps = abs(fill - arrival) / arrival * 10000
            print(f"    -> Filled @ {fill} (slippage: {slippage_bps:.2f} bps)")
        else:
            print(f"    -> {status}")
        
        print()
    
    # Run execution quality analysis
    print("\n" + "=" * 80)
    print("EXECUTION QUALITY ANALYSIS")
    print("=" * 80)
    
    # Calculate metrics
    metrics = eq_monitor.calculate_metrics()
    
    print(f"\nOrder Statistics:")
    print(f"  Total Orders:    {metrics.total_orders}")
    print(f"  Filled:          {metrics.filled_orders} ({metrics.fill_rate:.1%})")
    print(f"  Rejected:        {metrics.rejected_orders} ({metrics.rejection_rate:.1%})")
    print(f"  Partial Fills:   {metrics.partial_fills}")
    
    print(f"\nSlippage Analysis:")
    print(f"  Average:         {metrics.avg_slippage_bps:.2f} bps")
    print(f"  Median:          {metrics.median_slippage_bps:.2f} bps")
    print(f"  Std Dev:         {metrics.slippage_std_bps:.2f} bps")
    print(f"  Best:            {metrics.best_slippage_bps:.2f} bps")
    print(f"  Worst:           {metrics.worst_slippage_bps:.2f} bps")
    
    print(f"\nCosts:")
    print(f"  Total Fees:      ${metrics.total_fees:.2f}")
    print(f"  Avg Fee:         {metrics.avg_fee_bps:.2f} bps")
    
    # Check for alerts
    print("\n" + "=" * 80)
    print("CHECKING FOR EXECUTION QUALITY DEGRADATION")
    print("=" * 80)
    
    alerts = eq_monitor.check_for_degradation(metrics)
    
    if alerts:
        print(f"\n⚠️  {len(alerts)} alert(s) detected:\n")
        for i, alert in enumerate(alerts, 1):
            icon = "🔴" if alert.severity == "critical" else "🟡"
            print(f"{icon} Alert #{i}: {alert.alert_type}")
            print(f"   Severity: {alert.severity.upper()}")
            print(f"   {alert.message}")
            print()
    else:
        print("\n✅ No alerts - execution quality within acceptable parameters")
    
    # Get slippage distribution
    print("\n" + "=" * 80)
    print("SLIPPAGE DISTRIBUTION")
    print("=" * 80)
    
    slippage_dist = eq_monitor.get_slippage_distribution()
    if slippage_dist["count"] > 0:
        print(f"\nPercentiles (from {slippage_dist['count']} orders):")
        for pct, val in slippage_dist['percentiles'].items():
            print(f"  {pct.upper():6s}: {val:8.2f} bps")
    
    # Save snapshot
    print("\n" + "=" * 80)
    print("SAVING METRICS SNAPSHOT")
    print("=" * 80)
    
    eq_monitor.save_snapshot(metrics)
    print("✅ Metrics snapshot saved to database")
    
    # Save any alerts
    for alert in alerts:
        eq_monitor.save_alert(alert)
    print(f"✅ {len(alerts)} alert(s) saved to database")
    
    print("\n" + "=" * 80)
    print("PERIODIC MONITORING WORKFLOW")
    print("=" * 80)
    print("\nIn a production system, you would:")
    print("  1. Call tca.log_order() when submitting orders")
    print("  2. Call tca.update_fill() when orders are filled")
    print("  3. Run eq_monitor.monitor() periodically (e.g., every hour)")
    print("  4. React to critical alerts (e.g., pause trading, alert operators)")
    print("  5. Use eq_monitor.get_summary_report() for daily/weekly reviews")
    print()
    print("Example periodic monitoring:")
    print("  # Every hour")
    print("  metrics, alerts = eq_monitor.monitor(save_snapshot=True)")
    print("  for alert in [a for a in alerts if a.severity == 'critical']:")
    print("      notify_operators(alert)")
    print("      if should_pause_trading(alert):")
    print("          pause_trading_system()")
    print()


def demonstrate_broker_integration():
    """Show how to integrate with actual broker implementations."""
    print("\n" + "=" * 80)
    print("BROKER INTEGRATION EXAMPLE")
    print("=" * 80)
    print()
    print("The Alpaca broker already integrates TCAMonitor:")
    print()
    print("  from openquant.broker.alpaca_broker import AlpacaBroker")
    print("  from openquant.analysis.execution_quality import ExecutionQualityMonitor")
    print()
    print("  # Initialize broker (includes TCA)")
    print("  broker = AlpacaBroker()")
    print()
    print("  # Initialize execution quality monitor")
    print("  eq_monitor = ExecutionQualityMonitor()")
    print()
    print("  # Place orders (TCA automatically logs them)")
    print("  order = broker.place_order('AAPL', 100, 'buy')")
    print()
    print("  # Periodically sync TCA and check quality")
    print("  broker.sync_tca()  # Updates TCA with filled orders")
    print("  metrics, alerts = eq_monitor.monitor()")
    print()
    print("  # Handle alerts")
    print("  for alert in alerts:")
    print("      if alert.severity == 'critical':")
    print("          logger.critical(alert.message)")
    print("          # Take action based on alert type")
    print()


if __name__ == "__main__":
    simulate_trading_session()
    demonstrate_broker_integration()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print("\nCheck the following for results:")
    print("  - data/tca.duckdb (order execution data)")
    print("  - Run 'python scripts/monitor_execution_quality.py' for full report")
    print()
