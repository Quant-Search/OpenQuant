"""Order Execution Quality Monitoring.

Tracks fill rates, rejection rates, slippage distribution, and compares against
TCAMonitor benchmarks. Provides alerts for execution quality degradation.
"""
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class ExecutionQualityMetrics:
    """Container for execution quality metrics."""
    fill_rate: float = 0.0
    rejection_rate: float = 0.0
    partial_fill_rate: float = 0.0
    avg_slippage_bps: float = 0.0
    median_slippage_bps: float = 0.0
    slippage_std_bps: float = 0.0
    worst_slippage_bps: float = 0.0
    best_slippage_bps: float = 0.0
    avg_fill_time_ms: float = 0.0
    total_orders: int = 0
    filled_orders: int = 0
    rejected_orders: int = 0
    partial_fills: int = 0
    total_fees: float = 0.0
    avg_fee_bps: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ExecutionAlert:
    """Alert for execution quality degradation."""
    timestamp: datetime
    alert_type: str
    severity: str  # 'warning', 'critical'
    metric: str
    current_value: float
    threshold_value: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class ExecutionQualityMonitor:
    """Monitor order execution quality and alert on degradation."""
    
    def __init__(
        self,
        db_path: str = "data/tca.duckdb",
        fill_rate_threshold: float = 0.95,
        rejection_rate_threshold: float = 0.05,
        slippage_threshold_bps: float = 10.0,
        slippage_std_threshold_bps: float = 20.0,
        lookback_hours: int = 24
    ):
        """Initialize execution quality monitor.
        
        Args:
            db_path: Path to TCA database
            fill_rate_threshold: Minimum acceptable fill rate (0-1)
            rejection_rate_threshold: Maximum acceptable rejection rate (0-1)
            slippage_threshold_bps: Maximum acceptable average slippage in bps
            slippage_std_threshold_bps: Maximum acceptable slippage std dev in bps
            lookback_hours: Hours to look back for metrics calculation
        """
        self.db_path = db_path
        self.fill_rate_threshold = fill_rate_threshold
        self.rejection_rate_threshold = rejection_rate_threshold
        self.slippage_threshold_bps = slippage_threshold_bps
        self.slippage_std_threshold_bps = slippage_std_threshold_bps
        self.lookback_hours = lookback_hours
        
        self._init_db()
        
    def _init_db(self):
        """Initialize execution quality database schema."""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(self.db_path) as con:
                # Extend TCA schema if needed
                con.execute("""
                    CREATE TABLE IF NOT EXISTS execution_alerts (
                        alert_id INTEGER PRIMARY KEY,
                        timestamp TIMESTAMP,
                        alert_type VARCHAR,
                        severity VARCHAR,
                        metric VARCHAR,
                        current_value DOUBLE,
                        threshold_value DOUBLE,
                        message VARCHAR,
                        details VARCHAR
                    )
                """)
                
                # Create execution quality snapshots table
                con.execute("""
                    CREATE TABLE IF NOT EXISTS execution_quality_snapshots (
                        snapshot_id INTEGER PRIMARY KEY,
                        timestamp TIMESTAMP,
                        fill_rate DOUBLE,
                        rejection_rate DOUBLE,
                        partial_fill_rate DOUBLE,
                        avg_slippage_bps DOUBLE,
                        median_slippage_bps DOUBLE,
                        slippage_std_bps DOUBLE,
                        worst_slippage_bps DOUBLE,
                        best_slippage_bps DOUBLE,
                        avg_fill_time_ms DOUBLE,
                        total_orders INTEGER,
                        filled_orders INTEGER,
                        rejected_orders INTEGER,
                        partial_fills INTEGER,
                        total_fees DOUBLE,
                        avg_fee_bps DOUBLE
                    )
                """)
        except Exception as e:
            LOGGER.error(f"Failed to init execution quality db: {e}")
            
    def calculate_metrics(
        self,
        lookback_hours: Optional[int] = None,
        symbol: Optional[str] = None
    ) -> ExecutionQualityMetrics:
        """Calculate execution quality metrics from TCA data.
        
        Args:
            lookback_hours: Hours to look back (default: use instance setting)
            symbol: Optional symbol filter
            
        Returns:
            ExecutionQualityMetrics object
        """
        lookback = lookback_hours or self.lookback_hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback)
        
        try:
            with duckdb.connect(self.db_path) as con:
                # Build query with optional symbol filter
                symbol_filter = f"AND symbol = '{symbol}'" if symbol else ""
                
                # Get order statistics
                query = f"""
                    SELECT 
                        COUNT(*) as total_orders,
                        SUM(CASE WHEN status = 'FILLED' THEN 1 ELSE 0 END) as filled_orders,
                        SUM(CASE WHEN status = 'REJECTED' OR status = 'CANCELLED' THEN 1 ELSE 0 END) as rejected_orders,
                        SUM(CASE WHEN status = 'FILLED' AND fill_qty < quantity THEN 1 ELSE 0 END) as partial_fills,
                        AVG(CASE WHEN status = 'FILLED' THEN slippage_bps ELSE NULL END) as avg_slippage_bps,
                        MEDIAN(CASE WHEN status = 'FILLED' THEN slippage_bps ELSE NULL END) as median_slippage_bps,
                        STDDEV(CASE WHEN status = 'FILLED' THEN slippage_bps ELSE NULL END) as slippage_std_bps,
                        MAX(CASE WHEN status = 'FILLED' THEN slippage_bps ELSE NULL END) as worst_slippage_bps,
                        MIN(CASE WHEN status = 'FILLED' THEN slippage_bps ELSE NULL END) as best_slippage_bps,
                        AVG(CASE WHEN status = 'FILLED' THEN 
                            EXTRACT(EPOCH FROM (filled_at - created_at)) * 1000 
                            ELSE NULL END) as avg_fill_time_ms,
                        SUM(CASE WHEN status = 'FILLED' THEN fee ELSE 0 END) as total_fees,
                        AVG(CASE WHEN status = 'FILLED' AND fill_price > 0 THEN 
                            (fee / (fill_qty * fill_price)) * 10000 
                            ELSE NULL END) as avg_fee_bps
                    FROM orders 
                    WHERE created_at >= ?
                    {symbol_filter}
                """
                
                result = con.execute(query, (cutoff_time,)).fetchone()
                
                if not result or result[0] == 0:
                    return ExecutionQualityMetrics()
                
                total_orders = int(result[0] or 0)
                filled_orders = int(result[1] or 0)
                rejected_orders = int(result[2] or 0)
                partial_fills = int(result[3] or 0)
                
                fill_rate = filled_orders / total_orders if total_orders > 0 else 0.0
                rejection_rate = rejected_orders / total_orders if total_orders > 0 else 0.0
                partial_fill_rate = partial_fills / filled_orders if filled_orders > 0 else 0.0
                
                return ExecutionQualityMetrics(
                    fill_rate=fill_rate,
                    rejection_rate=rejection_rate,
                    partial_fill_rate=partial_fill_rate,
                    avg_slippage_bps=float(result[4] or 0.0),
                    median_slippage_bps=float(result[5] or 0.0),
                    slippage_std_bps=float(result[6] or 0.0),
                    worst_slippage_bps=float(result[7] or 0.0),
                    best_slippage_bps=float(result[8] or 0.0),
                    avg_fill_time_ms=float(result[9] or 0.0),
                    total_orders=total_orders,
                    filled_orders=filled_orders,
                    rejected_orders=rejected_orders,
                    partial_fills=partial_fills,
                    total_fees=float(result[10] or 0.0),
                    avg_fee_bps=float(result[11] or 0.0)
                )
                
        except Exception as e:
            LOGGER.error(f"Failed to calculate execution metrics: {e}")
            return ExecutionQualityMetrics()
            
    def get_slippage_distribution(
        self,
        lookback_hours: Optional[int] = None,
        symbol: Optional[str] = None,
        bins: int = 20
    ) -> Dict[str, Any]:
        """Get slippage distribution statistics.
        
        Args:
            lookback_hours: Hours to look back
            symbol: Optional symbol filter
            bins: Number of histogram bins
            
        Returns:
            Dictionary with distribution statistics and histogram data
        """
        lookback = lookback_hours or self.lookback_hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback)
        
        try:
            with duckdb.connect(self.db_path) as con:
                symbol_filter = f"AND symbol = '{symbol}'" if symbol else ""
                
                query = f"""
                    SELECT slippage_bps 
                    FROM orders 
                    WHERE created_at >= ? 
                    AND status = 'FILLED'
                    AND slippage_bps IS NOT NULL
                    {symbol_filter}
                """
                
                df = con.execute(query, (cutoff_time,)).df()
                
                if df.empty:
                    return {
                        "count": 0,
                        "percentiles": {},
                        "histogram": {"bins": [], "counts": []}
                    }
                
                slippage = df["slippage_bps"].values
                
                # Calculate percentiles
                percentiles = {
                    "p5": float(np.percentile(slippage, 5)),
                    "p25": float(np.percentile(slippage, 25)),
                    "p50": float(np.percentile(slippage, 50)),
                    "p75": float(np.percentile(slippage, 75)),
                    "p95": float(np.percentile(slippage, 95)),
                    "p99": float(np.percentile(slippage, 99))
                }
                
                # Create histogram
                counts, bin_edges = np.histogram(slippage, bins=bins)
                
                return {
                    "count": len(slippage),
                    "mean": float(np.mean(slippage)),
                    "median": float(np.median(slippage)),
                    "std": float(np.std(slippage)),
                    "min": float(np.min(slippage)),
                    "max": float(np.max(slippage)),
                    "percentiles": percentiles,
                    "histogram": {
                        "bins": bin_edges.tolist(),
                        "counts": counts.tolist()
                    }
                }
                
        except Exception as e:
            LOGGER.error(f"Failed to get slippage distribution: {e}")
            return {"count": 0, "percentiles": {}, "histogram": {"bins": [], "counts": []}}
            
    def compare_to_benchmark(
        self,
        current_metrics: ExecutionQualityMetrics,
        benchmark_lookback_hours: int = 168  # 1 week
    ) -> Dict[str, Any]:
        """Compare current metrics to historical benchmark.
        
        Args:
            current_metrics: Current execution quality metrics
            benchmark_lookback_hours: Hours to use for benchmark calculation
            
        Returns:
            Dictionary with comparison results
        """
        benchmark_cutoff = datetime.now(timezone.utc) - timedelta(hours=benchmark_lookback_hours)
        current_cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        
        try:
            with duckdb.connect(self.db_path) as con:
                # Calculate benchmark from older data
                query = """
                    SELECT 
                        COUNT(*) as total_orders,
                        SUM(CASE WHEN status = 'FILLED' THEN 1 ELSE 0 END) as filled_orders,
                        SUM(CASE WHEN status = 'REJECTED' OR status = 'CANCELLED' THEN 1 ELSE 0 END) as rejected_orders,
                        AVG(CASE WHEN status = 'FILLED' THEN slippage_bps ELSE NULL END) as avg_slippage_bps,
                        STDDEV(CASE WHEN status = 'FILLED' THEN slippage_bps ELSE NULL END) as slippage_std_bps
                    FROM orders 
                    WHERE created_at >= ? AND created_at < ?
                """
                
                result = con.execute(query, (benchmark_cutoff, current_cutoff)).fetchone()
                
                if not result or result[0] == 0:
                    return {
                        "has_benchmark": False,
                        "message": "Insufficient historical data for benchmark"
                    }
                
                total_orders = int(result[0] or 0)
                filled_orders = int(result[1] or 0)
                rejected_orders = int(result[2] or 0)
                
                benchmark_fill_rate = filled_orders / total_orders if total_orders > 0 else 0.0
                benchmark_rejection_rate = rejected_orders / total_orders if total_orders > 0 else 0.0
                benchmark_avg_slippage = float(result[3] or 0.0)
                benchmark_slippage_std = float(result[4] or 0.0)
                
                # Calculate deltas
                fill_rate_delta = current_metrics.fill_rate - benchmark_fill_rate
                rejection_rate_delta = current_metrics.rejection_rate - benchmark_rejection_rate
                slippage_delta = current_metrics.avg_slippage_bps - benchmark_avg_slippage
                slippage_std_delta = current_metrics.slippage_std_bps - benchmark_slippage_std
                
                # Calculate relative changes
                fill_rate_pct_change = (fill_rate_delta / benchmark_fill_rate * 100) if benchmark_fill_rate > 0 else 0.0
                rejection_rate_pct_change = (rejection_rate_delta / benchmark_rejection_rate * 100) if benchmark_rejection_rate > 0 else 0.0
                slippage_pct_change = (slippage_delta / benchmark_avg_slippage * 100) if benchmark_avg_slippage != 0 else 0.0
                
                return {
                    "has_benchmark": True,
                    "benchmark": {
                        "fill_rate": benchmark_fill_rate,
                        "rejection_rate": benchmark_rejection_rate,
                        "avg_slippage_bps": benchmark_avg_slippage,
                        "slippage_std_bps": benchmark_slippage_std
                    },
                    "current": {
                        "fill_rate": current_metrics.fill_rate,
                        "rejection_rate": current_metrics.rejection_rate,
                        "avg_slippage_bps": current_metrics.avg_slippage_bps,
                        "slippage_std_bps": current_metrics.slippage_std_bps
                    },
                    "deltas": {
                        "fill_rate": fill_rate_delta,
                        "rejection_rate": rejection_rate_delta,
                        "avg_slippage_bps": slippage_delta,
                        "slippage_std_bps": slippage_std_delta
                    },
                    "percent_changes": {
                        "fill_rate": fill_rate_pct_change,
                        "rejection_rate": rejection_rate_pct_change,
                        "avg_slippage": slippage_pct_change
                    }
                }
                
        except Exception as e:
            LOGGER.error(f"Failed to compare to benchmark: {e}")
            return {"has_benchmark": False, "error": str(e)}
            
    def check_for_degradation(
        self,
        metrics: Optional[ExecutionQualityMetrics] = None
    ) -> List[ExecutionAlert]:
        """Check for execution quality degradation and generate alerts.
        
        Args:
            metrics: Optional pre-calculated metrics, otherwise will calculate
            
        Returns:
            List of ExecutionAlert objects
        """
        if metrics is None:
            metrics = self.calculate_metrics()
            
        alerts = []
        timestamp = datetime.now(timezone.utc)
        
        # Check fill rate degradation
        if metrics.total_orders >= 10 and metrics.fill_rate < self.fill_rate_threshold:
            severity = "critical" if metrics.fill_rate < (self.fill_rate_threshold - 0.1) else "warning"
            alerts.append(ExecutionAlert(
                timestamp=timestamp,
                alert_type="fill_rate_degradation",
                severity=severity,
                metric="fill_rate",
                current_value=metrics.fill_rate,
                threshold_value=self.fill_rate_threshold,
                message=f"Fill rate {metrics.fill_rate:.2%} below threshold {self.fill_rate_threshold:.2%}",
                details={
                    "total_orders": metrics.total_orders,
                    "filled_orders": metrics.filled_orders,
                    "rejected_orders": metrics.rejected_orders
                }
            ))
            
        # Check rejection rate
        if metrics.total_orders >= 10 and metrics.rejection_rate > self.rejection_rate_threshold:
            severity = "critical" if metrics.rejection_rate > (self.rejection_rate_threshold + 0.1) else "warning"
            alerts.append(ExecutionAlert(
                timestamp=timestamp,
                alert_type="rejection_rate_high",
                severity=severity,
                metric="rejection_rate",
                current_value=metrics.rejection_rate,
                threshold_value=self.rejection_rate_threshold,
                message=f"Rejection rate {metrics.rejection_rate:.2%} above threshold {self.rejection_rate_threshold:.2%}",
                details={
                    "total_orders": metrics.total_orders,
                    "rejected_orders": metrics.rejected_orders
                }
            ))
            
        # Check average slippage
        if metrics.filled_orders >= 5 and abs(metrics.avg_slippage_bps) > self.slippage_threshold_bps:
            severity = "critical" if abs(metrics.avg_slippage_bps) > (self.slippage_threshold_bps * 2) else "warning"
            alerts.append(ExecutionAlert(
                timestamp=timestamp,
                alert_type="slippage_high",
                severity=severity,
                metric="avg_slippage_bps",
                current_value=metrics.avg_slippage_bps,
                threshold_value=self.slippage_threshold_bps,
                message=f"Average slippage {metrics.avg_slippage_bps:.2f} bps exceeds threshold {self.slippage_threshold_bps:.2f} bps",
                details={
                    "median_slippage_bps": metrics.median_slippage_bps,
                    "worst_slippage_bps": metrics.worst_slippage_bps,
                    "filled_orders": metrics.filled_orders
                }
            ))
            
        # Check slippage volatility
        if metrics.filled_orders >= 5 and metrics.slippage_std_bps > self.slippage_std_threshold_bps:
            severity = "warning"  # High volatility is typically a warning, not critical
            alerts.append(ExecutionAlert(
                timestamp=timestamp,
                alert_type="slippage_volatility_high",
                severity=severity,
                metric="slippage_std_bps",
                current_value=metrics.slippage_std_bps,
                threshold_value=self.slippage_std_threshold_bps,
                message=f"Slippage volatility {metrics.slippage_std_bps:.2f} bps exceeds threshold {self.slippage_std_threshold_bps:.2f} bps",
                details={
                    "avg_slippage_bps": metrics.avg_slippage_bps,
                    "worst_slippage_bps": metrics.worst_slippage_bps,
                    "best_slippage_bps": metrics.best_slippage_bps
                }
            ))
            
        return alerts
        
    def save_alert(self, alert: ExecutionAlert):
        """Save alert to database.
        
        Args:
            alert: ExecutionAlert to save
        """
        try:
            import json
            with duckdb.connect(self.db_path) as con:
                con.execute("""
                    INSERT INTO execution_alerts (
                        timestamp, alert_type, severity, metric,
                        current_value, threshold_value, message, details
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    alert.timestamp,
                    alert.alert_type,
                    alert.severity,
                    alert.metric,
                    alert.current_value,
                    alert.threshold_value,
                    alert.message,
                    json.dumps(alert.details)
                ))
            LOGGER.warning(f"Execution quality alert: {alert.message}")
        except Exception as e:
            LOGGER.error(f"Failed to save alert: {e}")
            
    def save_snapshot(self, metrics: ExecutionQualityMetrics):
        """Save metrics snapshot to database.
        
        Args:
            metrics: ExecutionQualityMetrics to save
        """
        try:
            with duckdb.connect(self.db_path) as con:
                con.execute("""
                    INSERT INTO execution_quality_snapshots (
                        timestamp, fill_rate, rejection_rate, partial_fill_rate,
                        avg_slippage_bps, median_slippage_bps, slippage_std_bps,
                        worst_slippage_bps, best_slippage_bps, avg_fill_time_ms,
                        total_orders, filled_orders, rejected_orders, partial_fills,
                        total_fees, avg_fee_bps
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metrics.timestamp,
                    metrics.fill_rate,
                    metrics.rejection_rate,
                    metrics.partial_fill_rate,
                    metrics.avg_slippage_bps,
                    metrics.median_slippage_bps,
                    metrics.slippage_std_bps,
                    metrics.worst_slippage_bps,
                    metrics.best_slippage_bps,
                    metrics.avg_fill_time_ms,
                    metrics.total_orders,
                    metrics.filled_orders,
                    metrics.rejected_orders,
                    metrics.partial_fills,
                    metrics.total_fees,
                    metrics.avg_fee_bps
                ))
        except Exception as e:
            LOGGER.error(f"Failed to save metrics snapshot: {e}")
            
    def monitor(self, save_snapshot: bool = True) -> Tuple[ExecutionQualityMetrics, List[ExecutionAlert]]:
        """Run full monitoring cycle: calculate metrics, check for degradation, save results.
        
        Args:
            save_snapshot: Whether to save metrics snapshot to database
            
        Returns:
            Tuple of (metrics, alerts)
        """
        # Calculate current metrics
        metrics = self.calculate_metrics()
        
        # Check for degradation
        alerts = self.check_for_degradation(metrics)
        
        # Save snapshot if requested
        if save_snapshot:
            self.save_snapshot(metrics)
            
        # Save and log alerts
        for alert in alerts:
            self.save_alert(alert)
            
        return metrics, alerts
        
    def get_recent_alerts(
        self,
        lookback_hours: int = 24,
        severity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get recent execution quality alerts.
        
        Args:
            lookback_hours: Hours to look back
            severity: Optional severity filter ('warning' or 'critical')
            
        Returns:
            List of alert dictionaries
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        
        try:
            with duckdb.connect(self.db_path) as con:
                severity_filter = f"AND severity = '{severity}'" if severity else ""
                
                query = f"""
                    SELECT * FROM execution_alerts 
                    WHERE timestamp >= ?
                    {severity_filter}
                    ORDER BY timestamp DESC
                """
                
                df = con.execute(query, (cutoff_time,)).df()
                
                if df.empty:
                    return []
                    
                return df.to_dict(orient="records")
                
        except Exception as e:
            LOGGER.error(f"Failed to get recent alerts: {e}")
            return []
            
    def get_metrics_history(
        self,
        lookback_hours: int = 168  # 1 week
    ) -> pd.DataFrame:
        """Get historical metrics snapshots.
        
        Args:
            lookback_hours: Hours of history to retrieve
            
        Returns:
            DataFrame with historical metrics
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        
        try:
            with duckdb.connect(self.db_path) as con:
                query = """
                    SELECT * FROM execution_quality_snapshots 
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                """
                
                df = con.execute(query, (cutoff_time,)).df()
                return df
                
        except Exception as e:
            LOGGER.error(f"Failed to get metrics history: {e}")
            return pd.DataFrame()
            
    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive execution quality summary report.
        
        Returns:
            Dictionary with summary statistics and analysis
        """
        metrics = self.calculate_metrics()
        slippage_dist = self.get_slippage_distribution()
        benchmark_comparison = self.compare_to_benchmark(metrics)
        recent_alerts = self.get_recent_alerts(lookback_hours=24)
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lookback_hours": self.lookback_hours,
            "current_metrics": {
                "fill_rate": metrics.fill_rate,
                "rejection_rate": metrics.rejection_rate,
                "partial_fill_rate": metrics.partial_fill_rate,
                "avg_slippage_bps": metrics.avg_slippage_bps,
                "median_slippage_bps": metrics.median_slippage_bps,
                "slippage_std_bps": metrics.slippage_std_bps,
                "worst_slippage_bps": metrics.worst_slippage_bps,
                "best_slippage_bps": metrics.best_slippage_bps,
                "avg_fill_time_ms": metrics.avg_fill_time_ms,
                "total_orders": metrics.total_orders,
                "filled_orders": metrics.filled_orders,
                "rejected_orders": metrics.rejected_orders,
                "partial_fills": metrics.partial_fills,
                "total_fees": metrics.total_fees,
                "avg_fee_bps": metrics.avg_fee_bps
            },
            "slippage_distribution": slippage_dist,
            "benchmark_comparison": benchmark_comparison,
            "recent_alerts": {
                "count": len(recent_alerts),
                "critical_count": sum(1 for a in recent_alerts if a.get("severity") == "critical"),
                "warning_count": sum(1 for a in recent_alerts if a.get("severity") == "warning"),
                "alerts": recent_alerts[:10]  # Most recent 10
            },
            "thresholds": {
                "fill_rate_threshold": self.fill_rate_threshold,
                "rejection_rate_threshold": self.rejection_rate_threshold,
                "slippage_threshold_bps": self.slippage_threshold_bps,
                "slippage_std_threshold_bps": self.slippage_std_threshold_bps
            }
        }
