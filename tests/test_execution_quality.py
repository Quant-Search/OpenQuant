"""Tests for execution quality monitoring."""
import pytest
import tempfile
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

from openquant.analysis.tca import TCAMonitor
from openquant.analysis.execution_quality import (
    ExecutionQualityMonitor,
    ExecutionQualityMetrics,
    ExecutionAlert
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".duckdb", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    try:
        os.unlink(db_path)
    except Exception:
        pass


@pytest.fixture
def tca_with_data(temp_db):
    """Create TCA monitor with sample data."""
    tca = TCAMonitor(db_path=temp_db)
    
    # Add some filled orders
    for i in range(10):
        order_id = f"ORD{i:03d}"
        symbol = "BTC/USD" if i % 2 == 0 else "ETH/USD"
        side = "buy" if i % 3 == 0 else "sell"
        arrival = 50000.0 + i * 100
        fill = arrival + (5.0 if side == "buy" else -5.0)  # 5 bps slippage
        
        tca.log_order(order_id, symbol, side, 1.0, arrival)
        tca.update_fill(order_id, fill, 1.0, 2.5)
    
    # Add a rejected order
    tca.log_order("ORD999", "BTC/USD", "buy", 1.0, 50000.0)
    
    return tca


def test_execution_quality_metrics_initialization():
    """Test ExecutionQualityMetrics initialization."""
    metrics = ExecutionQualityMetrics()
    
    assert metrics.fill_rate == 0.0
    assert metrics.rejection_rate == 0.0
    assert metrics.avg_slippage_bps == 0.0
    assert metrics.total_orders == 0
    assert isinstance(metrics.timestamp, datetime)


def test_execution_quality_monitor_init(temp_db):
    """Test ExecutionQualityMonitor initialization."""
    monitor = ExecutionQualityMonitor(db_path=temp_db)
    
    assert monitor.db_path == temp_db
    assert monitor.fill_rate_threshold == 0.95
    assert monitor.rejection_rate_threshold == 0.05
    assert monitor.slippage_threshold_bps == 10.0
    assert Path(temp_db).parent.exists()


def test_calculate_metrics(tca_with_data, temp_db):
    """Test metrics calculation."""
    monitor = ExecutionQualityMonitor(db_path=temp_db, lookback_hours=24)
    
    metrics = monitor.calculate_metrics()
    
    assert metrics.total_orders == 11  # 10 filled + 1 rejected
    assert metrics.filled_orders == 10
    assert metrics.rejected_orders == 0  # Status is "NEW" not "REJECTED"
    assert metrics.fill_rate == 10 / 11
    assert metrics.avg_slippage_bps > 0
    assert metrics.total_fees == 25.0  # 10 orders * 2.5 fee


def test_calculate_metrics_by_symbol(tca_with_data, temp_db):
    """Test metrics calculation for specific symbol."""
    monitor = ExecutionQualityMonitor(db_path=temp_db)
    
    btc_metrics = monitor.calculate_metrics(symbol="BTC/USD")
    eth_metrics = monitor.calculate_metrics(symbol="ETH/USD")
    
    assert btc_metrics.filled_orders == 5
    assert eth_metrics.filled_orders == 5


def test_slippage_distribution(tca_with_data, temp_db):
    """Test slippage distribution calculation."""
    monitor = ExecutionQualityMonitor(db_path=temp_db)
    
    dist = monitor.get_slippage_distribution()
    
    assert dist["count"] == 10
    assert "mean" in dist
    assert "median" in dist
    assert "std" in dist
    assert "percentiles" in dist
    assert "histogram" in dist
    assert "p50" in dist["percentiles"]
    assert "p95" in dist["percentiles"]


def test_check_for_degradation_no_alerts(tca_with_data, temp_db):
    """Test degradation check with good execution quality."""
    monitor = ExecutionQualityMonitor(
        db_path=temp_db,
        fill_rate_threshold=0.80,  # Lenient threshold
        slippage_threshold_bps=50.0  # Lenient threshold
    )
    
    metrics = monitor.calculate_metrics()
    alerts = monitor.check_for_degradation(metrics)
    
    assert len(alerts) == 0


def test_check_for_degradation_with_alerts(tca_with_data, temp_db):
    """Test degradation check with poor execution quality."""
    monitor = ExecutionQualityMonitor(
        db_path=temp_db,
        fill_rate_threshold=0.99,  # Strict threshold
        slippage_threshold_bps=1.0   # Strict threshold
    )
    
    metrics = monitor.calculate_metrics()
    alerts = monitor.check_for_degradation(metrics)
    
    # Should have alerts for slippage
    assert len(alerts) > 0
    assert any(a.alert_type == "slippage_high" for a in alerts)


def test_save_and_retrieve_alert(temp_db):
    """Test saving and retrieving alerts."""
    monitor = ExecutionQualityMonitor(db_path=temp_db)
    
    alert = ExecutionAlert(
        timestamp=datetime.now(timezone.utc),
        alert_type="test_alert",
        severity="warning",
        metric="test_metric",
        current_value=1.0,
        threshold_value=0.5,
        message="Test alert message",
        details={"key": "value"}
    )
    
    monitor.save_alert(alert)
    
    recent = monitor.get_recent_alerts(lookback_hours=1)
    assert len(recent) == 1
    assert recent[0]["alert_type"] == "test_alert"
    assert recent[0]["severity"] == "warning"


def test_save_and_retrieve_snapshot(tca_with_data, temp_db):
    """Test saving and retrieving metrics snapshots."""
    monitor = ExecutionQualityMonitor(db_path=temp_db)
    
    metrics = monitor.calculate_metrics()
    monitor.save_snapshot(metrics)
    
    history = monitor.get_metrics_history(lookback_hours=1)
    assert len(history) == 1
    assert history.iloc[0]["fill_rate"] == metrics.fill_rate


def test_monitor_cycle(tca_with_data, temp_db):
    """Test full monitoring cycle."""
    monitor = ExecutionQualityMonitor(
        db_path=temp_db,
        fill_rate_threshold=0.99,  # Will trigger alert
        slippage_threshold_bps=1.0
    )
    
    metrics, alerts = monitor.monitor(save_snapshot=True)
    
    # Check metrics were calculated
    assert metrics.total_orders > 0
    
    # Check alerts were generated
    assert len(alerts) > 0
    
    # Check snapshot was saved
    history = monitor.get_metrics_history(lookback_hours=1)
    assert len(history) == 1
    
    # Check alerts were saved
    recent = monitor.get_recent_alerts(lookback_hours=1)
    assert len(recent) == len(alerts)


def test_compare_to_benchmark_no_data(temp_db):
    """Test benchmark comparison with insufficient data."""
    monitor = ExecutionQualityMonitor(db_path=temp_db)
    
    metrics = ExecutionQualityMetrics()
    comparison = monitor.compare_to_benchmark(metrics)
    
    assert not comparison["has_benchmark"]


def test_summary_report(tca_with_data, temp_db):
    """Test summary report generation."""
    monitor = ExecutionQualityMonitor(db_path=temp_db)
    
    report = monitor.get_summary_report()
    
    assert "timestamp" in report
    assert "lookback_hours" in report
    assert "current_metrics" in report
    assert "slippage_distribution" in report
    assert "benchmark_comparison" in report
    assert "recent_alerts" in report
    assert "thresholds" in report
    
    # Check current metrics
    assert report["current_metrics"]["fill_rate"] > 0
    assert report["current_metrics"]["total_orders"] > 0


def test_execution_alert_dataclass():
    """Test ExecutionAlert dataclass."""
    alert = ExecutionAlert(
        timestamp=datetime.now(timezone.utc),
        alert_type="fill_rate_degradation",
        severity="critical",
        metric="fill_rate",
        current_value=0.85,
        threshold_value=0.95,
        message="Fill rate below threshold",
        details={"orders": 100}
    )
    
    assert alert.severity == "critical"
    assert alert.current_value < alert.threshold_value
    assert "orders" in alert.details


def test_empty_database(temp_db):
    """Test monitor behavior with empty database."""
    monitor = ExecutionQualityMonitor(db_path=temp_db)
    
    metrics = monitor.calculate_metrics()
    
    assert metrics.total_orders == 0
    assert metrics.fill_rate == 0.0
    assert metrics.avg_slippage_bps == 0.0
    
    # Should not generate alerts with no data
    alerts = monitor.check_for_degradation(metrics)
    assert len(alerts) == 0


def test_filter_by_severity(tca_with_data, temp_db):
    """Test filtering alerts by severity."""
    monitor = ExecutionQualityMonitor(
        db_path=temp_db,
        fill_rate_threshold=0.99,
        slippage_threshold_bps=1.0
    )
    
    # Generate some alerts
    metrics, alerts = monitor.monitor(save_snapshot=True)
    
    # Get only critical alerts
    critical = monitor.get_recent_alerts(lookback_hours=1, severity="critical")
    warning = monitor.get_recent_alerts(lookback_hours=1, severity="warning")
    
    # All returned alerts should match the requested severity
    for alert in critical:
        assert alert["severity"] == "critical"
    
    for alert in warning:
        assert alert["severity"] == "warning"


def test_custom_thresholds():
    """Test custom threshold initialization."""
    monitor = ExecutionQualityMonitor(
        fill_rate_threshold=0.98,
        rejection_rate_threshold=0.03,
        slippage_threshold_bps=5.0,
        slippage_std_threshold_bps=15.0,
        lookback_hours=12
    )
    
    assert monitor.fill_rate_threshold == 0.98
    assert monitor.rejection_rate_threshold == 0.03
    assert monitor.slippage_threshold_bps == 5.0
    assert monitor.slippage_std_threshold_bps == 15.0
    assert monitor.lookback_hours == 12
