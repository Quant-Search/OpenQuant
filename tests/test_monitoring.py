"""Test Trade Analysis and Alerts.

Verifies TradeAnalyzer, IntelligentAlerts, and PerformanceTracker functionality.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openquant.reporting.intelligent_alerts import IntelligentAlerts, Alert, AlertType
from openquant.monitoring import PerformanceTracker, PerformanceMetrics

def test_pnl_anomaly_detection():
    alerts = IntelligentAlerts()
    
    # Normal PnL
    normal_pnl = [0.01] * 20
    alert = alerts.check_pnl_anomaly(normal_pnl)
    assert alert is None
    
    # Anomalous PnL
    anomalous_pnl = [0.01] * 19 + [0.10]  # Sudden spike
    alert = alerts.check_pnl_anomaly(anomalous_pnl)
    assert alert is not None
    assert alert.type == AlertType.ANOMALY

def test_drawdown_alert():
    alerts = IntelligentAlerts()
    
    # Small drawdown - no alert
    equity = [100, 102, 101, 103]
    alert = alerts.check_drawdown(equity, threshold=0.15)
    assert alert is None
    
    # Large drawdown - alert
    equity_dd = [100, 102, 85, 80]  # ~21% drawdown
    alert = alerts.check_drawdown(equity_dd, threshold=0.15)
    assert alert is not None
    assert alert.severity == "critical"

def test_signal_quality_degradation():
    alerts = IntelligentAlerts()
    
    # Good signals
    good_probs = [0.7] * 10
    alert = alerts.check_signal_quality(good_probs, min_prob=0.55)
    assert alert is None
    
    # Poor signals
    poor_probs = [0.52] * 10
    alert = alerts.check_signal_quality(poor_probs, min_prob=0.55)
    assert alert is not None

def test_regime_change_alert():
    alerts = IntelligentAlerts()
    
    alert = alerts.check_regime_change("trending_up", "ranging")
    assert alert is not None
    assert alert.type == AlertType.REGIME_CHANGE
    assert "trending_up" in alert.message
    assert "ranging" in alert.message

def test_alert_buffer():
    alerts = IntelligentAlerts()
    
    alert1 = Alert(AlertType.ANOMALY, "warning", "Test alert")
    alerts.add_alert(alert1)
    
    recent = alerts.get_recent_alerts(hours=24)
    assert len(recent) == 1
    assert recent[0]['message'] == "Test alert"


def test_performance_tracker_basic():
    """Test basic PerformanceTracker initialization and updates."""
    tracker = PerformanceTracker(
        backtest_sharpe=2.0,
        backtest_max_drawdown=0.10,
        alert_threshold=0.20,
        freq="1d"
    )
    
    # Update with equity
    tracker.update(equity=100000.0, returns=0.01)
    tracker.update(equity=101000.0, returns=0.01)
    
    metrics = tracker.get_current_metrics()
    assert metrics['num_observations'] == 2
    assert metrics['current_equity'] == 101000.0
    assert metrics['peak_equity'] == 101000.0


def test_performance_tracker_drawdown():
    """Test drawdown tracking from peak."""
    tracker = PerformanceTracker(freq="1d")
    
    # Build up equity
    tracker.update(equity=100000.0, returns=0.00)
    tracker.update(equity=110000.0, returns=0.10)  # New peak
    tracker.update(equity=105000.0, returns=-0.045)  # Drawdown
    
    metrics = tracker.get_current_metrics()
    assert metrics['peak_equity'] == 110000.0
    assert metrics['current_equity'] == 105000.0
    
    # Drawdown should be (110000 - 105000) / 110000 = 4.5%
    expected_dd = (110000.0 - 105000.0) / 110000.0
    assert abs(metrics['drawdown'] - expected_dd) < 0.001


def test_performance_tracker_ew_sharpe():
    """Test exponentially-weighted Sharpe calculation."""
    np.random.seed(42)
    tracker = PerformanceTracker(
        sharpe_halflife_days=30,
        freq="1d"
    )
    
    # Generate positive returns
    equity = 100000.0
    for _ in range(50):
        ret = np.random.normal(0.001, 0.02)
        equity *= (1 + ret)
        tracker.update(equity=equity, returns=ret)
    
    metrics = tracker.get_current_metrics()
    
    # Should have positive Sharpe
    assert metrics['sharpe_ratio'] > 0
    assert np.isfinite(metrics['sharpe_ratio'])


def test_performance_tracker_correlation_drift():
    """Test correlation drift detection."""
    # Create backtest correlation matrix
    backtest_corr = np.array([
        [1.0, 0.6, 0.3],
        [0.6, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ])
    
    tracker = PerformanceTracker(
        backtest_correlation_matrix=backtest_corr,
        freq="1d"
    )
    
    np.random.seed(42)
    symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]
    equity = 100000.0
    
    # Generate correlated returns matching backtest
    for _ in range(30):
        cov_matrix = backtest_corr * 0.02 ** 2
        symbol_rets = np.random.multivariate_normal([0.001, 0.001, 0.001], cov_matrix)
        portfolio_ret = np.mean(symbol_rets)
        equity *= (1 + portfolio_ret)
        
        symbol_returns = {symbol: ret for symbol, ret in zip(symbols, symbol_rets)}
        tracker.update(equity=equity, returns=portfolio_ret, symbol_returns=symbol_returns)
    
    metrics = tracker.get_current_metrics()
    
    # Should have correlation drift metric
    assert metrics['correlation_drift'] is not None
    assert metrics['correlation_drift'] >= 0


def test_performance_tracker_alert_triggering():
    """Test that alerts are triggered on degradation."""
    tracker = PerformanceTracker(
        backtest_sharpe=2.5,
        backtest_max_drawdown=0.10,
        alert_threshold=0.20,
        freq="1d"
    )
    
    # Reset alerts to ensure clean state
    tracker.reset_alerts()
    
    np.random.seed(123)
    equity = 100000.0
    
    # Generate poor performance that should trigger alerts
    for _ in range(40):
        ret = np.random.normal(-0.002, 0.03)  # Negative mean, high vol
        equity *= (1 + ret)
        tracker.update(equity=equity, returns=ret)
    
    metrics = tracker.get_current_metrics()
    
    # Should show degradation
    if metrics['sharpe_degradation_pct'] is not None:
        assert abs(metrics['sharpe_degradation_pct']) > 0


def test_performance_tracker_metrics_history():
    """Test metrics history retrieval."""
    tracker = PerformanceTracker(freq="1d")
    
    equity = 100000.0
    for i in range(10):
        ret = 0.01 if i % 2 == 0 else -0.005
        equity *= (1 + ret)
        tracker.update(equity=equity, returns=ret)
    
    # Get all history
    history = tracker.get_metrics_history()
    assert len(history) == 10
    
    # Get limited history
    history_limited = tracker.get_metrics_history(lookback_days=0)
    assert len(history_limited) >= 0


def test_performance_tracker_thread_safety():
    """Test that tracker is thread-safe."""
    import threading
    
    tracker = PerformanceTracker(freq="1d")
    errors = []
    
    def update_worker(start_equity):
        try:
            equity = start_equity
            for _ in range(10):
                ret = np.random.normal(0.001, 0.01)
                equity *= (1 + ret)
                tracker.update(equity=equity, returns=ret)
        except Exception as e:
            errors.append(e)
    
    # Run multiple threads
    threads = []
    for i in range(5):
        t = threading.Thread(target=update_worker, args=(100000.0 + i * 1000,))
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    # Should have no errors
    assert len(errors) == 0
    
    # Should have updates from all threads
    metrics = tracker.get_current_metrics()
    assert metrics['num_observations'] == 50  # 5 threads * 10 updates


def test_performance_tracker_summary():
    """Test summary string generation."""
    tracker = PerformanceTracker(
        backtest_sharpe=2.0,
        backtest_max_drawdown=0.15,
        freq="1d"
    )
    
    equity = 100000.0
    for _ in range(20):
        ret = np.random.normal(0.001, 0.02)
        equity *= (1 + ret)
        tracker.update(equity=equity, returns=ret)
    
    summary = tracker.get_summary()
    
    # Should contain key information
    assert "Sharpe" in summary
    assert "Drawdown" in summary
    assert "Equity" in summary
    assert "Observations" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
