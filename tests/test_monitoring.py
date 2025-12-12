"""Test Trade Analysis and Alerts.

Verifies TradeAnalyzer and IntelligentAlerts functionality.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openquant.reporting.intelligent_alerts import IntelligentAlerts, Alert, AlertType

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

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
