"""Tests for Performance Attribution Module."""
import pytest
from datetime import datetime, timedelta
from openquant.analysis.attribution import (
    PerformanceAttributor,
    AttributionResult,
    TradeAttribution
)
from openquant.storage.audit_trail import AuditTrail, EventType


class TestPerformanceAttributor:
    """Test suite for PerformanceAttributor."""
    
    def test_initialization(self):
        """Test attributor initialization."""
        attributor = PerformanceAttributor()
        assert attributor is not None
        assert attributor.audit_trail is not None
        assert attributor.tca_monitor is not None
        
    def test_analyze_no_trades(self):
        """Test attribution with no trades."""
        attributor = PerformanceAttributor()
        result = attributor.analyze(days=1)
        
        assert isinstance(result, AttributionResult)
        assert result.total_return == 0.0
        assert result.timing_effect == 0.0
        assert result.selection_effect == 0.0
        assert result.sizing_effect == 0.0
        assert result.cost_drag == 0.0
        assert "message" in result.details
        
    def test_analyze_with_mock_trades(self):
        """Test attribution with mock trade data."""
        audit = AuditTrail(db_path=":memory:")
        attributor = PerformanceAttributor(audit_trail=audit)
        
        now = datetime.now()
        
        audit.log_signal(
            symbol="BTC/USD",
            strategy="test_strategy",
            side="BUY",
            price=50000.0,
            message="Test signal"
        )
        
        audit.log_execution(
            symbol="BTC/USD",
            strategy="test_strategy",
            side="BUY",
            quantity=1.0,
            price=50100.0,
            message="Test buy"
        )
        
        later = now + timedelta(hours=2)
        audit.log_execution(
            symbol="BTC/USD",
            strategy="test_strategy",
            side="SELL",
            quantity=1.0,
            price=51000.0,
            message="Test sell"
        )
        
        result = attributor.analyze(days=1)
        
        assert isinstance(result, AttributionResult)
        assert result.details["num_trades"] >= 0
        
    def test_get_completed_trades(self):
        """Test extraction of completed trades."""
        audit = AuditTrail(db_path=":memory:")
        attributor = PerformanceAttributor(audit_trail=audit)
        
        now = datetime.now()
        
        audit.log_execution(
            symbol="BTC/USD",
            strategy="test",
            side="BUY",
            quantity=1.0,
            price=50000.0
        )
        
        audit.log_execution(
            symbol="BTC/USD",
            strategy="test",
            side="SELL",
            quantity=1.0,
            price=51000.0
        )
        
        trades = attributor._get_completed_trades(
            now - timedelta(days=1),
            now + timedelta(days=1)
        )
        
        assert isinstance(trades, list)
        
    def test_compare_strategies_empty(self):
        """Test strategy comparison with no data."""
        attributor = PerformanceAttributor()
        result = attributor.compare_strategies(days=1)
        
        assert isinstance(result, dict)
        assert len(result) == 0
        
    def test_compare_instruments_empty(self):
        """Test instrument comparison with no data."""
        attributor = PerformanceAttributor()
        result = attributor.compare_instruments(days=1)
        
        assert isinstance(result, dict)
        assert len(result) == 0
        
    def test_get_trade_level_attribution_empty(self):
        """Test trade-level attribution with no data."""
        attributor = PerformanceAttributor()
        trades = attributor.get_trade_level_attribution(days=1)
        
        assert isinstance(trades, list)
        assert len(trades) == 0
        
    def test_attribution_result_to_dict(self):
        """Test AttributionResult serialization."""
        now = datetime.now()
        result = AttributionResult(
            period_start=now - timedelta(days=30),
            period_end=now,
            total_return=0.05,
            timing_effect=0.02,
            selection_effect=0.01,
            sizing_effect=0.015,
            cost_drag=-0.005,
            residual=0.01,
            details={"test": "data"}
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "period_start" in result_dict
        assert "total_return" in result_dict
        assert result_dict["total_return"] == 0.05
        
    def test_attribution_result_summary(self):
        """Test AttributionResult summary generation."""
        now = datetime.now()
        result = AttributionResult(
            period_start=now - timedelta(days=30),
            period_end=now,
            total_return=0.05,
            timing_effect=0.02,
            selection_effect=0.01,
            sizing_effect=0.015,
            cost_drag=-0.005,
            residual=0.01,
            details={}
        )
        
        summary = result.summary()
        
        assert isinstance(summary, str)
        assert "Total Return" in summary
        assert "Timing Effect" in summary
        assert "5.00%" in summary
        
    def test_generate_report(self, tmp_path):
        """Test report generation."""
        attributor = PerformanceAttributor()
        output_path = tmp_path / "test_report.json"
        
        report = attributor.generate_report(
            days=1,
            output_path=str(output_path)
        )
        
        assert isinstance(report, dict)
        assert "generated_at" in report
        assert "period" in report
        assert "overall_attribution" in report
        assert "strategy_comparison" in report
        assert "instrument_comparison" in report


class TestAuditTrailIntegration:
    """Test audit trail integration features."""
    
    def test_get_trade_pairs(self):
        """Test getting trade pairs from audit trail."""
        audit = AuditTrail(db_path=":memory:")
        
        now = datetime.now()
        
        audit.log_execution(
            symbol="BTC/USD",
            strategy="test",
            side="BUY",
            quantity=1.0,
            price=50000.0
        )
        
        pairs = audit.get_trade_pairs()
        
        assert isinstance(pairs, list)
        
    def test_get_execution_quality_metrics(self):
        """Test execution quality metrics."""
        audit = AuditTrail(db_path=":memory:")
        
        now = datetime.now()
        
        audit.log_order_decision(
            symbol="BTC/USD",
            strategy="test",
            side="BUY",
            quantity=1.0,
            price=50000.0
        )
        
        audit.log_execution(
            symbol="BTC/USD",
            strategy="test",
            side="BUY",
            quantity=1.0,
            price=50100.0
        )
        
        metrics = audit.get_execution_quality_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_decisions" in metrics
        assert "total_executions" in metrics
        assert "execution_rate" in metrics
        
    def test_get_signal_to_execution_lag(self):
        """Test signal to execution lag calculation."""
        audit = AuditTrail(db_path=":memory:")
        
        now = datetime.now()
        
        audit.log_signal(
            symbol="BTC/USD",
            strategy="test",
            side="BUY",
            price=50000.0
        )
        
        audit.log_execution(
            symbol="BTC/USD",
            strategy="test",
            side="BUY",
            quantity=1.0,
            price=50100.0
        )
        
        lag = audit.get_signal_to_execution_lag()
        
        assert isinstance(lag, dict)
        assert "count" in lag
        assert "avg_lag_seconds" in lag
        assert "median_lag_seconds" in lag
