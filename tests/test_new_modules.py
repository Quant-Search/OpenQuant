"""
Tests for newly created modules:
- Audit Trail Database
- Retrain Scheduler
- Circuit Breaker
- Asset Limits
"""
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json


class TestAuditTrail:
    """Tests for the audit trail database module."""
    
    def test_audit_trail_log_and_query(self):
        """Test logging events and querying them back."""
        from openquant.storage.audit_trail import AuditTrail, EventType
        
        # Create a temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_audit.duckdb"
            audit = AuditTrail(db_path=str(db_path))
            
            # Log a signal event
            audit.log_signal(
                symbol="EURUSD",
                strategy="kalman",
                side="BUY",
                price=1.10,
                message="Test signal"
            )
            
            # Log an execution event
            audit.log_execution(
                symbol="EURUSD",
                strategy="kalman",
                side="BUY",
                quantity=10000,
                price=1.1001,
                message="Test execution"
            )
            
            # Query all events
            events = audit.query(limit=10)
            assert len(events) == 2
            
            # Query by event type
            signals = audit.query(event_type=EventType.SIGNAL)
            assert len(signals) == 1
            assert signals[0]["symbol"] == "EURUSD"
    
    def test_audit_trail_risk_event(self):
        """Test logging risk events."""
        from openquant.storage.audit_trail import AuditTrail, EventType
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_audit.duckdb"
            audit = AuditTrail(db_path=str(db_path))
            
            # Log a circuit breaker event
            audit.log_risk_event(
                event_type=EventType.CIRCUIT_BREAKER,
                message="Circuit breaker tripped",
                details={"daily_loss_pct": 2.5}
            )
            
            events = audit.query(event_type=EventType.CIRCUIT_BREAKER)
            assert len(events) == 1
            assert "Circuit breaker" in events[0]["message"]


class TestRetrainScheduler:
    """Tests for the retrain scheduler module."""
    
    def test_scheduler_is_retrain_due_first_time(self):
        """Test that retrain is due when never run before."""
        from openquant.research.retrain_scheduler import RetrainScheduler, RetrainFrequency
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "retrain_state.json"
            scheduler = RetrainScheduler(
                frequency=RetrainFrequency.WEEKLY,
                state_file=str(state_file)
            )
            
            # First time should always be due
            assert scheduler.is_retrain_due() is True
    
    def test_scheduler_is_retrain_due_after_run(self):
        """Test that retrain is not due immediately after running."""
        from openquant.research.retrain_scheduler import RetrainScheduler, RetrainFrequency
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "retrain_state.json"
            
            # Create a scheduler with a mock retrain function
            def mock_retrain():
                return True
            
            scheduler = RetrainScheduler(
                frequency=RetrainFrequency.WEEKLY,
                retrain_func=mock_retrain,
                state_file=str(state_file)
            )
            
            # Run retrain
            result = scheduler.run_retrain()
            assert result is True
            
            # Should not be due immediately after
            assert scheduler.is_retrain_due() is False
    
    def test_scheduler_get_status(self):
        """Test getting scheduler status."""
        from openquant.research.retrain_scheduler import RetrainScheduler, RetrainFrequency
        
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "retrain_state.json"
            scheduler = RetrainScheduler(
                frequency=RetrainFrequency.DAILY,
                state_file=str(state_file)
            )
            
            status = scheduler.get_status()
            assert status["frequency"] == "daily"
            assert "is_retrain_due" in status
            assert "next_retrain_time" in status


class TestCircuitBreaker:
    """Tests for the circuit breaker module."""

    def test_circuit_breaker_daily_loss(self):
        """Test circuit breaker trips on daily loss.

        The circuit breaker calculates daily loss as:
        daily_loss = (start_of_day_equity - current_equity) / start_of_day_equity

        If daily_loss >= daily_loss_limit, the breaker trips.
        """
        from openquant.risk.circuit_breaker import CircuitBreaker

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "cb_state.json"
            cb = CircuitBreaker(
                daily_loss_limit=0.02,  # 2%
                state_file=str(state_file)
            )

            # Reset to start fresh
            cb.reset()

            # First update sets start_of_day_equity
            cb.update(current_equity=100000)
            assert cb.is_tripped() is False

            # Update with a small loss (1%) - should not trip
            cb.update(current_equity=99000)  # 1% loss
            assert cb.is_tripped() is False

            # Update with a large loss (3%) - should trip
            cb.update(current_equity=97000)  # 3% loss from start
            assert cb.is_tripped() is True

    def test_circuit_breaker_reset(self):
        """Test circuit breaker reset."""
        from openquant.risk.circuit_breaker import CircuitBreaker

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = Path(tmpdir) / "cb_state.json"
            cb = CircuitBreaker(
                daily_loss_limit=0.02,
                state_file=str(state_file)
            )

            # Reset and set initial equity
            cb.reset()
            cb.update(current_equity=100000)

            # Trip the breaker with 5% loss
            cb.update(current_equity=95000)
            assert cb.is_tripped() is True

            # Reset
            cb.reset()
            assert cb.is_tripped() is False

