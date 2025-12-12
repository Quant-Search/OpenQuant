"""
Audit Trail Database Module.

Provides persistent logging of ALL trading decisions and events:
- Signal generation (strategy outputs)
- Order decisions (before execution)
- Order executions (fills)
- Risk events (kill switch, circuit breaker, limit violations)
- System events (startup, shutdown, errors)

This creates a complete audit trail for:
- Regulatory compliance
- Post-trade analysis
- Debugging and troubleshooting
- Performance attribution

All events are stored in DuckDB for efficient querying.
"""
from __future__ import annotations
import duckdb
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json

from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)


class EventType(Enum):
    """Types of audit events."""
    SIGNAL = "SIGNAL"  # Strategy generated a signal
    ORDER_DECISION = "ORDER_DECISION"  # Decision to place an order
    ORDER_EXECUTION = "ORDER_EXECUTION"  # Order was executed
    ORDER_REJECTED = "ORDER_REJECTED"  # Order was rejected
    KILL_SWITCH = "KILL_SWITCH"  # Kill switch activated/deactivated
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"  # Circuit breaker tripped/reset
    LIMIT_VIOLATION = "LIMIT_VIOLATION"  # Asset limit violated
    SYSTEM_START = "SYSTEM_START"  # System started
    SYSTEM_STOP = "SYSTEM_STOP"  # System stopped
    ERROR = "ERROR"  # Error occurred
    WARNING = "WARNING"  # Warning issued


@dataclass
class AuditEvent:
    """A single audit event."""
    event_type: EventType
    timestamp: datetime
    symbol: Optional[str] = None
    strategy: Optional[str] = None
    side: Optional[str] = None  # BUY, SELL
    quantity: Optional[float] = None
    price: Optional[float] = None
    message: str = ""
    details: Optional[Dict[str, Any]] = None  # JSON-serializable extra data

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "strategy": self.strategy,
            "side": self.side,
            "quantity": self.quantity,
            "price": self.price,
            "message": self.message,
            "details": json.dumps(self.details) if self.details else None,
        }


class AuditTrail:
    """
    Persistent audit trail for all trading decisions and events.

    Usage:
        audit = AuditTrail()

        # Log a signal
        audit.log_signal(
            symbol="EURUSD",
            strategy="kalman",
            side="BUY",
            price=1.10,
            message="Kalman filter detected undervaluation"
        )

        # Log an execution
        audit.log_execution(
            symbol="EURUSD",
            strategy="kalman",
            side="BUY",
            quantity=10000,
            price=1.1001,
            message="Market order filled"
        )

        # Query events
        events = audit.query(
            event_type=EventType.ORDER_EXECUTION,
            symbol="EURUSD",
            start_time=datetime(2024, 1, 1)
        )
    """

    def __init__(self, db_path: str = "data/audit_trail.duckdb"):
        """Initialize audit trail with database path."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _get_connection(self):
        """Get a database connection."""
        return duckdb.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        """Create the audit table if it does not exist."""
        con = self._get_connection()
        try:
            con.execute("""
                CREATE SEQUENCE IF NOT EXISTS audit_events_id_seq;
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER DEFAULT nextval('audit_events_id_seq') PRIMARY KEY,
                    event_type VARCHAR NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    symbol VARCHAR,
                    strategy VARCHAR,
                    side VARCHAR,
                    quantity DOUBLE,
                    price DOUBLE,
                    message VARCHAR,
                    details VARCHAR
                );
            """)
            # Create index for common queries
            con.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON audit_events(timestamp);
            """)
            con.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_type
                ON audit_events(event_type);
            """)
            con.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_symbol
                ON audit_events(symbol);
            """)
        finally:
            con.close()

    def log_event(self, event: AuditEvent) -> None:
        """Log a single audit event."""
        con = self._get_connection()
        try:
            con.execute("""
                INSERT INTO audit_events
                (event_type, timestamp, symbol, strategy, side, quantity, price, message, details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_type.value,
                event.timestamp,
                event.symbol,
                event.strategy,
                event.side,
                event.quantity,
                event.price,
                event.message,
                json.dumps(event.details) if event.details else None,
            ))
        finally:
            con.close()

    def log_signal(
        self,
        symbol: str,
        strategy: str,
        side: str,
        price: float,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a trading signal."""
        self.log_event(AuditEvent(
            event_type=EventType.SIGNAL,
            timestamp=datetime.now(),
            symbol=symbol,
            strategy=strategy,
            side=side,
            price=price,
            message=message,
            details=details,
        ))

    def log_order_decision(
        self,
        symbol: str,
        strategy: str,
        side: str,
        quantity: float,
        price: float,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an order decision (before execution)."""
        self.log_event(AuditEvent(
            event_type=EventType.ORDER_DECISION,
            timestamp=datetime.now(),
            symbol=symbol,
            strategy=strategy,
            side=side,
            quantity=quantity,
            price=price,
            message=message,
            details=details,
        ))

    def log_execution(
        self,
        symbol: str,
        strategy: str,
        side: str,
        quantity: float,
        price: float,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an order execution."""
        self.log_event(AuditEvent(
            event_type=EventType.ORDER_EXECUTION,
            timestamp=datetime.now(),
            symbol=symbol,
            strategy=strategy,
            side=side,
            quantity=quantity,
            price=price,
            message=message,
            details=details,
        ))

    def log_rejection(
        self,
        symbol: str,
        strategy: str,
        side: str,
        quantity: float,
        price: float,
        reason: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log an order rejection."""
        self.log_event(AuditEvent(
            event_type=EventType.ORDER_REJECTED,
            timestamp=datetime.now(),
            symbol=symbol,
            strategy=strategy,
            side=side,
            quantity=quantity,
            price=price,
            message=reason,
            details=details,
        ))

    def log_risk_event(
        self,
        event_type: EventType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a risk event (kill switch, circuit breaker, etc.)."""
        self.log_event(AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            message=message,
            details=details,
        ))

    def log_system_event(
        self,
        event_type: EventType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a system event (start, stop, error)."""
        self.log_event(AuditEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            message=message,
            details=details,
        ))

    def query(
        self,
        event_type: Optional[EventType] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query audit events with filters."""
        con = self._get_connection()
        try:
            conditions = []
            params = []

            if event_type:
                conditions.append("event_type = ?")
                params.append(event_type.value)
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            if strategy:
                conditions.append("strategy = ?")
                params.append(strategy)
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            result = con.execute(f"""
                SELECT * FROM audit_events
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """, params + [limit]).fetchall()

            columns = ["id", "event_type", "timestamp", "symbol", "strategy",
                      "side", "quantity", "price", "message", "details"]
            return [dict(zip(columns, row)) for row in result]
        finally:
            con.close()

    def get_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get summary statistics for recent events."""
        con = self._get_connection()
        try:
            result = con.execute("""
                SELECT
                    event_type,
                    COUNT(*) as count
                FROM audit_events
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL ? DAY
                GROUP BY event_type
            """, [days]).fetchall()

            return {row[0]: row[1] for row in result}
        finally:
            con.close()

    def get_trade_pairs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get matched entry/exit pairs for performance attribution.
        
        Returns list of completed trades with entry and exit information.
        """
        con = self._get_connection()
        try:
            conditions = ["event_type = 'ORDER_EXECUTION'"]
            params = []
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            if strategy:
                conditions.append("strategy = ?")
                params.append(strategy)
                
            where_clause = " AND ".join(conditions)
            
            result = con.execute(f"""
                SELECT 
                    symbol,
                    strategy,
                    side,
                    quantity,
                    price,
                    timestamp,
                    message,
                    details
                FROM audit_events
                WHERE {where_clause}
                ORDER BY symbol, strategy, timestamp
            """, params).fetchall()
            
            columns = ["symbol", "strategy", "side", "quantity", "price", "timestamp", "message", "details"]
            return [dict(zip(columns, row)) for row in result]
        finally:
            con.close()

    def get_execution_quality_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Calculate execution quality metrics from audit trail.
        
        Compares ORDER_DECISION prices (intended) vs ORDER_EXECUTION prices (actual).
        """
        con = self._get_connection()
        try:
            conditions = []
            params = []
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
                
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            decisions = con.execute(f"""
                SELECT symbol, strategy, price, timestamp
                FROM audit_events
                WHERE event_type = 'ORDER_DECISION' AND {where_clause}
                ORDER BY timestamp
            """, params).fetchall()
            
            executions = con.execute(f"""
                SELECT symbol, strategy, price, timestamp
                FROM audit_events
                WHERE event_type = 'ORDER_EXECUTION' AND {where_clause}
                ORDER BY timestamp
            """, params).fetchall()
            
            if not decisions or not executions:
                return {
                    "total_decisions": len(decisions),
                    "total_executions": len(executions),
                    "execution_rate": 0.0,
                    "avg_slippage_bps": 0.0
                }
                
            execution_rate = len(executions) / len(decisions) if decisions else 0
            
            return {
                "total_decisions": len(decisions),
                "total_executions": len(executions),
                "execution_rate": execution_rate
            }
        finally:
            con.close()

    def get_signal_to_execution_lag(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Analyze lag between signal generation and execution.
        
        Useful for understanding timing delays in the trading system.
        """
        con = self._get_connection()
        try:
            conditions = []
            params = []
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
                
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            result = con.execute(f"""
                WITH signals AS (
                    SELECT symbol, strategy, timestamp as signal_time
                    FROM audit_events
                    WHERE event_type = 'SIGNAL' AND {where_clause}
                ),
                executions AS (
                    SELECT symbol, strategy, timestamp as exec_time
                    FROM audit_events
                    WHERE event_type = 'ORDER_EXECUTION' AND {where_clause}
                )
                SELECT 
                    s.symbol,
                    s.strategy,
                    s.signal_time,
                    e.exec_time,
                    EXTRACT(EPOCH FROM (e.exec_time - s.signal_time)) as lag_seconds
                FROM signals s
                JOIN executions e 
                    ON s.symbol = e.symbol 
                    AND s.strategy = e.strategy
                    AND e.exec_time >= s.signal_time
                    AND e.exec_time <= s.signal_time + INTERVAL '5 minutes'
            """, params).fetchall()
            
            if not result:
                return {
                    "count": 0,
                    "avg_lag_seconds": 0.0,
                    "median_lag_seconds": 0.0,
                    "max_lag_seconds": 0.0
                }
                
            lags = [row[4] for row in result if row[4] is not None]
            
            if not lags:
                return {
                    "count": 0,
                    "avg_lag_seconds": 0.0,
                    "median_lag_seconds": 0.0,
                    "max_lag_seconds": 0.0
                }
                
            import statistics
            return {
                "count": len(lags),
                "avg_lag_seconds": statistics.mean(lags),
                "median_lag_seconds": statistics.median(lags),
                "max_lag_seconds": max(lags),
                "min_lag_seconds": min(lags)
            }
        finally:
            con.close()


# Global instance
AUDIT_TRAIL = AuditTrail()
