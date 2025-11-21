"""Transaction Cost Analysis (TCA) module.
Tracks order execution quality (slippage, fees) and persists to DuckDB.
"""
import duckdb
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

class TCAMonitor:
    def __init__(self, db_path: str = "data/tca.duckdb"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize TCA database schema."""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(self.db_path) as con:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        order_id VARCHAR PRIMARY KEY,
                        symbol VARCHAR,
                        side VARCHAR,
                        quantity DOUBLE,
                        arrival_price DOUBLE,
                        created_at TIMESTAMP,
                        fill_price DOUBLE,
                        fill_qty DOUBLE,
                        fee DOUBLE,
                        filled_at TIMESTAMP,
                        slippage_bps DOUBLE,
                        status VARCHAR
                    )
                """)
        except Exception as e:
            LOGGER.error(f"Failed to init TCA db: {e}")

    def log_order(self, order_id: str, symbol: str, side: str, quantity: float, arrival_price: float):
        """Log a new order submission (Arrival)."""
        try:
            with duckdb.connect(self.db_path) as con:
                con.execute("""
                    INSERT INTO orders (order_id, symbol, side, quantity, arrival_price, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (order_id, symbol, side, quantity, arrival_price, datetime.now(timezone.utc), "NEW"))
            LOGGER.info(f"TCA: Logged order {order_id} {side} {symbol} @ {arrival_price}")
        except Exception as e:
            LOGGER.error(f"TCA log_order failed: {e}")

    def update_fill(self, order_id: str, fill_price: float, fill_qty: float, fee: float = 0.0):
        """Update an order with execution details."""
        try:
            with duckdb.connect(self.db_path) as con:
                # Calculate slippage
                # Slippage = (Exec - Arrival) / Arrival * 10000 (bps)
                # For BUY: Higher exec is bad (positive slippage = bad cost)
                # For SELL: Lower exec is bad (negative slippage? No, usually cost is defined as shortfall)
                # Implementation Shortfall: Side * (Exec - Arrival)
                # Let's stick to simple price diff relative to side.
                # Buy: (Fill - Arrival) / Arrival
                # Sell: (Arrival - Fill) / Arrival
                
                row = con.execute("SELECT side, arrival_price FROM orders WHERE order_id = ?", (order_id,)).fetchone()
                if not row:
                    LOGGER.warning(f"TCA: Order {order_id} not found for update.")
                    return

                side, arrival = row
                if arrival and arrival > 0:
                    if side.lower() == "buy":
                        slippage_bps = ((fill_price - arrival) / arrival) * 10000
                    else:
                        slippage_bps = ((arrival - fill_price) / arrival) * 10000
                else:
                    slippage_bps = 0.0

                con.execute("""
                    UPDATE orders 
                    SET fill_price = ?, fill_qty = ?, fee = ?, filled_at = ?, slippage_bps = ?, status = 'FILLED'
                    WHERE order_id = ?
                """, (fill_price, fill_qty, fee, datetime.now(timezone.utc), slippage_bps, order_id))
                
            LOGGER.info(f"TCA: Updated fill {order_id}. Slippage: {slippage_bps:.2f} bps")
        except Exception as e:
            LOGGER.error(f"TCA update_fill failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate TCA statistics."""
        try:
            with duckdb.connect(self.db_path) as con:
                df = con.execute("SELECT * FROM orders WHERE status = 'FILLED'").df()
                if df.empty:
                    return {"count": 0, "avg_slippage_bps": 0.0, "total_fees": 0.0}
                
                return {
                    "count": len(df),
                    "avg_slippage_bps": float(df["slippage_bps"].mean()),
                    "total_fees": float(df["fee"].sum()),
                    "worst_slippage": float(df["slippage_bps"].max()),
                    "recent_orders": df.tail(5).to_dict(orient="records")
                }
        except Exception as e:
            LOGGER.error(f"TCA get_stats failed: {e}")
            return {}
