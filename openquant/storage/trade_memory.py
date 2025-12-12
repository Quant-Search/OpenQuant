"""Persistent Trade Memory using DuckDB.

Stores detailed trade history including features and market conditions
to enable post-mortem analysis and online learning.
"""
import duckdb
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

class TradeMemory:
    """
    Manages persistent storage of trade history.
    """
    def __init__(self, db_path: str = "data/trades_history.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path))
        self._init_schema()
        
    def _init_schema(self):
        """Initialize database schema."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id VARCHAR,
                symbol VARCHAR,
                side VARCHAR,
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                entry_price DOUBLE,
                exit_price DOUBLE,
                quantity DOUBLE,
                pnl DOUBLE,
                pnl_pct DOUBLE,
                exit_reason VARCHAR,
                market_regime VARCHAR,
                features_json JSON,
                strategy_name VARCHAR
            );
        """)
        
    def save_trade(self, trade_data: Dict[str, Any]):
        """
        Save a closed trade to the database.
        
        Args:
            trade_data: Dictionary containing trade details.
        """
        try:
            # Ensure features are JSON string
            if isinstance(trade_data.get("features_json"), dict):
                trade_data["features_json"] = json.dumps(trade_data["features_json"])
                
            # Prepare query
            keys = [
                "id", "symbol", "side", "entry_time", "exit_time", 
                "entry_price", "exit_price", "quantity", "pnl", "pnl_pct",
                "exit_reason", "market_regime", "features_json", "strategy_name"
            ]
            
            # Fill missing keys with None
            values = [trade_data.get(k) for k in keys]
            
            placeholders = ", ".join(["?"] * len(keys))
            col_names = ", ".join(keys)
            
            query = f"INSERT INTO trades ({col_names}) VALUES ({placeholders})"
            
            self.con.execute(query, values)
            LOGGER.info(f"Saved trade {trade_data.get('symbol')} PnL: {trade_data.get('pnl_pct', 0):.2%}")
            
        except Exception as e:
            LOGGER.error(f"Failed to save trade: {e}")
            
    def load_recent_trades(self, limit: int = 100) -> pd.DataFrame:
        """Load recent trades."""
        return self.con.execute(f"SELECT * FROM trades ORDER BY exit_time DESC LIMIT {limit}").df()
        
    def get_losing_trades(self, limit: int = 100) -> pd.DataFrame:
        """Get trades with negative PnL."""
        return self.con.execute(f"SELECT * FROM trades WHERE pnl < 0 ORDER BY exit_time DESC LIMIT {limit}").df()
        
    def close(self):
        """Close connection."""
        self.con.close()
