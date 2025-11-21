"""
Test TCA Integration.
Simulates order logging and filling, then checks stats.
"""
import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from openquant.analysis.tca import TCAMonitor

def test_tca():
    print("\n--- Testing TCA Module ---")
    
    # Use a test db
    db_path = "data/test_tca.duckdb"
    # Clean up
    if Path(db_path).exists():
        Path(db_path).unlink()
        
    tca = TCAMonitor(db_path=db_path)
    
    # 1. Log Order
    order_id = "test_order_1"
    arrival_price = 100.0
    print(f"Logging Order: {order_id} BUY 1.0 @ {arrival_price}")
    tca.log_order(order_id, "AAPL", "buy", 1.0, arrival_price)
    
    # 2. Update Fill (Slippage)
    # Fill at 100.05 (5 bps slippage)
    fill_price = 100.05
    print(f"Filling Order: {order_id} @ {fill_price}")
    tca.update_fill(order_id, fill_price, 1.0, fee=0.0)
    
    # 3. Check Stats
    stats = tca.get_stats()
    print(f"Stats: {stats}")
    
    assert stats["count"] == 1
    assert abs(stats["avg_slippage_bps"] - 5.0) < 0.01
    print("✅ TCA Logic Verified (Slippage calc correct)")
    
    # Clean up
    if Path(db_path).exists():
        Path(db_path).unlink()

if __name__ == "__main__":
    test_tca()
