"""
Test TCA Integration.
Simulates order logging and filling, then checks stats.

Uses pytest tmp_path fixture to ensure isolated test database.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from openquant.analysis.tca import TCAMonitor


def test_tca(tmp_path: Path):
    """Test TCA order logging, fill update, and stats calculation.

    Uses tmp_path fixture to create isolated test database.
    This avoids path issues when running from different directories.
    """
    print("\n--- Testing TCA Module ---")

    # Use tmp_path for isolated test database
    db_path = str(tmp_path / "test_tca.duckdb")

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

    assert stats["count"] == 1, f"Expected count=1, got {stats['count']}"
    assert abs(stats["avg_slippage_bps"] - 5.0) < 0.1, \
        f"Expected slippage ~5 bps, got {stats['avg_slippage_bps']}"
    print("TCA Logic Verified (Slippage calc correct)")


if __name__ == "__main__":
    # For manual testing, create a temp directory
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_tca(Path(tmpdir))
