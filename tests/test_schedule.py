"""Test schedule checking functionality.

Tests the check_schedule function from paper_apply_allocation script.
Uses module-level mocks for ccxt and duckdb to allow importing the script,
but restores them after the module is loaded to avoid polluting other tests.
"""
import sys
import unittest
from datetime import datetime, time
from unittest.mock import MagicMock, patch

# Save original modules before mocking
_original_ccxt = sys.modules.get("ccxt")
_original_duckdb = sys.modules.get("duckdb")

# Mock ccxt and duckdb before importing the script
# These are required because paper_apply_allocation imports them at module level
sys.modules["ccxt"] = MagicMock()
sys.modules["duckdb"] = MagicMock()

# Import the function to test
from scripts.paper_apply_allocation import check_schedule

# Restore original modules after import to avoid polluting other tests
if _original_ccxt is not None:
    sys.modules["ccxt"] = _original_ccxt
else:
    del sys.modules["ccxt"]

if _original_duckdb is not None:
    sys.modules["duckdb"] = _original_duckdb
else:
    del sys.modules["duckdb"]


class TestSchedule(unittest.TestCase):
    """Test schedule checking for trading hours and days."""

    @patch('scripts.paper_apply_allocation.datetime')
    def test_days(self, mock_dt):
        """Test day-of-week schedule checking."""
        # Mock Monday
        mock_dt.utcnow.return_value = datetime(2025, 1, 6, 12, 0, 0)  # Jan 6 2025 is Monday

        self.assertTrue(check_schedule(None, "MON,TUE"))
        self.assertFalse(check_schedule(None, "WED,THU"))
        self.assertTrue(check_schedule(None, "mon,tue"))  # Case insensitive check

    @patch('scripts.paper_apply_allocation.datetime')
    def test_hours(self, mock_dt):
        """Test hour-of-day schedule checking including midnight crossing."""
        # Mock 10:00
        mock_dt.utcnow.return_value = datetime(2025, 1, 6, 10, 0, 0)
        mock_dt.strptime = datetime.strptime  # Restore strptime

        self.assertTrue(check_schedule("09:00-17:00", None))
        self.assertFalse(check_schedule("11:00-17:00", None))

        # Midnight crossing
        # Mock 23:00
        mock_dt.utcnow.return_value = datetime(2025, 1, 6, 23, 0, 0)
        self.assertTrue(check_schedule("22:00-02:00", None))

        # Mock 01:00
        mock_dt.utcnow.return_value = datetime(2025, 1, 6, 1, 0, 0)
        self.assertTrue(check_schedule("22:00-02:00", None))

        # Mock 03:00
        mock_dt.utcnow.return_value = datetime(2025, 1, 6, 3, 0, 0)
        self.assertFalse(check_schedule("22:00-02:00", None))


if __name__ == "__main__":
    unittest.main()
