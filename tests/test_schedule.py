import sys
import unittest
from datetime import datetime, time
from unittest.mock import MagicMock, patch

# Mock ccxt before importing the script
sys.modules["ccxt"] = MagicMock()
sys.modules["duckdb"] = MagicMock()

# Import the function to test. 
# Since it's in a script, we might need to import it carefully or extract it.
# For now, let's copy the logic or try to import if possible.
# The script has `if __name__ == "__main__":`, so it is importable.
from scripts.paper_apply_allocation import check_schedule

class TestSchedule(unittest.TestCase):
    @patch('scripts.paper_apply_allocation.datetime')
    def test_days(self, mock_dt):
        # Mock Monday
        mock_dt.utcnow.return_value = datetime(2025, 1, 6, 12, 0, 0) # Jan 6 2025 is Monday
        
        self.assertTrue(check_schedule(None, "MON,TUE"))
        self.assertFalse(check_schedule(None, "WED,THU"))
        self.assertTrue(check_schedule(None, "mon,tue")) # Case insensitive check

    @patch('scripts.paper_apply_allocation.datetime')
    def test_hours(self, mock_dt):
        # Mock 10:00
        mock_dt.utcnow.return_value = datetime(2025, 1, 6, 10, 0, 0)
        mock_dt.strptime = datetime.strptime # Restore strptime
        
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
