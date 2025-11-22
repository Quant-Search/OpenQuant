import unittest
from unittest.mock import MagicMock, patch
import shutil
from pathlib import Path
import tempfile
import csv

from openquant.paper.mt5_bridge import export_signals_to_csv

class TestMT5Export(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.local_csv = Path(self.test_dir) / "data" / "signals.csv"
        self.mt5_data_path = Path(self.test_dir) / "MT5_Data"
        self.mt5_csv = self.mt5_data_path / "MQL5" / "Files" / "signals.csv"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_export_signals(self):
        allocations = [
            {"symbol": "EURUSD", "weight": 0.5},
            {"symbol": "GBPUSD", "weight": -0.3},
            {"symbol": "USDJPY", "weight": 0.0}
        ]

        # Test with explicit mt5_data_path
        export_signals_to_csv(
            allocations, 
            path=str(self.local_csv), 
            mt5_data_path=str(self.mt5_data_path)
        )

        # Verify local file
        self.assertTrue(self.local_csv.exists())
        with open(self.local_csv, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            self.assertEqual(rows[0], ["Symbol", "Side", "Weight", "Timestamp"])
            self.assertEqual(rows[1][:3], ["EURUSD", "BUY", "0.5000"])
            self.assertEqual(rows[2][:3], ["GBPUSD", "SELL", "-0.3000"])
            # Flat might be skipped or included depending on logic? 
            # Logic says: if sym: rows.append...
            # weight 0 -> side FLAT.
            self.assertEqual(rows[3][:3], ["USDJPY", "FLAT", "0.0000"])

        # Verify MT5 file
        self.assertTrue(self.mt5_csv.exists())
        with open(self.mt5_csv, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            self.assertEqual(rows[0], ["Symbol", "Side", "Weight", "Timestamp"])
            self.assertEqual(rows[1][:3], ["EURUSD", "BUY", "0.5000"])

    @patch("openquant.paper.mt5_bridge._lazy_import")
    def test_auto_detect_path(self, mock_import):
        # Mock MT5 terminal info
        mock_mt5 = MagicMock()
        mock_info = MagicMock()
        mock_info.data_path = str(self.mt5_data_path)
        mock_mt5.terminal_info.return_value = mock_info
        mock_import.return_value = mock_mt5

        allocations = [{"symbol": "BTCUSD", "weight": 1.0}]
        
        export_signals_to_csv(allocations, path=str(self.local_csv))
        
        self.assertTrue(self.mt5_csv.exists())
        with open(self.mt5_csv, "r") as f:
            content = f.read()
            self.assertIn("BTCUSD", content)

if __name__ == "__main__":
    unittest.main()
