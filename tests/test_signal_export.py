import unittest
import shutil
import csv
from pathlib import Path
from openquant.paper.mt5_bridge import export_signals_to_csv

class TestSignalExport(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/temp_signals")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.test_dir / "signals.csv"
        
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_export(self):
        alloc = [
            {"symbol": "EURUSD", "weight": 0.5},
            {"symbol": "BTCUSD", "weight": -0.2}
        ]
        
        export_signals_to_csv(alloc, str(self.csv_path))
        
        self.assertTrue(self.csv_path.exists())
        
        with open(self.csv_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            
            # Check Header
            self.assertEqual(rows[0], ["Symbol", "Side", "Weight", "Timestamp"])
            
            # Check Rows
            self.assertEqual(rows[1][0], "EURUSD")
            self.assertEqual(rows[1][1], "BUY")
            self.assertEqual(rows[1][2], "0.5000")
            
            self.assertEqual(rows[2][0], "BTCUSD")
            self.assertEqual(rows[2][1], "SELL")
            self.assertEqual(rows[2][2], "-0.2000")

if __name__ == "__main__":
    unittest.main()
