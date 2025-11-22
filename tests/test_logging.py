"""
Tests for structured logging system.
"""
import unittest
import json
import tempfile
import shutil
from pathlib import Path
from openquant.utils.logging import get_logger, SensitiveDataFilter, TradeLogger

class TestLogging(unittest.TestCase):
    def setUp(self):
        """Create temporary log directory"""
        self.log_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary logs"""
        shutil.rmtree(self.log_dir, ignore_errors=True)
        
    def test_json_logging(self):
        """Test that logs are written in JSON format"""
        logger = get_logger(__name__, log_dir=self.log_dir)
        logger.info("Test message", extra={'symbol': 'BTC/USD', 'price': 50000.0})
        
        # Read the log file
        log_file = Path(self.log_dir) / 'openquant.log'
        self.assertTrue(log_file.exists())
        
        with open(log_file, 'r') as f:
            line = f.readline()
            log_data = json.loads(line)
            
        # Verify JSON structure
        self.assertIn('timestamp', log_data)
        self.assertIn('level', log_data)
        self.assertIn('message', log_data)
        self.assertEqual(log_data['symbol'], 'BTC/USD')
        self.assertEqual(log_data['price'], 50000.0)
        
    def test_sensitive_data_redaction(self):
        """Test that sensitive data is redacted"""
        import logging
        
        # Create a filter
        filter_obj = SensitiveDataFilter()
        
        # Create a fake log record
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg="Connecting with api_key=secret123 and password='mypass'",
            args=(), exc_info=None
        )
        
        # Apply filter
        filter_obj.filter(record)
        
        # Check that sensitive data is redacted
        self.assertIn('***REDACTED***', record.msg)
        self.assertNotIn('secret123', record.msg)
        self.assertNotIn('mypass', record.msg)
        
    def test_trade_logger(self):
        """Test TradeLogger context manager"""
        logger = get_logger(__name__, log_dir=self.log_dir)
        
        # Verify logging doesn't crash
        try:
            with TradeLogger(logger, 'ETH/USD', 'ml') as tl:
                tl.log_decision('buy', 10.0, 3000.0, 'ML signal positive')
                tl.log_execution('trade_123', 'buy', 10.0, 3005.0, slippage=5.0)
            success = True
        except Exception:
            success = False
            
        self.assertTrue(success, "TradeLogger should not crash")

if __name__ == "__main__":
    unittest.main()
