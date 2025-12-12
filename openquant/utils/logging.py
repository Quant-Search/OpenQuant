"""
Structured logging utilities for OpenQuant.

Provides JSON-formatted logs with daily rotation and sensitive data redaction.
"""
import logging
import json
import os
import re
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Sensitive field patterns to redact
SENSITIVE_PATTERNS = [
    r'(?i)(password|passwd|pwd)',
    r'(?i)(api[_-]?key|apikey)',
    r'(?i)(secret|token)',
    r'(?i)(auth)',
]

class SensitiveDataFilter(logging.Filter):
    """Filter to redact sensitive data from log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for pattern in SENSITIVE_PATTERNS:
                record.msg = re.sub(
                    rf'{pattern}[\'\"]?\s*[:=]\s*[\'\"]?([^\s\'"]+)',
                    r'\1=***REDACTED***',
                    record.msg,
                    flags=re.IGNORECASE
                )
        return True

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'trade_id'):
            log_data['trade_id'] = record.trade_id
        if hasattr(record, 'symbol'):
            log_data['symbol'] = record.symbol
        if hasattr(record, 'strategy'):
            log_data['strategy'] = record.strategy
        if hasattr(record, 'action'):
            log_data['action'] = record.action
        if hasattr(record, 'quantity'):
            log_data['quantity'] = record.quantity
        if hasattr(record, 'price'):
            log_data['price'] = record.price
        if hasattr(record, 'reason'):
            log_data['reason'] = record.reason
            
        #Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Get a structured logger with JSON formatting and daily rotation.
    
    Args:
        name: Logger name (usually __name__)
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Use per-process log file to avoid Windows rename locks on rotation
    pid = os.getpid()
    date_str = datetime.now().strftime("%Y%m%d")
    logfile = os.path.join(log_dir, f"openquant_{date_str}_{pid}.log")
    json_handler = logging.FileHandler(filename=logfile, encoding='utf-8', delay=True)
    json_handler.setFormatter(JSONFormatter())
    json_handler.addFilter(SensitiveDataFilter())
    # Also write to a stable file name expected by tests/tools
    stable_logfile = os.path.join(log_dir, "openquant.log")
    json_handler2 = logging.FileHandler(filename=stable_logfile, encoding='utf-8', delay=True)
    json_handler2.setFormatter(JSONFormatter())
    json_handler2.addFilter(SensitiveDataFilter())
    
    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(SensitiveDataFilter())
    
    logger.addHandler(json_handler)
    logger.addHandler(json_handler2)
    logger.addHandler(console_handler)
    
    return logger

class TradeLogger:
    """Context manager for logging trade decisions and executions"""
    
    def __init__(self, logger: logging.Logger, symbol: str, strategy: str):
        self.logger = logger
        self.symbol = symbol
        self.strategy = strategy
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now(timezone.utc)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.logger.error(
                f"Trade execution failed for {self.symbol}",
                extra={
                    'symbol': self.symbol,
                    'strategy': self.strategy,
                    'action': 'trade_error',
                    'duration_ms': (datetime.now(timezone.utc) - self.start_time).total_seconds() * 1000
                },
                exc_info=True
            )
        return False
        
    def log_decision(self, action: str, quantity: float, price: float, reason: str):
        """Log a trading decision"""
        self.logger.info(
            f"Decision: {action} {quantity} {self.symbol} @ {price}",
            extra={
                'symbol': self.symbol,
                'strategy': self.strategy,
                'action': action,
                'quantity': quantity,
                'price': price,
                'reason': reason,
                'event_type': 'decision'
            }
        )
        
    def log_execution(self, trade_id: str, action: str, quantity: float, 
                     fill_price: float, slippage: float = 0.0):
        """Log a trade execution"""
        self.logger.info(
            f"Executed: {action} {quantity} {self.symbol} @ {fill_price}",
            extra={
                'trade_id': trade_id,
                'symbol': self.symbol,
                'strategy': self.strategy,
                'action': action,
                'quantity': quantity,
                'price': fill_price,
                'slippage': slippage,
                'event_type': 'execution',
                'duration_ms': (datetime.now(timezone.utc) - self.start_time).total_seconds() * 1000
            }
        )
