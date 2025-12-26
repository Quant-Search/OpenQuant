"""
Input Validation Module for OpenQuant
======================================
Provides validation functions for all user inputs and data.
"""
import pandas as pd
import numpy as np
from typing import List, Optional, Union, Any
from dataclasses import dataclass


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    message: str
    value: Any = None


def validate_symbol(symbol: str) -> ValidationResult:
    """
    Validate a trading symbol.
    
    Args:
        symbol: Trading symbol (e.g., EURUSD)
        
    Returns:
        ValidationResult
    """
    if not symbol or not isinstance(symbol, str):
        return ValidationResult(False, "Symbol must be a non-empty string")
    
    symbol = symbol.strip().upper()
    
    if len(symbol) < 3:
        return ValidationResult(False, f"Symbol '{symbol}' is too short")
    
    if len(symbol) > 10:
        return ValidationResult(False, f"Symbol '{symbol}' is too long")
    
    if not symbol.isalnum():
        return ValidationResult(False, f"Symbol '{symbol}' contains invalid characters")
    
    return ValidationResult(True, "Valid symbol", symbol)


def validate_timeframe(timeframe: str) -> ValidationResult:
    """
    Validate a timeframe string.
    
    Args:
        timeframe: Timeframe (e.g., H1, H4, D1)
        
    Returns:
        ValidationResult
    """
    valid_timeframes = {"M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1",
                        "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"}
    
    if not timeframe or not isinstance(timeframe, str):
        return ValidationResult(False, "Timeframe must be a non-empty string")
    
    tf = timeframe.strip().upper()
    tf_lower = timeframe.strip().lower()
    
    if tf not in valid_timeframes and tf_lower not in valid_timeframes:
        return ValidationResult(
            False, 
            f"Invalid timeframe '{timeframe}'. Valid: {sorted(valid_timeframes)}"
        )
    
    return ValidationResult(True, "Valid timeframe", timeframe)


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str] = None,
    min_rows: int = 1
) -> ValidationResult:
    """
    Validate a DataFrame for trading operations.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        ValidationResult
    """
    if df is None:
        return ValidationResult(False, "DataFrame is None")
    
    if not isinstance(df, pd.DataFrame):
        return ValidationResult(False, f"Expected DataFrame, got {type(df).__name__}")
    
    if df.empty:
        return ValidationResult(False, "DataFrame is empty")
    
    if len(df) < min_rows:
        return ValidationResult(False, f"DataFrame has {len(df)} rows, need at least {min_rows}")
    
    required_columns = required_columns or ["Open", "High", "Low", "Close"]
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        return ValidationResult(False, f"Missing required columns: {missing}")
    
    # Check for NaN values in price columns
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    nan_counts = df[price_cols].isna().sum()
    if nan_counts.any():
        return ValidationResult(False, f"NaN values found: {nan_counts.to_dict()}")
    
    return ValidationResult(True, "Valid DataFrame", df)


def validate_positive_number(
    value: Union[int, float],
    name: str = "value",
    allow_zero: bool = False
) -> ValidationResult:
    """Validate that a number is positive."""
    if value is None:
        return ValidationResult(False, f"{name} cannot be None")
    
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ValidationResult(False, f"{name} must be a number, got {type(value).__name__}")
    
    if np.isnan(num) or np.isinf(num):
        return ValidationResult(False, f"{name} cannot be NaN or Inf")
    
    if allow_zero:
        if num < 0:
            return ValidationResult(False, f"{name} must be >= 0, got {num}")
    else:
        if num <= 0:
            return ValidationResult(False, f"{name} must be > 0, got {num}")
    
    return ValidationResult(True, f"Valid {name}", num)


def validate_range(
    value: Union[int, float],
    min_val: float,
    max_val: float,
    name: str = "value"
) -> ValidationResult:
    """Validate that a number is within a range."""
    result = validate_positive_number(value, name, allow_zero=True)
    if not result.is_valid:
        return result
    
    if not (min_val <= result.value <= max_val):
        return ValidationResult(
            False, 
            f"{name} must be between {min_val} and {max_val}, got {result.value}"
        )
    
    return ValidationResult(True, f"Valid {name}", result.value)

