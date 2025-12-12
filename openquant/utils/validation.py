"""Input validation decorators and utilities for parameter bounds checking."""
from __future__ import annotations
import functools
from typing import Any, Callable, Dict, List, Optional, Set, Union
import numpy as np


class ValidationError(ValueError):
    """Raised when parameter validation fails."""
    pass


def validate_positive(value: float, name: str, allow_zero: bool = False) -> float:
    """Validate that a value is positive (optionally allowing zero).
    
    Args:
        value: Value to validate.
        name: Parameter name for error message.
        allow_zero: If True, allow zero values.
        
    Returns:
        The validated value.
        
    Raises:
        ValidationError: If value is not positive.
    """
    val = float(value)
    if allow_zero:
        if val < 0:
            raise ValidationError(f"{name} must be non-negative, got {val}")
    else:
        if val <= 0:
            raise ValidationError(f"{name} must be positive, got {val}")
    return val


def validate_range(value: float, name: str, min_val: Optional[float] = None, 
                   max_val: Optional[float] = None, inclusive: bool = True) -> float:
    """Validate that a value is within a specified range.
    
    Args:
        value: Value to validate.
        name: Parameter name for error message.
        min_val: Minimum allowed value (None for no minimum).
        max_val: Maximum allowed value (None for no maximum).
        inclusive: If True, endpoints are included in range.
        
    Returns:
        The validated value.
        
    Raises:
        ValidationError: If value is outside the specified range.
    """
    val = float(value)
    
    if min_val is not None:
        if inclusive and val < min_val:
            raise ValidationError(f"{name} must be >= {min_val}, got {val}")
        elif not inclusive and val <= min_val:
            raise ValidationError(f"{name} must be > {min_val}, got {val}")
    
    if max_val is not None:
        if inclusive and val > max_val:
            raise ValidationError(f"{name} must be <= {max_val}, got {val}")
        elif not inclusive and val >= max_val:
            raise ValidationError(f"{name} must be < {max_val}, got {val}")
    
    return val


def validate_probability(value: float, name: str) -> float:
    """Validate that a value is a valid probability (0 <= x <= 1).
    
    Args:
        value: Value to validate.
        name: Parameter name for error message.
        
    Returns:
        The validated value.
        
    Raises:
        ValidationError: If value is not a valid probability.
    """
    return validate_range(value, name, min_val=0.0, max_val=1.0, inclusive=True)


def validate_positive_int(value: int, name: str, allow_zero: bool = False) -> int:
    """Validate that a value is a positive integer.
    
    Args:
        value: Value to validate.
        name: Parameter name for error message.
        allow_zero: If True, allow zero values.
        
    Returns:
        The validated value.
        
    Raises:
        ValidationError: If value is not a positive integer.
    """
    val = int(value)
    if allow_zero:
        if val < 0:
            raise ValidationError(f"{name} must be non-negative integer, got {val}")
    else:
        if val <= 0:
            raise ValidationError(f"{name} must be positive integer, got {val}")
    return val


def validate_in_set(value: Any, name: str, valid_values: Set[Any]) -> Any:
    """Validate that a value is in a set of allowed values.
    
    Args:
        value: Value to validate.
        name: Parameter name for error message.
        valid_values: Set of allowed values.
        
    Returns:
        The validated value.
        
    Raises:
        ValidationError: If value is not in the allowed set.
    """
    if value not in valid_values:
        raise ValidationError(
            f"{name} must be one of {valid_values}, got {value}"
        )
    return value


def validate_timeframe(timeframe: str) -> str:
    """Validate that a timeframe string is valid.
    
    Args:
        timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d').
        
    Returns:
        The validated timeframe.
        
    Raises:
        ValidationError: If timeframe format is invalid.
    """
    valid_units = {'s', 'm', 'h', 'd', 'w', 'M', 'y'}
    valid_timeframes = {
        '1s', '5s', '15s', '30s',
        '1m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '12h',
        '1d', '1w', '1M', '1y'
    }
    
    if timeframe not in valid_timeframes:
        raise ValidationError(
            f"Invalid timeframe '{timeframe}'. Must be one of {sorted(valid_timeframes)}"
        )
    
    return timeframe


def normalize_weights(weights: List[float], name: str = "weights") -> List[float]:
    """Normalize weights to sum to 1.0.
    
    Args:
        weights: List of weight values.
        name: Parameter name for error messages.
        
    Returns:
        Normalized weights that sum to 1.0.
        
    Raises:
        ValidationError: If weights cannot be normalized (e.g., all zero, negative values).
    """
    if not weights:
        raise ValidationError(f"{name} cannot be empty")
    
    weights_array = np.array(weights, dtype=float)
    
    if np.any(weights_array < 0):
        raise ValidationError(f"{name} cannot contain negative values")
    
    total = np.sum(weights_array)
    
    if total == 0:
        raise ValidationError(f"{name} cannot all be zero")
    
    if not np.isfinite(total):
        raise ValidationError(f"{name} must contain finite values")
    
    return (weights_array / total).tolist()


def validate_params(**validators: Dict[str, Callable[[Any], Any]]) -> Callable:
    """Decorator to validate constructor parameters.
    
    Args:
        validators: Dict mapping parameter names to validation functions.
        
    Example:
        @validate_params(
            lookback=lambda x: validate_positive_int(x, 'lookback'),
            threshold=lambda x: validate_range(x, 'threshold', 0.0, 10.0)
        )
        def __init__(self, lookback: int, threshold: float):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature to map args to kwargs
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter that has a validator
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    # Skip validation for None if parameter has a default of None
                    if value is None:
                        param = sig.parameters.get(param_name)
                        if param and param.default is None:
                            continue
                    # Apply validation
                    bound_args.arguments[param_name] = validator(value)
            
            # Call original function with validated arguments
            return func(*bound_args.args, **bound_args.kwargs)
        
        return wrapper
    return decorator


def validate_positive_param(name: str, allow_zero: bool = False) -> Callable[[Any], float]:
    """Create a validator for positive float parameters.
    
    Args:
        name: Parameter name.
        allow_zero: If True, allow zero values.
        
    Returns:
        Validation function.
    """
    return lambda x: validate_positive(x, name, allow_zero)


def validate_positive_int_param(name: str, allow_zero: bool = False) -> Callable[[Any], int]:
    """Create a validator for positive integer parameters.
    
    Args:
        name: Parameter name.
        allow_zero: If True, allow zero values.
        
    Returns:
        Validation function.
    """
    return lambda x: validate_positive_int(x, name, allow_zero)


def validate_range_param(name: str, min_val: Optional[float] = None,
                        max_val: Optional[float] = None,
                        inclusive: bool = True) -> Callable[[Any], float]:
    """Create a validator for range-bounded parameters.
    
    Args:
        name: Parameter name.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
        inclusive: If True, endpoints are included.
        
    Returns:
        Validation function.
    """
    return lambda x: validate_range(x, name, min_val, max_val, inclusive)


def validate_probability_param(name: str) -> Callable[[Any], float]:
    """Create a validator for probability parameters (0 to 1).
    
    Args:
        name: Parameter name.
        
    Returns:
        Validation function.
    """
    return lambda x: validate_probability(x, name)


def validate_in_set_param(name: str, valid_values: Set[Any]) -> Callable[[Any], Any]:
    """Create a validator for parameters with a discrete set of valid values.
    
    Args:
        name: Parameter name.
        valid_values: Set of allowed values.
        
    Returns:
        Validation function.
    """
    return lambda x: validate_in_set(x, name, valid_values)
