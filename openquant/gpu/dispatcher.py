"""Smart GPU/CPU dispatcher for automatic acceleration.

This module automatically uses GPU when:
1. GPU is available (CUDA + CuPy installed)
2. Dataset is large enough to benefit from GPU (overhead vs speedup)
3. Falls back gracefully to CPU if GPU fails or is unavailable

Usage:
    from openquant.gpu.dispatcher import gpu_dispatch
    
    result = gpu_dispatch(
        gpu_func=lambda arr: kalman_filter_gpu(arr, ...),
        cpu_func=lambda arr: kalman_filter_cpu(arr, ...),
        data=prices,
        min_size=100  # Only use GPU if len(prices) >= 100
    )
"""
from __future__ import annotations
from typing import Callable, Any, Optional
import numpy as np

from .cuda_kernels import is_gpu_available
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

# Global GPU policy
_GPU_ENABLED = True
_MIN_DATASET_SIZE = 100  # Don't use GPU for small datasets


def set_gpu_policy(enabled: bool = True, min_dataset_size: int = 100):
    """Configure global GPU acceleration policy.
    
    Args:
        enabled: Enable/disable GPU acceleration globally
        min_dataset_size: Minimum data size to use GPU (avoid overhead for small data)
    """
    global _GPU_ENABLED, _MIN_DATASET_SIZE
    _GPU_ENABLED = enabled
    _MIN_DATASET_SIZE = min_dataset_size
    
    LOGGER.info(f"GPU policy: enabled={enabled}, min_size={min_dataset_size}")


def get_gpu_policy() -> dict:
    """Get current GPU policy settings."""
    return {
        "enabled": _GPU_ENABLED,
        "available": is_gpu_available(),
        "min_dataset_size": _MIN_DATASET_SIZE
    }


def should_use_gpu(data_size: int, force_gpu: Optional[bool] = None) -> bool:
    """Determine if GPU should be used for given dataset size.
    
    Args:
        data_size: Size of the dataset to process
        force_gpu: If True, force GPU usage; if False, force CPU; if None, use policy
        
    Returns:
        True if GPU should be used
    """
    if force_gpu is True:
        return is_gpu_available()
    if force_gpu is False:
        return False
        
    # Use policy
    return (
        _GPU_ENABLED and
        is_gpu_available() and
        data_size >= _MIN_DATASET_SIZE
    )


def gpu_dispatch(
    gpu_func: Callable,
    cpu_func: Callable,
    data: Any,
    min_size: Optional[int] = None,
    force_gpu: Optional[bool] = None,
    **kwargs
) -> Any:
    """Dispatch computation to GPU or CPU based on policy and data size.
    
    This is the main entry point for GPU-accelerated computations.
    It handles:
    - Automatic GPU/CPU selection
    - Graceful fallback on GPU errors
    - Logging of GPU usage
    
    Args:
        gpu_func: Function to call on GPU (must accept numpy arrays)
        cpu_func: Function to call on CPU (fallback)
        data: Input data (numpy array or list of arrays)
        min_size: Override global min dataset size
        force_gpu: Force GPU (True) or CPU (False), or use policy (None)
        **kwargs: Additional arguments passed to both functions
        
    Returns:
        Result from either GPU or CPU function
        
    Example:
        # Automatic dispatch
        estimated, deviations = gpu_dispatch(
            gpu_func=lambda p: kalman_filter_gpu(p, Q=1e-5, R=1e-3),
            cpu_func=lambda p: kalman_filter_cpu(p, Q=1e-5, R=1e-3),
            data=prices
        )
    """
    # Determine data size
    if isinstance(data, np.ndarray):
        data_size = len(data)
    elif isinstance(data, (list, tuple)):
        data_size = len(data[0]) if data else 0
    else:
        data_size = 0
        
    # Use custom min_size if provided
    if min_size is not None:
        effective_min_size = min_size
    else:
        effective_min_size = _MIN_DATASET_SIZE
        
    # Check if we should use GPU
    use_gpu = should_use_gpu(data_size, force_gpu)
    
    if not use_gpu:
        # Use CPU
        try:
            result = cpu_func(data, **kwargs) if kwargs else cpu_func(data)
            return result
        except Exception as e:
            LOGGER.error(f"CPU function failed: {e}")
            raise
            
    # Try GPU
    try:
        result = gpu_func(data, **kwargs) if kwargs else gpu_func(data)
        LOGGER.debug(f"GPU computation successful (size={data_size})")
        return result
        
    except Exception as e:
        LOGGER.warning(f"GPU computation failed, falling back to CPU: {e}")
        
        # Fallback to CPU
        try:
            result = cpu_func(data, **kwargs) if kwargs else cpu_func(data)
            return result
        except Exception as cpu_e:
            LOGGER.error(f"CPU fallback also failed: {cpu_e}")
            raise cpu_e


# Convenience wrappers for common operations

def kalman_filter(prices: np.ndarray, process_noise: float = 1e-5, measurement_noise: float = 1e-3):
    """Kalman filter with automatic GPU dispatch.
    
    Args:
        prices: Price series
        process_noise: Process noise variance
        measurement_noise: Measurement noise variance
        
    Returns:
        Tuple of (estimated_prices, deviations)
    """
    from .cuda_kernels import kalman_filter_gpu
    
    def cpu_kalman(p):
        # CPU implementation (existing)
        n = len(p)
        x = p[0]
        P = 1.0
        Q = process_noise
        R = measurement_noise
        
        estimated = np.zeros(n)
        deviations = np.zeros(n)
        
        for i in range(n):
            x_pred = x
            P_pred = P + Q
            z = p[i]
            K = P_pred / (P_pred + R)
            x = x_pred + K * (z - x_pred)
            P = (1 - K) * P_pred
            estimated[i] = x
            deviations[i] = z - x
            
        return estimated, deviations
    
    return gpu_dispatch(
        gpu_func=lambda p: kalman_filter_gpu(p, process_noise, measurement_noise),
        cpu_func=cpu_kalman,
        data=prices
    )


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
    """ATR calculation with automatic GPU dispatch.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        
    Returns:
        ATR series
    """
    from .cuda_kernels import atr_gpu
    
    def cpu_atr(data):
        h, l, c = data
        n = len(h)
        tr = np.zeros(n)
        
        tr[0] = h[0] - l[0]
        for i in range(1, n):
            tr[i] = max(
                h[i] - l[i],
                abs(h[i] - c[i-1]),
                abs(l[i] - c[i-1])
            )
        
        # Simple moving average for initial value
        atr = np.zeros(n)
        atr[period-1] = np.mean(tr[:period])
        
        # EMA for rest
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
        
        # Forward fill
        atr[:period-1] = atr[period-1]
        
        return atr
    
    return gpu_dispatch(
        gpu_func=lambda d: atr_gpu(d[0], d[1], d[2], period),
        cpu_func=cpu_atr,
        data=(high, low, close)
    )
