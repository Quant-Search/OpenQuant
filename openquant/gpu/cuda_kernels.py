"""GPU-accelerated computation kernels using CuPy.

This module provides CUDA-accelerated implementations of key algorithms:
- Kalman Filter (mean reversion strategy)
- Hurst Exponent (regime detection)
- ATR Calculation (volatility-based TP/SL)

All functions accept NumPy arrays and return NumPy arrays,
with automatic GPU memory management via CuPy.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

# Lazy import CuPy (may not be available on all systems)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    
    # Test GPU availability
    try:
        cp.cuda.Device(0).compute_capability
        GPU_AVAILABLE = True
    except:
        GPU_AVAILABLE = False
        CUPY_AVAILABLE = False
except ImportError:
    CUPY_AVAILABLE = False
    GPU_AVAILABLE = False
    cp = None


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return GPU_AVAILABLE


def kalman_filter_gpu(
    prices: np.ndarray,
    process_noise: float = 1e-5,
    measurement_noise: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated Kalman Filter for mean reversion.
    
    This runs the Kalman filter on the GPU, processing large price arrays
    much faster than the CPU version.
    
    Args:
        prices: Price series (1D numpy array)
        process_noise: Process noise variance (Q)
        measurement_noise: Measurement noise variance (R)
        
    Returns:
        Tuple of (estimated_prices, deviations) as numpy arrays
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available for Kalman filter")
        
    n = len(prices)
    
    # Transfer data to GPU
    prices_gpu = cp.asarray(prices, dtype=cp.float32)
    
    # Kalman filter state
    x = prices_gpu[0]  # Initial state estimate
    P = cp.float32(1.0)  # Initial covariance
    Q = cp.float32(process_noise)
    R = cp.float32(measurement_noise)
    
    # Preallocate GPU arrays
    estimated_prices = cp.zeros(n, dtype=cp.float32)
    deviations = cp.zeros(n, dtype=cp.float32)
    
    # Kalman filter loop (sequential, but on GPU for vectorized ops)
    for i in range(n):
        # Prediction
        x_pred = x
        P_pred = P + Q
        
        # Update
        z = prices_gpu[i]
        K = P_pred / (P_pred + R)  # Kalman gain
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        
        estimated_prices[i] = x
        deviations[i] = z - x
    
    # Transfer back to CPU
    return cp.asnumpy(estimated_prices), cp.asnumpy(deviations)


def batch_kalman_filter_gpu(
    price_matrix: np.ndarray,
    process_noise: float = 1e-5,
    measurement_noise: float = 1e-3
) -> Tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated batch Kalman Filter for multiple symbols.
    
    Process multiple price series in parallel on GPU.
    
    Args:
        price_matrix: 2D array where each row is a different symbol's prices
        process_noise: Process noise variance
        measurement_noise: Measurement noise variance
        
    Returns:
        Tuple of (estimated_prices, deviations) matrices as numpy arrays
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available for batch Kalman filter")
        
    n_symbols, n_bars = price_matrix.shape
    
    # Transfer to GPU
    prices_gpu = cp.asarray(price_matrix, dtype=cp.float32)
    
    # Vectorized Kalman filter for all symbols
    # Initial states
    x = prices_gpu[:, 0].copy()  # (n_symbols,)
    P = cp.ones(n_symbols, dtype=cp.float32)
    Q = cp.float32(process_noise)
    R = cp.float32(measurement_noise)
    
    # Preallocate
    estimated = cp.zeros_like(prices_gpu)
    deviations = cp.zeros_like(prices_gpu)
    
    # Process all symbols in parallel
    for i in range(n_bars):
        # Prediction (vectorized across all symbols)
        x_pred = x
        P_pred = P + Q
        
        # Update
        z = prices_gpu[:, i]
        K = P_pred / (P_pred + R)
        x = x_pred + K * (z - x_pred)
        P = (1 - K) * P_pred
        
        estimated[:, i] = x
        deviations[:, i] = z - x
    
    return cp.asnumpy(estimated), cp.asnumpy(deviations)


def hurst_exponent_gpu(prices: np.ndarray, lags: np.ndarray) -> float:
    """GPU-accelerated Hurst Exponent calculation.
    
    Computes the Hurst exponent using rescaled range (R/S) analysis.
    H < 0.5: Mean reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    
    Args:
        prices: Price series (1D numpy array)
        lags: Array of lag values to use
        
    Returns:
        Hurst exponent (float)
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available for Hurst exponent")
        
    prices_gpu = cp.asarray(prices, dtype=cp.float32)
    lags_gpu = cp.asarray(lags, dtype=cp.int32)
    
    n_lags = len(lags)
    RS = cp.zeros(n_lags, dtype=cp.float32)
    
    # Calculate R/S for each lag
    for idx, lag in enumerate(lags_gpu):
        lag = int(lag)
        
        # Split into chunks
        n_chunks = len(prices_gpu) // lag
        if n_chunks == 0:
            continue
            
        # Reshape to chunks (may truncate data)
        chunks = prices_gpu[:n_chunks * lag].reshape(n_chunks, lag)
        
        # Calculate mean and cumulative deviations for each chunk
        means = cp.mean(chunks, axis=1, keepdims=True)
        deviations = chunks - means
        cumdev = cp.cumsum(deviations, axis=1)
        
        # Range (max - min)
        R = cp.max(cumdev, axis=1) - cp.min(cumdev, axis=1)
        
        # Standard deviation
        S = cp.std(chunks, axis=1)
        
        # Avoid division by zero
        S = cp.where(S == 0, 1e-10, S)
        
        # R/S ratio
        RS[idx] = cp.mean(R / S)
    
    # Filter out invalid values
    valid_mask = (RS > 0) & cp.isfinite(RS)
    if cp.sum(valid_mask) < 2:
        # Fallback: return 0.5 (random walk)
        return 0.5
        
    RS_valid = RS[valid_mask]
    lags_valid = lags_gpu[valid_mask]
    
    # Linear regression in log-log space
    log_lags = cp.log(lags_valid.astype(cp.float32))
    log_RS = cp.log(RS_valid)
    
    # Hurst = slope of log(R/S) vs log(lag)
    # Using least squares: H = cov(x,y) / var(x)
    n = len(log_lags)
    mean_x = cp.mean(log_lags)
    mean_y = cp.mean(log_RS)
    
    cov_xy = cp.sum((log_lags - mean_x) * (log_RS - mean_y))
    var_x = cp.sum((log_lags - mean_x) ** 2)
    
    if var_x == 0:
        return 0.5
        
    hurst = float(cp.asnumpy(cov_xy / var_x))
    
    # Clamp to reasonable range
    return max(0.0, min(1.0, hurst))


def atr_gpu(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """GPU-accelerated Average True Range calculation.
    
    Computes ATR much faster on GPU for large datasets.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)
        
    Returns:
        ATR series as numpy array
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available for ATR calculation")
        
    # Transfer to GPU
    high_gpu = cp.asarray(high, dtype=cp.float32)
    low_gpu = cp.asarray(low, dtype=cp.float32)
    close_gpu = cp.asarray(close, dtype=cp.float32)
    
    # True Range calculation (vectorized)
    prev_close = cp.roll(close_gpu, 1)
    prev_close[0] = close_gpu[0]  # No previous for first bar
    
    tr1 = high_gpu - low_gpu
    tr2 = cp.abs(high_gpu - prev_close)
    tr3 = cp.abs(low_gpu - prev_close)
    
    tr = cp.maximum(tr1, cp.maximum(tr2, tr3))
    
    # ATR as exponential moving average of TR
    # Using simple moving average for first value, then EMA
    alpha = 1.0 / period
    atr = cp.zeros_like(tr)
    
    # Initial SMA
    atr[period-1] = cp.mean(tr[:period])
    
    # EMA for rest
    for i in range(period, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]
    
    # Forward fill initial values
    atr[:period-1] = atr[period-1]
    
    return cp.asnumpy(atr)


def batch_atr_gpu(
    high_matrix: np.ndarray,
    low_matrix: np.ndarray,
    close_matrix: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """GPU-accelerated batch ATR for multiple symbols.
    
    Args:
        high_matrix: 2D array (n_symbols, n_bars)
        low_matrix: 2D array (n_symbols, n_bars)
        close_matrix: 2D array (n_symbols, n_bars)
        period: ATR period
        
    Returns:
        ATR matrix as numpy array (n_symbols, n_bars)
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available for batch ATR")
        
    # Transfer to GPU
    high_gpu = cp.asarray(high_matrix, dtype=cp.float32)
    low_gpu = cp.asarray(low_matrix, dtype=cp.float32)
    close_gpu = cp.asarray(close_matrix, dtype=cp.float32)
    
    n_symbols, n_bars = high_gpu.shape
    
    # Vectorized TR calculation across all symbols
    prev_close = cp.roll(close_gpu, 1, axis=1)
    prev_close[:, 0] = close_gpu[:, 0]
    
    tr1 = high_gpu - low_gpu
    tr2 = cp.abs(high_gpu - prev_close)
    tr3 = cp.abs(low_gpu - prev_close)
    
    tr = cp.maximum(tr1, cp.maximum(tr2, tr3))
    
    # ATR for each symbol
    alpha = 1.0 / period
    atr = cp.zeros_like(tr)
    
    # Initial SMA
    atr[:, period-1] = cp.mean(tr[:, :period], axis=1)
    
    # EMA
    for i in range(period, n_bars):
        atr[:, i] = alpha * tr[:, i] + (1 - alpha) * atr[:, i-1]
    
    # Forward fill
    for i in range(period-1):
        atr[:, i] = atr[:, period-1]
    
    return cp.asnumpy(atr)
