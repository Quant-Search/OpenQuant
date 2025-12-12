"""GPU-accelerated technical indicators and features using CuPy.

Provides vectorized implementations of common indicators (RSI, Bollinger, MACD, etc.)
that run on GPU.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

cp = None
try:
    import cupy as cp
except Exception:
    cp = None


def is_gpu_features_available() -> bool:
    if cp is None:
        return False
    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            return False
        _ = (cp.arange(8, dtype=cp.float32).sum()).item()
        return True
    except Exception:
        return False


def rsi_gpu(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index (RSI) on GPU."""
    if not is_gpu_features_available():
        raise RuntimeError("GPU not available for RSI")
        
    close_gpu = cp.asarray(close, dtype=cp.float32)
    delta = cp.diff(close_gpu, prepend=close_gpu[0])
    
    gain = cp.where(delta > 0, delta, 0)
    loss = cp.where(delta < 0, -delta, 0)
    
    # Exponential Moving Average for Gain/Loss
    # Note: Standard RSI uses Wilder's Smoothing which is equivalent to EMA(2*n-1)
    # But often implemented as simple EMA(n) or SMA(n) followed by EMA.
    # Here we use standard Wilder's smoothing:
    # avg_gain = (prev_avg_gain * (period - 1) + current_gain) / period
    
    alpha = 1.0 / period
    
    # We can use a linear filter or iterative loop. 
    # For speed on GPU with sequential dependency, simple loop is okay-ish 
    # but CuPy elementwise kernel is better.
    # Or we can use simple SMA for first, then EMA.
    
    # Let's use a simple EMA implementation for now as it's vectorized-friendly
    # if we use cp.exp and convolution? No, EMA is recursive.
    # We'll use a loop for now, it's still faster than CPU for large arrays.
    
    n = len(close)
    avg_gain = cp.zeros(n, dtype=cp.float32)
    avg_loss = cp.zeros(n, dtype=cp.float32)
    
    # Initial SMA
    avg_gain[period] = cp.mean(gain[1:period+1])
    avg_loss[period] = cp.mean(loss[1:period+1])
    
    # Loop for Wilder's smoothing
    # This part is slow in Python loop even with CuPy.
    # Ideally should be a custom CUDA kernel.
    # For now, let's use a slightly different approximation: standard EMA
    # which can be computed faster or just accept the loop overhead.
    
    # Actually, let's write a raw kernel for EMA, it's a good practice here.
    
    ema_kernel = cp.ElementwiseKernel(
        'raw T x, float32 alpha, int32 n',
        'raw T y',
        '''
        // This is not easy in ElementwiseKernel as it needs state.
        // We need a scan kernel or just loop in Python.
        ''',
        'ema_kernel'
    )
    
    # Fallback to Python loop over CuPy arrays (slow but data stays on GPU)
    # Or transfer to CPU, compute, transfer back? No, that defeats purpose.
    
    # Let's use the iterative approach, it's acceptable for N < 100k
    g = gain
    l = loss
    
    ag = cp.zeros_like(g)
    al = cp.zeros_like(l)
    
    ag[period] = cp.mean(g[1:period+1])
    al[period] = cp.mean(l[1:period+1])
    
    # Pre-calculate constants
    p_minus_1 = period - 1
    
    # To optimize: we can't easily vectorize recursive filter without scan.
    # But we can use pandas ewm on CPU if we really have to.
    # Wait, if we are on GPU, we want to stay on GPU.
    
    # Let's use a simple convolution for SMA as a proxy if period is small?
    # No, RSI needs Wilder.
    
    # Let's implement the loop.
    for i in range(period + 1, n):
        ag[i] = (ag[i-1] * p_minus_1 + g[i]) / period
        al[i] = (al[i-1] * p_minus_1 + l[i]) / period
        
    rs = ag / al
    rsi = 100 - (100 / (1 + rs))
    
    # Fill NaNs
    rsi[:period] = 50.0 # Neutral
    
    return cp.asnumpy(rsi)


def bollinger_bands_gpu(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands on GPU."""
    if not is_gpu_features_available():
        raise RuntimeError("GPU not available")
        
    close_gpu = cp.asarray(close, dtype=cp.float32)
    
    # SMA
    # Convolution is efficient on GPU
    kernel = cp.ones(period, dtype=cp.float32) / period
    sma = cp.convolve(close_gpu, kernel, mode='valid')
    
    # Pad beginning
    pad = cp.full(period - 1, cp.nan, dtype=cp.float32)
    sma = cp.concatenate([pad, sma])
    
    # Rolling STD
    # std = sqrt(E[x^2] - (E[x])^2)
    # This is susceptible to precision errors but fast.
    
    c2 = close_gpu ** 2
    sma2 = cp.convolve(c2, kernel, mode='valid')
    sma2 = cp.concatenate([pad, sma2])
    
    var = sma2 - sma ** 2
    std = cp.sqrt(cp.maximum(var, 0)) # Clip negative due to precision
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return cp.asnumpy(upper), cp.asnumpy(sma), cp.asnumpy(lower)


def macd_gpu(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD on GPU."""
    if not is_gpu_features_available():
        raise RuntimeError("GPU not available")
        
    # We need EMA. 
    # Since we don't have a fast GPU EMA yet, we can use a recursive implementation
    # or just use the CPU version if it's too slow.
    # However, for this task, let's assume we want GPU.
    
    # Let's define a helper for EMA
    def ema_gpu(data: cp.ndarray, span: int) -> cp.ndarray:
        alpha = 2 / (span + 1)
        # Use simple recursive loop for now
        res = cp.zeros_like(data)
        res[0] = data[0]
        for i in range(1, len(data)):
            res[i] = alpha * data[i] + (1 - alpha) * res[i-1]
        return res

    close_gpu = cp.asarray(close, dtype=cp.float32)
    
    ema_fast = ema_gpu(close_gpu, fast)
    ema_slow = ema_gpu(close_gpu, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema_gpu(macd_line, signal)
    histogram = macd_line - signal_line
    
    return cp.asnumpy(macd_line), cp.asnumpy(signal_line), cp.asnumpy(histogram)


def log_returns_gpu(close: np.ndarray) -> np.ndarray:
    """Log returns on GPU."""
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")
    
    close_gpu = cp.asarray(close, dtype=cp.float32)
    return cp.asnumpy(cp.diff(cp.log(close_gpu), prepend=cp.log(close_gpu[0])))


def volatility_gpu(returns: np.ndarray, window: int = 20) -> np.ndarray:
    """Rolling volatility (std dev) on GPU."""
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")
        
    ret_gpu = cp.asarray(returns, dtype=cp.float32)
    
    # E[x^2] - (E[x])^2
    kernel = cp.ones(window, dtype=cp.float32) / window
    
    mean = cp.convolve(ret_gpu, kernel, mode='valid')
    mean2 = cp.convolve(ret_gpu**2, kernel, mode='valid')
    
    pad = cp.full(window - 1, cp.nan, dtype=cp.float32)
    mean = cp.concatenate([pad, mean])
    mean2 = cp.concatenate([pad, mean2])
    
    var = mean2 - mean**2
    std = cp.sqrt(cp.maximum(var, 0))
    
    return cp.asnumpy(std)
