"""Benchmark GPU vs CPU performance for strategy computations.

This script compares the performance of GPU-accelerated vs CPU-only
implementations for various dataset sizes.
"""
import sys
from pathlib import Path
import time
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from openquant.gpu.cuda_kernels import is_gpu_available
from openquant.gpu.dispatcher import set_gpu_policy, kalman_filter
import pandas as pd


def benchmark_kalman_filter():
    """Benchmark Kalman filter GPU vs CPU."""
    print("=" * 70)
    print("GPU vs CPU Benchmark - Kalman Filter")
    print("=" * 70)
    
    # Check GPU availability
    if not is_gpu_available():
        print("\nWARNING: GPU not available!")
        print("Make sure you have:")
        print("  1. CUDA-capable GPU")
        print("  2. CUDA toolkit installed")
        print("  3. CuPy installed (pip install cupy-cuda11x)")
        return
    
    print(f"\nGPU Available: Yes")
    
    # Test different dataset sizes
    sizes = [100, 500, 1000, 5000, 10000]
    
    print("\n" + "-" * 70)
    print(f"{'Dataset Size':<15} {'CPU Time (ms)':<20} {'GPU Time (ms)':<20} {'Speedup':<15}")
    print("-" * 70)
    
    for size in sizes:
        # Generate synthetic price data
        np.random.seed(42)
        prices = np.cumsum(np.random.randn(size)) + 100
        
        # CPU benchmark
        set_gpu_policy(enabled=False)
        
        start = time.perf_counter()
        cpu_estimated, cpu_deviations = kalman_filter(prices, 1e-5, 1e-3)
        cpu_time = (time.perf_counter() - start) * 1000  # ms
        
        # GPU benchmark
        set_gpu_policy(enabled=True, min_dataset_size=0)  # Force GPU even for small data
        
        start = time.perf_counter()
        gpu_estimated, gpu_deviations = kalman_filter(prices, 1e-5, 1e-3)
        gpu_time = (time.perf_counter() - start) * 1000  # ms
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        # Verify results match (within tolerance)
        max_diff = np.max(np.abs(cpu_estimated - gpu_estimated))
        if max_diff > 1e-4:
            print(f"  WARNING: Results differ by {max_diff:.6f}")
        
        print(f"{size:<15} {cpu_time:<20.2f} {gpu_time:<20.2f} {speedup:<15.2f}x")
    
    print("-" * 70)
    print("\nConclusion:")
    print("  - GPU overhead is significant for small datasets (< 1000 bars)")
    print("  - GPU acceleration becomes beneficial for larger datasets (> 1000 bars)")
    print("  - Expected speedup: 2-5x for datasets with 5000+ bars")


def benchmark_strategy_signals():
    """Benchmark full strategy signal generation."""
    print("\n\n" + "=" * 70)
    print("GPU vs CPU Benchmark - Strategy Signal Generation")
    print("=" * 70)
    
    if not is_gpu_available():
        print("\nGPU not available, skipping...")
        return
    
    from openquant.strategies.quant.kalman import KalmanMeanReversionStrategy
    
    # Create synthetic OHLCV data
    size = 5000
    np.random.seed(42)
    
    df = pd.DataFrame({
        'Close': np.cumsum(np.random.randn(size)) + 100,
        'Open': np.cumsum(np.random.randn(size)) + 100,
        'High': np.cumsum(np.random.randn(size)) + 102,
        'Low': np.cumsum(np.random.randn(size)) + 98,
        'Volume': np.random.randint(1000, 10000, size)
    })
    
    print(f"\nDataset: {size} bars")
    
    # CPU version
    print("\nRunning CPU version...")
    strategy_cpu = KalmanMeanReversionStrategy(use_gpu=False)
    
    start = time.perf_counter()
    signals_cpu = strategy_cpu.generate_signals(df)
    cpu_time = (time.perf_counter() - start) * 1000
    
    print(f"  CPU Time: {cpu_time:.2f} ms")
    print(f"  Signals generated: {len(signals_cpu)}")
    print(f"  Long signals: {(signals_cpu == 1).sum()}")
    print(f"  Short signals: {(signals_cpu == -1).sum()}")
    
    # GPU version
    print("\nRunning GPU version...")
    strategy_gpu = KalmanMeanReversionStrategy(use_gpu=True)
    
    start = time.perf_counter()
    signals_gpu = strategy_gpu.generate_signals(df)
    gpu_time = (time.perf_counter() - start) * 1000
    
    print(f"  GPU Time: {gpu_time:.2f} ms")
    print(f"  Signals generated: {len(signals_gpu)}")
    print(f"  Long signals: {(signals_gpu == 1).sum()}")
    print(f"  Short signals: {(signals_gpu == -1).sum()}")
    
    # Comparison
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"\n  Speedup: {speedup:.2f}x")
    
    # Verify signals match
    mismatch = (signals_cpu != signals_gpu).sum()
    if mismatch > 0:
        print(f"\n  WARNING: {mismatch} signals differ between CPU and GPU!")
    else:
        print(f"\n  Results match: All {len(signals_cpu)} signals identical")


def test_gpu_memory():
    """Test GPU memory usage and limits."""
    print("\n\n" + "=" * 70)
    print("GPU Memory Test")
    print("=" * 70)
    
    if not is_gpu_available():
        print("\nGPU not available, skipping...")
        return
    
    try:
        import cupy as cp
        
        # Get GPU info
        device = cp.cuda.Device(0)
        props = device.attributes
        
        print(f"\nGPU: Device 0")
        print(f"  Name: {device.name.decode() if isinstance(device.name, bytes) else device.name}")
        print(f"  Compute Capability: {device.compute_capability}")
        print(f"  Total Memory: {device.mem_info[1] / 1024**3:.2f} GB")
        print(f"  Free Memory: {device.mem_info[0] / 1024**3:.2f} GB")
        
        # Test maximum dataset size
        print("\nTesting maximum dataset size...")
        
        sizes_to_test = [10000, 50000, 100000, 500000, 1000000]
        max_size = 0
        
        for size in sizes_to_test:
            try:
                prices = cp.random.randn(size, dtype=cp.float32)
                cp.cuda.Stream.null.synchronize()  # Ensure allocation completed
                max_size = size
                print(f"  {size:>10} elements: OK")
                del prices  # Free memory
            except Exception as e:
                print(f"  {size:>10} elements: FAILED ({e})")
                break
        
        print(f"\nMaximum dataset size: ~{max_size:,} elements")
        
    except Exception as e:
        print(f"\nError testing GPU memory: {e}")


if __name__ == "__main__":
    benchmark_kalman_filter()
    benchmark_strategy_signals()
    test_gpu_memory()
    
    print("\n" + "=" * 70)
    print("Benchmark Complete")
    print("=" * 70)
