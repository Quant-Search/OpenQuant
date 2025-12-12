import os
import sys
import subprocess
from pathlib import Path

def warm_startup_cache():
    """Warm cache with frequently used data on dashboard startup."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from openquant.data import get_cache, DataLoader
        from openquant.utils.logging import get_logger
        
        logger = get_logger(__name__)
        
        cache = get_cache()
        loader = DataLoader(use_cache=False)
        
        # Common symbols and timeframes for the dashboard
        symbols = ["BTC/USDT", "ETH/USDT", "AAPL", "SPY"]
        timeframes = ["1d", "4h", "1h"]
        
        logger.info("Warming cache for dashboard startup...")
        cache.warm_cache(
            symbols=symbols,
            timeframes=timeframes,
            data_loader=loader.get_ohlcv,
            lookback_days=30
        )
        logger.info("Cache warming complete")
    except Exception as e:
        print(f"Cache warming skipped: {e}")

def main():
    """
    Launcher for OpenQuant Dashboard with cache warming.
    Handles path resolution and runs streamlit.
    """
    project_root = Path(__file__).parent.parent.resolve()
    dashboard_path = project_root / "openquant" / "gui" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)
    
    # Warm cache before starting dashboard
    warm_startup_cache()
    
    print(f"🚀 Launching OpenQuant Dashboard...")
    print(f"📂 Project Root: {project_root}")
    
    # Construct command: streamlit run <path> --server.port 8501 --server.headless true
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(dashboard_path),
        "--server.port", "8501",
        "--server.headless", "true",
        "--theme.base", "dark"
    ]
    
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped.")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main()
