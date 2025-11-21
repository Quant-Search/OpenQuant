import os
import sys
import subprocess
from pathlib import Path

def main():
    """
    Launcher for OpenQuant Dashboard.
    Handles path resolution and runs streamlit.
    """
    project_root = Path(__file__).parent.parent.resolve()
    dashboard_path = project_root / "openquant" / "gui" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard not found at {dashboard_path}")
        sys.exit(1)
        
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
