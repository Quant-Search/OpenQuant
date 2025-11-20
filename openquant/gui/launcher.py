"""Launcher for the OpenQuant Robot App (Streamlit).
This script is the entry point for the PyInstaller executable.
"""
import sys
import os
from pathlib import Path
from datetime import datetime

def main():
    # 1. Set up environment
    # When running as frozen (exe), sys._MEIPASS is the temp folder where assets are unpacked
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)  # type: ignore
        # Ensure we can find our own packages
        sys.path.insert(0, str(base_path))
        # Set env var for dashboard to find DB if not set
        if "OPENQUANT_RESULTS_DB" not in os.environ:
            # Store data in user's home dir to avoid read-only errors in AppImage
            user_data_dir = Path.home() / "OpenQuant"
            data_dir = user_data_dir / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup launcher log
            log_file = user_data_dir / "launcher.log"
            with open(log_file, "w") as f:
                f.write(f"Launcher started at {datetime.now()}\n")
            
            os.environ["OPENQUANT_RESULTS_DB"] = str(data_dir / "results.duckdb")
            
            # Change CWD to user_data_dir so relative paths (like 'reports') go there
            os.chdir(user_data_dir)
            print(f"📂 Working Directory set to: {user_data_dir}")
        log_file = user_data_dir / "openquant_launcher.log"
    else:
        base_path = Path(__file__).parent.parent.parent
        log_file = Path("openquant_launcher.log") # For dev, log in current dir

    # 2. Locate dashboard script
    # 3. Launch Streamlit
    # We need to point to the absolute path of the dashboard script
    dashboard_script = base_path / "openquant" / "gui" / "dashboard.py"
    
    if not dashboard_script.exists():
        msg = f"Error: Dashboard script not found at {dashboard_script}"
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        sys.exit(1)

    print(f"🚀 Launching OpenQuant Robot from: {dashboard_script}")
    print(f"📂 Working Directory: {os.getcwd()}")
    
    with open(log_file, "a") as f:
        f.write(f"Launching Streamlit: {dashboard_script}\n")
        f.write(f"CWD: {os.getcwd()}\n")

    # 3. Launch Streamlit
    # We use the cli module to mimic 'streamlit run'
    from streamlit.web import cli as st_cli
    
    # Force headless=false to ensure browser opens
    # We set config via environment variables to ensure they are picked up
    # AND we write a config.toml to be absolutely sure.
    
    config_path = user_data_dir / "config.toml"
    with open(config_path, "w") as f:
        f.write('[browser]\n')
        f.write('gatherUsageStats = false\n')
        f.write('[server]\n')
        f.write('headless = false\n')
        f.write('port = 8501\n')
        f.write('address = "localhost"\n')
        
    os.environ["STREAMLIT_CONFIG_FILE"] = str(config_path)
    
    sys.argv = [
        "streamlit",
        "run",
        str(dashboard_script),
    ]
    
    sys.exit(st_cli.main())

if __name__ == "__main__":
    main()
