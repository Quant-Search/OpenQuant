"""Build script for OpenQuant Robot App using PyInstaller.
Usage: python scripts/build_app.py
"""
import PyInstaller.__main__
import os
import sys
import shutil
from pathlib import Path
import streamlit

def build():
    # 1. Determine paths
    repo_root = Path(__file__).parent.parent.resolve()
    launcher_path = repo_root / "openquant" / "gui" / "launcher.py"
    dist_dir = repo_root / "dist"
    build_dir = repo_root / "build"
    
    # Clean previous builds
    if dist_dir.exists(): shutil.rmtree(dist_dir)
    if build_dir.exists(): shutil.rmtree(build_dir)

    # 2. Collect Streamlit static assets
    st_path = Path(streamlit.__file__).parent
    print(f"📦 Streamlit location: {st_path}")
    
    # 3. Define PyInstaller args
    sep = os.pathsep
    
    # Hidden imports that PyInstaller might miss
    hidden_imports = [
        "streamlit",
        "duckdb",
        "pandas",
        "numpy",
        "plotly",
        "altair",
        "pydeck",
        "watchdog",
        "ccxt",
        "openquant",
        "openquant.gui.dashboard",
        "openquant.gui.scheduler",
        # Add MT5 if on Windows, but safe to omit on Linux build if not installed
    ]
    
    args = [
        str(launcher_path),
        "--name=OpenQuantRobot",
        "--clean",
        "--noconfirm",
        # Bundle everything into one dir (easier for debugging than --onefile initially)
        # We can switch to --onefile later if desired, but --onedir is faster to build
        "--onedir", 
        
        # Paths
        f"--distpath={dist_dir}",
        f"--workpath={build_dir}",
        f"--paths={repo_root}",
        
        # Data: Streamlit static files
        f"--add-data={st_path / 'static'}{os.pathsep}streamlit/static",
        f"--add-data={st_path / 'runtime'}{os.pathsep}streamlit/runtime",
        
        # Data: Our source code (needed for dashboard.py to run via streamlit)
        f"--add-data={repo_root / 'openquant'}{os.pathsep}openquant",
        
        # Metadata
        "--copy-metadata=streamlit",
        "--copy-metadata=tqdm",
        "--copy-metadata=requests",
        "--copy-metadata=packaging",
        "--copy-metadata=duckdb",
        "--copy-metadata=numpy",
        "--copy-metadata=pandas",
        "--copy-metadata=plotly",
    ]
    
    for imp in hidden_imports:
        args.append(f"--hidden-import={imp}")

    print("🔨 Starting PyInstaller build...")
    PyInstaller.__main__.run(args)
    print("✅ Build complete!")
    
    # Post-build instructions
    exe_path = dist_dir / "OpenQuantRobot" / ("OpenQuantRobot.exe" if os.name == 'nt' else "OpenQuantRobot")
    print(f"\n🎉 Executable created at: {exe_path}")
    print("To run:")
    print(f"  {exe_path}")

if __name__ == "__main__":
    build()
