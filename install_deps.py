import subprocess
import sys

# Read requirements and skip packages incompatible with Python 3.14
skip_packages = ["pandas_ta", "numba", "dask", "redis"]
with open("requirements.txt") as f:
    reqs = [line.strip() for line in f if line.strip() and not line.startswith("#") and not any(pkg in line for pkg in skip_packages)]

# Install each package
python_exe = r".venv\Scripts\python.exe"
for req in reqs:
    print(f"Installing {req}...")
    result = subprocess.run([python_exe, "-m", "pip", "install", req], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to install {req}")
        print(result.stderr)
    else:
        print(f"Successfully installed {req}")

print("\nInstallation complete!")
