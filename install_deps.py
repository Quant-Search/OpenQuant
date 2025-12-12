import subprocess
import sys

# Read requirements and skip pandas_ta (incompatible with Python 3.14)
with open("requirements.txt") as f:
    reqs = [line.strip() for line in f if line.strip() and not line.startswith("#") and "pandas_ta" not in line]

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
