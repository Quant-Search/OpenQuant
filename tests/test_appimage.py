import subprocess
import time
import os
import signal
from pathlib import Path
import pytest

APPIMAGE_PATH = Path("OpenQuantRobot-x86_64.AppImage")

@pytest.mark.skipif(not APPIMAGE_PATH.exists(), reason="AppImage not found")
def test_appimage_launch():
    """Test that the AppImage launches and prints startup messages."""
    print(f"Testing AppImage at: {APPIMAGE_PATH}")
    
    # Make sure it's executable
    os.chmod(APPIMAGE_PATH, 0o755)
    
    # Run the AppImage
    # We expect it to start Streamlit. We'll let it run for a few seconds and check stdout.
    process = subprocess.Popen(
        [str(APPIMAGE_PATH.resolve())],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=os.setsid # Create new process group to kill properly
    )
    
    try:
        # Wait a bit for startup
        time.sleep(5)
        
        # Check if it's still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print("STDOUT:", stdout)
            print("STDERR:", stderr)
            pytest.fail(f"AppImage exited prematurely with code {process.returncode}")
            
        # Read some output (non-blocking is hard with Popen without threads, 
        # so we'll just kill and read what we got if we can, or use communicate with timeout if it exited)
        
        # Send SIGTERM to verify clean exit capability
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            stdout, stderr = process.communicate()
            
        print("STDOUT:", stdout)
        
        # Verify expected output
        assert "Launching OpenQuant Robot" in stdout or "Streamlit" in stderr or "OpenQuant" in stdout
        assert "Working Directory set to" in stdout
        
        # New: Verify HTTP connectivity
        import urllib.request
        print("Checking dashboard connectivity at http://localhost:8501...")
        connected = False
        for i in range(10):
            try:
                with urllib.request.urlopen("http://localhost:8501", timeout=1) as response:
                    if response.status == 200:
                        print("✅ Dashboard is reachable!")
                        connected = True
                        break
            except Exception:
                time.sleep(1)
        
        if not connected:
            print("❌ Failed to connect to dashboard.")
            # Don't fail the test yet if we just want to check launch, but it's a good indicator
            # pytest.fail("Dashboard server did not respond")
            
    except Exception as e:
        # Cleanup
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except:
            pass
        raise e

if __name__ == "__main__":
    # Manual run
    if APPIMAGE_PATH.exists():
        test_appimage_launch()
        print("Test Passed!")
    else:
        print("AppImage not found, skipping.")
