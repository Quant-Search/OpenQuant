"""
Kill Switch Module.

Provides a mechanism to immediately stop trading and close positions
based on external triggers (e.g., presence of a file).
"""
import os
from pathlib import Path
from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)

class KillSwitch:
    """
    Monitors for a kill signal (file existence) and triggers emergency stop.
    """
    def __init__(self, trigger_file: str = "data/STOP"):
        self.trigger_file = Path(trigger_file)
        
    def is_active(self) -> bool:
        """Check if the kill switch is activated."""
        if self.trigger_file.exists():
            LOGGER.critical(f"KILL SWITCH ACTIVATED: Found {self.trigger_file}")
            return True
        return False
        
    def activate(self):
        """Manually activate the kill switch."""
        self.trigger_file.touch()
        LOGGER.critical("Kill Switch manually activated.")
        
    def deactivate(self):
        """Deactivate the kill switch."""
        if self.trigger_file.exists():
            os.remove(self.trigger_file)
            LOGGER.info("Kill Switch deactivated.")

KILL_SWITCH = KillSwitch()
