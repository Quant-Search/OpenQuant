"""
Configuration Management

Single Responsibility: Only handles configuration loading and validation.
Includes MT5 auto-detection and secure credential storage.
"""
import os
import json
from pathlib import Path
from typing import Optional, List
import base64


# ---------------------------------------------------------------------------
# Helpers: Safe type conversion and encryption
# ---------------------------------------------------------------------------

def _safe_int(value: Optional[str]) -> Optional[int]:
    """Convert string to int safely, return None on failure."""
    # value: The string to convert (can be None)
    # Returns: Integer if conversion succeeds, None otherwise
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _simple_encrypt(text: str, key: str = "openquant_key") -> str:
    """
    Simple XOR obfuscation for credentials storage.
    NOT cryptographically secure - just prevents plain text storage.
    For production, use proper encryption (keyring, vault, etc.)

    Args:
        text: Plain text to encrypt
        key: Encryption key (default for simplicity)

    Returns:
        Base64-encoded encrypted string
    """
    # XOR each character with key character (cycling through key)
    encrypted = []
    for i, char in enumerate(text):
        # XOR the character code with key character code
        key_char = key[i % len(key)]
        encrypted_char = chr(ord(char) ^ ord(key_char))
        encrypted.append(encrypted_char)
    # Encode to base64 for safe storage
    return base64.b64encode(''.join(encrypted).encode('latin-1')).decode('ascii')


def _simple_decrypt(encrypted: str, key: str = "openquant_key") -> str:
    """
    Decrypt XOR-obfuscated string.

    Args:
        encrypted: Base64-encoded encrypted string
        key: Decryption key (same as encryption key)

    Returns:
        Original plain text
    """
    try:
        # Decode from base64
        decoded = base64.b64decode(encrypted.encode('ascii')).decode('latin-1')
        # XOR again to decrypt (XOR is symmetric)
        decrypted = []
        for i, char in enumerate(decoded):
            key_char = key[i % len(key)]
            decrypted_char = chr(ord(char) ^ ord(key_char))
            decrypted.append(decrypted_char)
        return ''.join(decrypted)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# MT5 Path Auto-Detection
# ---------------------------------------------------------------------------

def _find_mt5_terminal() -> Optional[str]:
    """
    Auto-detect MetaTrader 5 terminal path on Windows.

    Search strategy:
    1. Check common installation directories
    2. Search Program Files and AppData
    3. Look for any terminal64.exe

    Returns:
        Full path to terminal64.exe if found, None otherwise
    """
    # Common MT5 installation paths (ordered by likelihood)
    common_paths = [
        # Default Program Files locations
        r"C:\Program Files\MetaTrader 5\terminal64.exe",
        r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        # Broker-specific common locations
        r"C:\Program Files\ICMarkets - MetaTrader 5\terminal64.exe",
        r"C:\Program Files\XM MT5\terminal64.exe",
        r"C:\Program Files\Exness MT5\terminal64.exe",
        r"C:\Program Files\FXCM MetaTrader 5\terminal64.exe",
        r"C:\Program Files\Admiral Markets MT5\terminal64.exe",
        r"C:\Program Files\Pepperstone MetaTrader 5\terminal64.exe",
    ]

    # Check common paths first (fastest)
    for path in common_paths:
        if Path(path).exists():
            return path

    # Search Program Files directories
    search_dirs = []

    # Add Program Files directories
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

    if Path(pf).exists():
        search_dirs.append(Path(pf))
    if Path(pf86).exists():
        search_dirs.append(Path(pf86))

    # Also check AppData for portable installations
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        appdata_local = Path(appdata).parent / "Local"
        if appdata_local.exists():
            search_dirs.append(appdata_local)

    # Search for terminal64.exe in subdirectories
    for search_dir in search_dirs:
        try:
            # Look for folders containing "MetaTrader" or "MT5"
            for folder in search_dir.iterdir():
                if folder.is_dir():
                    folder_name = folder.name.lower()
                    if "metatrader" in folder_name or "mt5" in folder_name:
                        terminal_path = folder / "terminal64.exe"
                        if terminal_path.exists():
                            return str(terminal_path)
        except PermissionError:
            continue  # Skip folders we can't access

    return None


# ---------------------------------------------------------------------------
# Credentials Storage
# ---------------------------------------------------------------------------

def _get_credentials_file() -> Path:
    """
    Get the path to the credentials file.
    Stored in user's home directory for persistence.

    Returns:
        Path to credentials JSON file
    """
    # Use user home directory for persistence across sessions
    home = Path.home()
    creds_dir = home / ".openquant"
    creds_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist
    return creds_dir / "mt5_credentials.json"


def load_saved_credentials() -> dict:
    """
    Load saved MT5 credentials from secure storage.

    Returns:
        Dictionary with login, server, terminal_path, and encrypted password
    """
    creds_file = _get_credentials_file()
    if not creds_file.exists():
        return {}

    try:
        with open(creds_file, 'r') as f:
            data = json.load(f)

        # Decrypt password if present
        if data.get("password_encrypted"):
            data["password"] = _simple_decrypt(data["password_encrypted"])
            del data["password_encrypted"]

        return data
    except Exception:
        return {}


def save_credentials(
    login: int,
    password: str,
    server: str,
    terminal_path: str
) -> bool:
    """
    Save MT5 credentials to secure storage.
    Password is encrypted (obfuscated) before storage.

    Args:
        login: MT5 account number
        password: MT5 password (will be encrypted)
        server: MT5 server name
        terminal_path: Path to terminal64.exe

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        creds_file = _get_credentials_file()

        # Prepare data with encrypted password
        from datetime import datetime
        data = {
            "login": login,
            "password_encrypted": _simple_encrypt(password),
            "server": server,
            "terminal_path": terminal_path,
            "saved_at": datetime.now().isoformat()  # Timestamp
        }

        with open(creds_file, 'w') as f:
            json.dump(data, f, indent=2)

        # Set restrictive permissions on Unix-like systems
        try:
            os.chmod(creds_file, 0o600)  # Owner read/write only
        except (AttributeError, OSError):
            pass  # Windows doesn't support chmod the same way

        return True
    except Exception as e:
        print(f"[ERROR] Failed to save credentials: {e}")
        return False


def delete_saved_credentials() -> bool:
    """
    Delete saved MT5 credentials.

    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        creds_file = _get_credentials_file()
        if creds_file.exists():
            creds_file.unlink()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main Configuration Class
# ---------------------------------------------------------------------------

class Config:
    """
    Robot configuration - all settings in one place.

    Configuration priority (highest to lowest):
    1. Environment variables (for CI/CD and security)
    2. Saved credentials file (for convenience)
    3. Default values
    """

    # -----------------------------------------------------------------------
    # Trading Configuration
    # -----------------------------------------------------------------------

    # Trading symbols (MT5 format)
    SYMBOLS: List[str] = ["EURUSD", "GBPUSD", "USDJPY"]

    # Timeframe for analysis (1h, 4h, 1d)
    TIMEFRAME: str = "1h"

    # -----------------------------------------------------------------------
    # Strategy Parameters (Kalman Filter Mean Reversion)
    # -----------------------------------------------------------------------

    PROCESS_NOISE: float = 1e-5       # Q: How much true price varies
    MEASUREMENT_NOISE: float = 1e-3   # R: How noisy are observations
    SIGNAL_THRESHOLD: float = 1.5     # Z-score threshold for signals

    # -----------------------------------------------------------------------
    # Risk Management
    # -----------------------------------------------------------------------

    RISK_PER_TRADE: float = 0.02      # Risk 2% of equity per trade
    MAX_POSITIONS: int = 3            # Maximum concurrent positions
    STOP_LOSS_ATR_MULT: float = 2.0   # Stop loss = 2x ATR
    TAKE_PROFIT_ATR_MULT: float = 3.0 # Take profit = 3x ATR

    # -----------------------------------------------------------------------
    # Loop Settings
    # -----------------------------------------------------------------------

    LOOP_INTERVAL_SECONDS: int = 3600  # Run every hour

    # -----------------------------------------------------------------------
    # MT5 Credentials - Load from environment, then saved file, then auto-detect
    # -----------------------------------------------------------------------

    # Try environment variables first
    _env_login = _safe_int(os.getenv("MT5_LOGIN"))
    _env_password = os.getenv("MT5_PASSWORD")
    _env_server = os.getenv("MT5_SERVER")
    _env_terminal = os.getenv("MT5_TERMINAL_PATH")

    # Load saved credentials as fallback
    _saved = load_saved_credentials()

    # Auto-detect terminal path if not provided
    _detected_terminal = _find_mt5_terminal()

    # Final values (priority: env > saved > detected)
    MT5_LOGIN: Optional[int] = _env_login or _saved.get("login")
    MT5_PASSWORD: Optional[str] = _env_password or _saved.get("password")
    MT5_SERVER: Optional[str] = _env_server or _saved.get("server")
    MT5_TERMINAL_PATH: Optional[str] = (
        _env_terminal or
        _saved.get("terminal_path") or
        _detected_terminal
    )

    @classmethod
    def is_mt5_configured(cls) -> bool:
        """
        Check if all required MT5 credentials are configured.

        Returns:
            True if login, password, server, and terminal path are all set
        """
        return all([
            cls.MT5_LOGIN,
            cls.MT5_PASSWORD,
            cls.MT5_SERVER,
            cls.MT5_TERMINAL_PATH
        ])

    @classmethod
    def get_mt5_status(cls) -> dict:
        """
        Get MT5 configuration status for display.

        Returns:
            Dictionary with status of each credential
        """
        return {
            "login": "Set" if cls.MT5_LOGIN else "Missing",
            "password": "Set" if cls.MT5_PASSWORD else "Missing",
            "server": "Set" if cls.MT5_SERVER else "Missing",
            "terminal_path": cls.MT5_TERMINAL_PATH or "Not found",
            "fully_configured": cls.is_mt5_configured()
        }

    @classmethod
    def reload_credentials(cls):
        """
        Reload credentials from saved file.
        Useful after saving new credentials.
        """
        saved = load_saved_credentials()

        # Only update if not overridden by environment variables
        if not os.getenv("MT5_LOGIN"):
            cls.MT5_LOGIN = saved.get("login")
        if not os.getenv("MT5_PASSWORD"):
            cls.MT5_PASSWORD = saved.get("password")
        if not os.getenv("MT5_SERVER"):
            cls.MT5_SERVER = saved.get("server")
        if not os.getenv("MT5_TERMINAL_PATH"):
            cls.MT5_TERMINAL_PATH = saved.get("terminal_path") or _find_mt5_terminal()


