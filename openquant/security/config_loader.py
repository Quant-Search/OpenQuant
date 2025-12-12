"""Enhanced config loader with encrypted credentials support.

This module extends openquant.utils.config with encryption support,
allowing seamless integration of encrypted .env files.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional


def load_env_with_encryption(
    prefer_encrypted: bool = True,
    encrypted_path: Optional[str | Path] = None,
    plaintext_path: Optional[str | Path] = None,
    passphrase: Optional[str] = None,
    fallback_to_dotenv: bool = True
) -> Dict[str, str]:
    """Load environment variables with encryption support.
    
    This function attempts to load encrypted credentials first, then falls back
    to plaintext .env if needed. It's designed to be a drop-in replacement for
    dotenv.load_dotenv() with encryption support.
    
    Args:
        prefer_encrypted: Try encrypted file first (default: True)
        encrypted_path: Path to encrypted .env (default: .env.encrypted)
        plaintext_path: Path to plaintext .env (default: .env)
        passphrase: Decryption passphrase (or use OPENQUANT_MASTER_KEY env var)
        fallback_to_dotenv: Use python-dotenv if no files found (default: True)
        
    Returns:
        Dictionary of loaded environment variables
        
    Example:
        # In your application startup code:
        from openquant.security.config_loader import load_env_with_encryption
        
        load_env_with_encryption()
        # Now all env vars are available via os.environ
    """
    if encrypted_path is None:
        encrypted_path = Path(".env.encrypted")
    else:
        encrypted_path = Path(encrypted_path)
    
    if plaintext_path is None:
        plaintext_path = Path(".env")
    else:
        plaintext_path = Path(plaintext_path)
    
    try:
        from openquant.security.secrets import secure_env_loader
        
        return secure_env_loader(
            prefer_encrypted=prefer_encrypted,
            encrypted_path=encrypted_path,
            plaintext_path=plaintext_path,
            passphrase=passphrase
        )
    
    except FileNotFoundError:
        if fallback_to_dotenv:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                return {}
            except Exception:
                pass
        
        return {}
    
    except ImportError:
        if fallback_to_dotenv:
            from dotenv import load_dotenv
            load_dotenv()
        return {}


def env(key: str, default: Any | None = None) -> Any:
    """Fetch environment variable with a default.
    
    Compatible with openquant.utils.config.env() for drop-in replacement.
    
    Args:
        key: Environment variable name.
        default: Fallback value if not set.
        
    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def load_config_with_encrypted_env(
    prefer_encrypted: bool = True,
    encrypted_path: Optional[str | Path] = None,
    plaintext_path: Optional[str | Path] = None,
    passphrase: Optional[str] = None
) -> None:
    """Initialize application with encrypted environment variables.
    
    This is a convenience function that loads encrypted credentials at
    application startup. Call this early in your application lifecycle.
    
    Args:
        prefer_encrypted: Try encrypted file first (default: True)
        encrypted_path: Path to encrypted .env (default: .env.encrypted)
        plaintext_path: Path to plaintext .env (default: .env)
        passphrase: Decryption passphrase (or use OPENQUANT_MASTER_KEY env var)
        
    Example:
        # In your main.py or __init__.py:
        from openquant.security.config_loader import load_config_with_encrypted_env
        
        load_config_with_encrypted_env()
        
        # Now use credentials normally:
        from openquant.utils.config import env
        mt5_password = env("MT5_PASSWORD")
    """
    load_env_with_encryption(
        prefer_encrypted=prefer_encrypted,
        encrypted_path=encrypted_path,
        plaintext_path=plaintext_path,
        passphrase=passphrase,
        fallback_to_dotenv=True
    )
