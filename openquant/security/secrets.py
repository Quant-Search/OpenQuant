"""Credentials encryption and decryption using Fernet symmetric encryption.

This module provides functionality to:
1. Encrypt .env files at rest using Fernet symmetric encryption
2. Derive encryption keys from user passphrase or environment variable
3. Decrypt credentials at runtime for use by the application

Usage:
    # Encrypt the .env file
    encrypt_env_file(passphrase="my_secret_passphrase")
    
    # Decrypt and load environment variables at runtime
    load_encrypted_env(passphrase="my_secret_passphrase")
    
    # Use environment variable for key
    os.environ["OPENQUANT_MASTER_KEY"] = "my_secret_passphrase"
    load_encrypted_env()
"""
from __future__ import annotations
import os
import base64
import getpass
from pathlib import Path
from typing import Dict, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a Fernet-compatible encryption key from a passphrase using PBKDF2.
    
    Args:
        passphrase: User-provided passphrase for key derivation
        salt: Salt bytes for key derivation (16 bytes recommended)
        
    Returns:
        32-byte key suitable for Fernet encryption (base64-encoded)
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=480000,
    )
    key = kdf.derive(passphrase.encode('utf-8'))
    return base64.urlsafe_b64encode(key)


def _get_passphrase(passphrase: Optional[str] = None) -> str:
    """Get passphrase from parameter, environment variable, or user prompt.
    
    Args:
        passphrase: Optional explicit passphrase
        
    Returns:
        Passphrase string
        
    Raises:
        ValueError: If no passphrase is available
    """
    if passphrase:
        return passphrase
    
    env_key = os.getenv("OPENQUANT_MASTER_KEY")
    if env_key:
        return env_key
    
    if os.isatty(0):
        return getpass.getpass("Enter passphrase for .env encryption: ")
    
    raise ValueError(
        "No passphrase provided. Set OPENQUANT_MASTER_KEY environment variable "
        "or pass passphrase parameter."
    )


def encrypt_env_file(
    env_path: str | Path = ".env",
    output_path: Optional[str | Path] = None,
    passphrase: Optional[str] = None,
    salt: Optional[bytes] = None
) -> Path:
    """Encrypt a .env file using Fernet symmetric encryption.
    
    Args:
        env_path: Path to the plaintext .env file
        output_path: Path for encrypted output (default: .env.encrypted)
        passphrase: Encryption passphrase (or use OPENQUANT_MASTER_KEY env var)
        salt: Salt for key derivation (generated if not provided)
        
    Returns:
        Path to the encrypted file
        
    Raises:
        FileNotFoundError: If env_path doesn't exist
        ValueError: If no passphrase is available
    """
    env_path = Path(env_path)
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")
    
    if output_path is None:
        output_path = env_path.parent / f"{env_path.name}.encrypted"
    else:
        output_path = Path(output_path)
    
    passphrase_str = _get_passphrase(passphrase)
    
    if salt is None:
        salt = os.urandom(16)
    
    key = _derive_key(passphrase_str, salt)
    fernet = Fernet(key)
    
    with env_path.open('rb') as f:
        plaintext = f.read()
    
    encrypted = fernet.encrypt(plaintext)
    
    with output_path.open('wb') as f:
        f.write(salt)
        f.write(encrypted)
    
    print(f"Encrypted {env_path} -> {output_path}")
    print(f"Salt (first 16 bytes) stored in encrypted file")
    return output_path


def decrypt_env_file(
    encrypted_path: str | Path = ".env.encrypted",
    output_path: Optional[str | Path] = None,
    passphrase: Optional[str] = None
) -> Path:
    """Decrypt an encrypted .env file.
    
    Args:
        encrypted_path: Path to the encrypted .env file
        output_path: Path for decrypted output (default: .env.decrypted)
        passphrase: Decryption passphrase (or use OPENQUANT_MASTER_KEY env var)
        
    Returns:
        Path to the decrypted file
        
    Raises:
        FileNotFoundError: If encrypted_path doesn't exist
        ValueError: If no passphrase is available or decryption fails
    """
    encrypted_path = Path(encrypted_path)
    if not encrypted_path.exists():
        raise FileNotFoundError(f"Encrypted file not found: {encrypted_path}")
    
    if output_path is None:
        output_path = encrypted_path.parent / ".env.decrypted"
    else:
        output_path = Path(output_path)
    
    passphrase_str = _get_passphrase(passphrase)
    
    with encrypted_path.open('rb') as f:
        salt = f.read(16)
        encrypted = f.read()
    
    key = _derive_key(passphrase_str, salt)
    fernet = Fernet(key)
    
    try:
        plaintext = fernet.decrypt(encrypted)
    except Exception as e:
        raise ValueError(f"Decryption failed. Invalid passphrase or corrupted file: {e}")
    
    with output_path.open('wb') as f:
        f.write(plaintext)
    
    print(f"Decrypted {encrypted_path} -> {output_path}")
    return output_path


def load_encrypted_env(
    encrypted_path: str | Path = ".env.encrypted",
    passphrase: Optional[str] = None
) -> Dict[str, str]:
    """Decrypt and load environment variables from an encrypted .env file.
    
    This function decrypts the .env file in memory and parses it into a dictionary
    without writing the plaintext to disk. Variables are also set in os.environ.
    
    Args:
        encrypted_path: Path to the encrypted .env file
        passphrase: Decryption passphrase (or use OPENQUANT_MASTER_KEY env var)
        
    Returns:
        Dictionary of environment variables
        
    Raises:
        FileNotFoundError: If encrypted_path doesn't exist
        ValueError: If no passphrase is available or decryption fails
    """
    encrypted_path = Path(encrypted_path)
    if not encrypted_path.exists():
        raise FileNotFoundError(f"Encrypted file not found: {encrypted_path}")
    
    passphrase_str = _get_passphrase(passphrase)
    
    with encrypted_path.open('rb') as f:
        salt = f.read(16)
        encrypted = f.read()
    
    key = _derive_key(passphrase_str, salt)
    fernet = Fernet(key)
    
    try:
        plaintext = fernet.decrypt(encrypted)
    except Exception as e:
        raise ValueError(f"Decryption failed. Invalid passphrase or corrupted file: {e}")
    
    env_vars = {}
    for line in plaintext.decode('utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        if '=' in line:
            key_part, _, value_part = line.partition('=')
            key_part = key_part.strip()
            value_part = value_part.strip().strip('"').strip("'")
            env_vars[key_part] = value_part
            os.environ[key_part] = value_part
    
    return env_vars


def secure_env_loader(
    prefer_encrypted: bool = True,
    encrypted_path: str | Path = ".env.encrypted",
    plaintext_path: str | Path = ".env",
    passphrase: Optional[str] = None
) -> Dict[str, str]:
    """Load environment variables with preference for encrypted version.
    
    This is a convenience function that attempts to load from encrypted .env first,
    falling back to plaintext if encrypted version is not available.
    
    Args:
        prefer_encrypted: If True, try encrypted file first
        encrypted_path: Path to encrypted .env file
        plaintext_path: Path to plaintext .env file
        passphrase: Decryption passphrase (or use OPENQUANT_MASTER_KEY env var)
        
    Returns:
        Dictionary of environment variables
        
    Raises:
        FileNotFoundError: If neither file exists
    """
    encrypted_path = Path(encrypted_path)
    plaintext_path = Path(plaintext_path)
    
    if prefer_encrypted and encrypted_path.exists():
        try:
            return load_encrypted_env(encrypted_path, passphrase)
        except ValueError as e:
            print(f"Warning: Failed to decrypt {encrypted_path}: {e}")
            print(f"Falling back to plaintext {plaintext_path}")
    
    if plaintext_path.exists():
        from dotenv import load_dotenv
        load_dotenv(plaintext_path)
        
        env_vars = {}
        with plaintext_path.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    key_part, _, value_part = line.partition('=')
                    key_part = key_part.strip()
                    value_part = value_part.strip().strip('"').strip("'")
                    env_vars[key_part] = value_part
        
        return env_vars
    
    if not prefer_encrypted and encrypted_path.exists():
        return load_encrypted_env(encrypted_path, passphrase)
    
    raise FileNotFoundError(
        f"No environment file found. Checked: {encrypted_path}, {plaintext_path}"
    )
