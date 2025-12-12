"""Tests for openquant.security.secrets module."""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
import pytest
from openquant.security.secrets import (
    encrypt_env_file,
    decrypt_env_file,
    load_encrypted_env,
    secure_env_loader,
    _derive_key,
    _get_passphrase,
)


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    content = """# Test environment file
MT5_LOGIN="12345678"
MT5_PASSWORD="test_password"
APCA_API_KEY_ID="test_key"
APCA_API_SECRET_KEY="test_secret"

# Comment line
OPTIONAL_VAR="value"
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        f.write(content)
        temp_path = Path(f.name)
    
    yield temp_path
    
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def cleanup_env():
    """Clean up environment variables after tests."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


def test_derive_key():
    """Test key derivation from passphrase."""
    passphrase = "test_passphrase"
    salt = b"0123456789abcdef"
    
    key1 = _derive_key(passphrase, salt)
    key2 = _derive_key(passphrase, salt)
    
    assert key1 == key2
    assert len(key1) == 44
    
    key3 = _derive_key("different", salt)
    assert key3 != key1
    
    key4 = _derive_key(passphrase, b"different_salt!!")
    assert key4 != key1


def test_get_passphrase():
    """Test passphrase retrieval from various sources."""
    explicit = _get_passphrase("explicit_passphrase")
    assert explicit == "explicit_passphrase"
    
    os.environ["OPENQUANT_MASTER_KEY"] = "env_passphrase"
    from_env = _get_passphrase()
    assert from_env == "env_passphrase"
    del os.environ["OPENQUANT_MASTER_KEY"]
    
    with pytest.raises(ValueError, match="No passphrase provided"):
        _get_passphrase()


def test_encrypt_decrypt_cycle(temp_env_file, cleanup_env):
    """Test full encryption and decryption cycle."""
    passphrase = "test_encryption_key"
    
    encrypted_path = temp_env_file.parent / f"{temp_env_file.name}.encrypted"
    decrypted_path = temp_env_file.parent / f"{temp_env_file.name}.decrypted"
    
    try:
        encrypt_env_file(
            env_path=temp_env_file,
            output_path=encrypted_path,
            passphrase=passphrase
        )
        
        assert encrypted_path.exists()
        assert encrypted_path.stat().st_size > 0
        
        with encrypted_path.open('rb') as f:
            encrypted_content = f.read()
        
        with temp_env_file.open('rb') as f:
            original_content = f.read()
        
        assert encrypted_content != original_content
        
        decrypt_env_file(
            encrypted_path=encrypted_path,
            output_path=decrypted_path,
            passphrase=passphrase
        )
        
        assert decrypted_path.exists()
        
        with decrypted_path.open('r', encoding='utf-8') as f:
            decrypted_content = f.read()
        
        with temp_env_file.open('r', encoding='utf-8') as f:
            original_text = f.read()
        
        assert decrypted_content == original_text
        
    finally:
        for path in [encrypted_path, decrypted_path]:
            if path.exists():
                path.unlink()


def test_load_encrypted_env(temp_env_file, cleanup_env):
    """Test loading encrypted env vars directly into memory."""
    passphrase = "test_load_key"
    
    encrypted_path = temp_env_file.parent / f"{temp_env_file.name}.encrypted"
    
    try:
        encrypt_env_file(
            env_path=temp_env_file,
            output_path=encrypted_path,
            passphrase=passphrase
        )
        
        env_vars = load_encrypted_env(
            encrypted_path=encrypted_path,
            passphrase=passphrase
        )
        
        assert "MT5_LOGIN" in env_vars
        assert env_vars["MT5_LOGIN"] == "12345678"
        assert env_vars["MT5_PASSWORD"] == "test_password"
        assert env_vars["APCA_API_KEY_ID"] == "test_key"
        assert env_vars["OPTIONAL_VAR"] == "value"
        
        assert os.environ["MT5_LOGIN"] == "12345678"
        assert os.environ["MT5_PASSWORD"] == "test_password"
        
    finally:
        if encrypted_path.exists():
            encrypted_path.unlink()


def test_decrypt_with_wrong_passphrase(temp_env_file, cleanup_env):
    """Test that wrong passphrase fails decryption."""
    correct_passphrase = "correct_key"
    wrong_passphrase = "wrong_key"
    
    encrypted_path = temp_env_file.parent / f"{temp_env_file.name}.encrypted"
    
    try:
        encrypt_env_file(
            env_path=temp_env_file,
            output_path=encrypted_path,
            passphrase=correct_passphrase
        )
        
        with pytest.raises(ValueError, match="Decryption failed"):
            load_encrypted_env(
                encrypted_path=encrypted_path,
                passphrase=wrong_passphrase
            )
        
    finally:
        if encrypted_path.exists():
            encrypted_path.unlink()


def test_secure_env_loader_encrypted_preferred(temp_env_file, cleanup_env):
    """Test secure_env_loader with encrypted file preferred."""
    passphrase = "test_secure_key"
    
    encrypted_path = temp_env_file.parent / f"{temp_env_file.name}.encrypted"
    
    try:
        encrypt_env_file(
            env_path=temp_env_file,
            output_path=encrypted_path,
            passphrase=passphrase
        )
        
        env_vars = secure_env_loader(
            prefer_encrypted=True,
            encrypted_path=encrypted_path,
            plaintext_path=temp_env_file,
            passphrase=passphrase
        )
        
        assert "MT5_LOGIN" in env_vars
        assert env_vars["MT5_LOGIN"] == "12345678"
        
    finally:
        if encrypted_path.exists():
            encrypted_path.unlink()


def test_secure_env_loader_plaintext_fallback(temp_env_file, cleanup_env):
    """Test secure_env_loader falls back to plaintext."""
    env_vars = secure_env_loader(
        prefer_encrypted=True,
        encrypted_path="nonexistent.encrypted",
        plaintext_path=temp_env_file
    )
    
    assert "MT5_LOGIN" in env_vars
    assert env_vars["MT5_LOGIN"] == "12345678"


def test_encrypt_nonexistent_file():
    """Test error handling for nonexistent input file."""
    with pytest.raises(FileNotFoundError):
        encrypt_env_file(
            env_path="nonexistent.env",
            passphrase="test"
        )


def test_decrypt_nonexistent_file():
    """Test error handling for nonexistent encrypted file."""
    with pytest.raises(FileNotFoundError):
        decrypt_env_file(
            encrypted_path="nonexistent.encrypted",
            passphrase="test"
        )


def test_load_nonexistent_file():
    """Test error handling for nonexistent encrypted file during load."""
    with pytest.raises(FileNotFoundError):
        load_encrypted_env(
            encrypted_path="nonexistent.encrypted",
            passphrase="test"
        )


def test_secure_env_loader_no_files():
    """Test secure_env_loader error when no files exist."""
    with pytest.raises(FileNotFoundError, match="No environment file found"):
        secure_env_loader(
            prefer_encrypted=True,
            encrypted_path="nonexistent.encrypted",
            plaintext_path="nonexistent.env"
        )


def test_env_file_with_quotes_and_spaces(cleanup_env):
    """Test handling of various .env file formats."""
    content = '''KEY1="value with spaces"
KEY2='single quotes'
KEY3=no_quotes
KEY4 = "spaces around equals"
'''
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        f.write(content)
        temp_path = Path(f.name)
    
    encrypted_path = temp_path.parent / f"{temp_path.name}.encrypted"
    
    try:
        encrypt_env_file(
            env_path=temp_path,
            output_path=encrypted_path,
            passphrase="test"
        )
        
        env_vars = load_encrypted_env(
            encrypted_path=encrypted_path,
            passphrase="test"
        )
        
        assert env_vars["KEY1"] == "value with spaces"
        assert env_vars["KEY2"] == "single quotes"
        assert env_vars["KEY3"] == "no_quotes"
        assert env_vars["KEY4"] == "spaces around equals"
        
    finally:
        temp_path.unlink()
        if encrypted_path.exists():
            encrypted_path.unlink()


def test_encryption_with_salt_persistence(temp_env_file, cleanup_env):
    """Test that salt is properly stored and retrieved."""
    passphrase = "test_salt_key"
    
    encrypted_path = temp_env_file.parent / f"{temp_env_file.name}.encrypted"
    
    try:
        encrypt_env_file(
            env_path=temp_env_file,
            output_path=encrypted_path,
            passphrase=passphrase
        )
        
        with encrypted_path.open('rb') as f:
            salt = f.read(16)
        
        assert len(salt) == 16
        
        env_vars = load_encrypted_env(
            encrypted_path=encrypted_path,
            passphrase=passphrase
        )
        
        assert len(env_vars) > 0
        
    finally:
        if encrypted_path.exists():
            encrypted_path.unlink()
