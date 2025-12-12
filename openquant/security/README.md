# OpenQuant Security Module

This module provides credentials encryption functionality to protect sensitive information in `.env` files at rest.

## Features

- **Fernet Symmetric Encryption**: Industry-standard encryption using `cryptography.fernet`
- **Key Derivation**: PBKDF2-based key derivation from user passphrase with 480,000 iterations
- **Multiple Key Sources**: Passphrase via parameter, environment variable, or interactive prompt
- **In-Memory Decryption**: Load encrypted credentials without writing plaintext to disk
- **Secure Fallback**: Gracefully fall back to plaintext `.env` if encrypted version unavailable

## Installation

The `cryptography` package is required and included in `requirements.txt`:

```bash
pip install cryptography
```

## Quick Start

### 1. Encrypt Your .env File

```bash
# Using CLI utility
python scripts/encrypt_env.py encrypt

# Or with explicit passphrase
python scripts/encrypt_env.py encrypt --passphrase "my_secret_passphrase"

# Or using environment variable
export OPENQUANT_MASTER_KEY="my_secret_passphrase"
python scripts/encrypt_env.py encrypt
```

This creates `.env.encrypted` containing your encrypted credentials.

### 2. Load Encrypted Credentials at Runtime

```python
from openquant.security.secrets import load_encrypted_env

# Using environment variable OPENQUANT_MASTER_KEY
load_encrypted_env()

# Or with explicit passphrase
load_encrypted_env(passphrase="my_secret_passphrase")

# Variables are automatically loaded into os.environ
```

### 3. Using Secure Loader with Fallback

```python
from openquant.security.secrets import secure_env_loader

# Try encrypted first, fall back to plaintext if needed
env_vars = secure_env_loader(prefer_encrypted=True)
```

## API Reference

### `encrypt_env_file()`

Encrypts a plaintext `.env` file.

```python
from openquant.security.secrets import encrypt_env_file

encrypt_env_file(
    env_path=".env",              # Input plaintext file
    output_path=".env.encrypted", # Output encrypted file
    passphrase="my_secret"        # Optional: or use OPENQUANT_MASTER_KEY
)
```

**Parameters:**
- `env_path`: Path to plaintext `.env` file (default: `.env`)
- `output_path`: Path for encrypted output (default: `.env.encrypted`)
- `passphrase`: Encryption passphrase (optional if `OPENQUANT_MASTER_KEY` set)
- `salt`: Salt bytes for key derivation (auto-generated if not provided)

**Returns:** Path object to encrypted file

### `decrypt_env_file()`

Decrypts an encrypted `.env` file to disk.

```python
from openquant.security.secrets import decrypt_env_file

decrypt_env_file(
    encrypted_path=".env.encrypted",
    output_path=".env.decrypted",
    passphrase="my_secret"
)
```

**Parameters:**
- `encrypted_path`: Path to encrypted file (default: `.env.encrypted`)
- `output_path`: Path for decrypted output (default: `.env.decrypted`)
- `passphrase`: Decryption passphrase (optional if `OPENQUANT_MASTER_KEY` set)

**Returns:** Path object to decrypted file

### `load_encrypted_env()`

Decrypts and loads environment variables in memory (recommended).

```python
from openquant.security.secrets import load_encrypted_env

env_vars = load_encrypted_env(
    encrypted_path=".env.encrypted",
    passphrase="my_secret"
)

# Variables are automatically loaded into os.environ
print(env_vars["MT5_PASSWORD"])
```

**Parameters:**
- `encrypted_path`: Path to encrypted file (default: `.env.encrypted`)
- `passphrase`: Decryption passphrase (optional if `OPENQUANT_MASTER_KEY` set)

**Returns:** Dictionary of environment variables

### `secure_env_loader()`

Loads environment variables with automatic fallback.

```python
from openquant.security.secrets import secure_env_loader

env_vars = secure_env_loader(
    prefer_encrypted=True,
    encrypted_path=".env.encrypted",
    plaintext_path=".env",
    passphrase="my_secret"
)
```

**Parameters:**
- `prefer_encrypted`: Try encrypted file first (default: `True`)
- `encrypted_path`: Path to encrypted file (default: `.env.encrypted`)
- `plaintext_path`: Path to plaintext file (default: `.env`)
- `passphrase`: Decryption passphrase (optional if `OPENQUANT_MASTER_KEY` set)

**Returns:** Dictionary of environment variables

## CLI Utility

The `scripts/encrypt_env.py` utility provides command-line access to encryption features.

### Encrypt

```bash
# Basic encryption
python scripts/encrypt_env.py encrypt

# With custom paths
python scripts/encrypt_env.py encrypt --input my.env --output my.env.enc

# With explicit passphrase
python scripts/encrypt_env.py encrypt --passphrase "my_secret"
```

### Decrypt

```bash
# Basic decryption
python scripts/encrypt_env.py decrypt

# With custom paths
python scripts/encrypt_env.py decrypt --input my.env.enc --output my.env.plain

# With explicit passphrase
python scripts/encrypt_env.py decrypt --passphrase "my_secret"
```

### Load (In-Memory)

```bash
# Load without writing plaintext to disk
python scripts/encrypt_env.py load

# With custom path
python scripts/encrypt_env.py load --input my.env.enc
```

## Security Best Practices

1. **Never commit `.env` or `.env.encrypted` to version control**
   - Both are already in `.gitignore`
   - Only commit `.env.example` with placeholder values

2. **Use strong passphrases**
   - Minimum 16 characters recommended
   - Mix of letters, numbers, and symbols

3. **Prefer `OPENQUANT_MASTER_KEY` environment variable**
   - Set in secure CI/CD environments
   - Avoid hardcoding in scripts

4. **Use `load_encrypted_env()` instead of `decrypt_env_file()`**
   - Keeps credentials in memory only
   - Reduces attack surface

5. **Rotate credentials regularly**
   - Re-encrypt with new passphrase periodically
   - Update `OPENQUANT_MASTER_KEY` in deployment environments

## Integration with Existing Code

To integrate with existing OpenQuant code that uses `python-dotenv`:

```python
# Before (in openquant/utils/config.py or similar)
from dotenv import load_dotenv
load_dotenv()

# After (with encryption support)
from openquant.security.secrets import secure_env_loader

try:
    secure_env_loader(prefer_encrypted=True)
except FileNotFoundError:
    # Fall back to standard dotenv if no .env files exist
    from dotenv import load_dotenv
    load_dotenv()
```

## File Format

The encrypted `.env.encrypted` file has the following binary format:

```
[16 bytes: salt][N bytes: Fernet-encrypted data]
```

- First 16 bytes: Random salt for PBKDF2 key derivation
- Remaining bytes: Fernet-encrypted plaintext `.env` content

## Technical Details

- **Encryption Algorithm**: Fernet (AES-128-CBC + HMAC-SHA256)
- **Key Derivation**: PBKDF2-HMAC-SHA256 with 480,000 iterations
- **Salt**: 16 random bytes, stored with encrypted data
- **Key Length**: 32 bytes (256 bits), base64-encoded for Fernet

## Error Handling

The module raises clear exceptions for common issues:

- `FileNotFoundError`: Input file doesn't exist
- `ValueError`: Invalid passphrase, no passphrase available, or corrupted file
- `cryptography` exceptions: Low-level encryption errors

## Examples

### Example 1: Encrypt Existing .env

```python
from openquant.security.secrets import encrypt_env_file
import os

os.environ["OPENQUANT_MASTER_KEY"] = "my_secure_passphrase"
encrypt_env_file()
# Creates .env.encrypted
```

### Example 2: Load in Application

```python
from openquant.security.secrets import load_encrypted_env

env_vars = load_encrypted_env()
mt5_password = env_vars["MT5_PASSWORD"]
alpaca_key = env_vars["APCA_API_KEY_ID"]
```

### Example 3: Secure Deployment

```bash
# In CI/CD or production environment
export OPENQUANT_MASTER_KEY="production_passphrase_from_secrets_manager"
python scripts/run_dashboard.py
# Application uses load_encrypted_env() to access credentials
```

### Example 4: Development Workflow

```python
# Developer's local machine
from openquant.security.secrets import secure_env_loader

# Try encrypted first, fall back to plaintext for development
env_vars = secure_env_loader(prefer_encrypted=True)
```

## Troubleshooting

**Q: "No passphrase provided" error**  
A: Set `OPENQUANT_MASTER_KEY` environment variable or pass `passphrase` parameter

**Q: "Decryption failed" error**  
A: Wrong passphrase or corrupted file. Verify passphrase matches encryption

**Q: Want to change passphrase**  
A: Decrypt to plaintext, then encrypt with new passphrase:
```bash
python scripts/encrypt_env.py decrypt --output .env
python scripts/encrypt_env.py encrypt --passphrase "new_passphrase"
```

**Q: Lost passphrase**  
A: No recovery possible. Restore from `.env` backup or reconfigure credentials

## Migration Guide

To migrate existing OpenQuant installation to use encrypted credentials:

1. Backup your `.env` file
2. Install cryptography: `pip install -r requirements.txt`
3. Encrypt your credentials: `python scripts/encrypt_env.py encrypt`
4. Set `OPENQUANT_MASTER_KEY` in your environment
5. Test by running: `python scripts/encrypt_env.py load`
6. (Optional) Delete plaintext `.env` file for security

## License

Part of OpenQuant project. See LICENSE file for details.
