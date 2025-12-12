# Usage Examples for OpenQuant Security Module

## Example 1: Basic Encryption Workflow

```python
from openquant.security import encrypt_env_file, load_encrypted_env
import os

# Step 1: Encrypt your .env file
os.environ["OPENQUANT_MASTER_KEY"] = "my_secure_passphrase_123"
encrypted_path = encrypt_env_file()
print(f"Credentials encrypted to: {encrypted_path}")

# Step 2: Load encrypted credentials at runtime
env_vars = load_encrypted_env()
print(f"Loaded {len(env_vars)} environment variables")

# Step 3: Use credentials normally
mt5_password = env_vars["MT5_PASSWORD"]
alpaca_key = env_vars["APCA_API_KEY_ID"]
```

## Example 2: CLI Workflow

```bash
# Set master key in environment
export OPENQUANT_MASTER_KEY="my_secure_passphrase_123"

# Encrypt .env file
python scripts/encrypt_env.py encrypt

# Verify by loading (shows loaded variable names)
python scripts/encrypt_env.py load

# Your application can now use encrypted credentials
python scripts/run_dashboard.py
```

## Example 3: Integration with Existing Application

Replace this in your application entry point:

```python
# OLD CODE (openquant/utils/config.py or main script)
from dotenv import load_dotenv
load_dotenv()

# NEW CODE (with encryption support)
from openquant.security import load_config_with_encrypted_env

# Try encrypted first, fall back to plaintext .env if needed
load_config_with_encrypted_env(prefer_encrypted=True)

# Now use environment variables as before
import os
mt5_password = os.environ.get("MT5_PASSWORD")
```

## Example 4: Programmatic Encryption with Custom Paths

```python
from openquant.security import encrypt_env_file, decrypt_env_file
from pathlib import Path

# Encrypt with custom paths
encrypt_env_file(
    env_path="config/production.env",
    output_path="config/production.env.encrypted",
    passphrase="production_key_xyz"
)

# Decrypt for inspection (be careful with plaintext!)
decrypt_env_file(
    encrypted_path="config/production.env.encrypted",
    output_path="config/production.env.decrypted",
    passphrase="production_key_xyz"
)
```

## Example 5: Secure Loader with Fallback

```python
from openquant.security import secure_env_loader

# Automatically try encrypted, fall back to plaintext
env_vars = secure_env_loader(
    prefer_encrypted=True,
    encrypted_path=".env.encrypted",
    plaintext_path=".env"
)

# Environment variables are now available
import os
print(f"MT5 Login: {os.environ.get('MT5_LOGIN')}")
```

## Example 6: Production Deployment

```python
# In your production startup script (e.g., scripts/run_production.py)
import os
import sys
from pathlib import Path

# Production uses encrypted credentials with key from secrets manager
if not os.getenv("OPENQUANT_MASTER_KEY"):
    print("ERROR: OPENQUANT_MASTER_KEY not set in environment")
    sys.exit(1)

# Load encrypted credentials
from openquant.security import load_encrypted_env

try:
    env_vars = load_encrypted_env(encrypted_path=".env.encrypted")
    print(f"✓ Loaded {len(env_vars)} encrypted credentials")
except Exception as e:
    print(f"✗ Failed to load credentials: {e}")
    sys.exit(1)

# Now start your application
from openquant.paper.simulator import PaperTradingSimulator
# ... rest of your application
```

## Example 7: Development vs Production

```python
# config/env_loader.py
import os
from openquant.security import secure_env_loader

def load_environment():
    """Load environment based on deployment context."""
    
    if os.getenv("ENVIRONMENT") == "production":
        # Production: require encrypted credentials
        print("Loading production credentials (encrypted)...")
        return secure_env_loader(
            prefer_encrypted=True,
            encrypted_path=".env.encrypted",
            plaintext_path=None  # Don't allow plaintext in prod
        )
    else:
        # Development: allow plaintext for convenience
        print("Loading development credentials...")
        return secure_env_loader(
            prefer_encrypted=False,
            encrypted_path=".env.encrypted",
            plaintext_path=".env"
        )

# In your main application
from config.env_loader import load_environment
load_environment()
```

## Example 8: Rotating Credentials

```python
from openquant.security import encrypt_env_file
import os

def rotate_credentials(old_passphrase: str, new_passphrase: str):
    """Rotate encryption passphrase."""
    
    # Decrypt with old passphrase
    from openquant.security import decrypt_env_file
    
    decrypt_env_file(
        encrypted_path=".env.encrypted",
        output_path=".env.temp",
        passphrase=old_passphrase
    )
    
    # Re-encrypt with new passphrase
    encrypt_env_file(
        env_path=".env.temp",
        output_path=".env.encrypted.new",
        passphrase=new_passphrase
    )
    
    # Cleanup
    os.remove(".env.temp")
    os.rename(".env.encrypted.new", ".env.encrypted")
    
    print("✓ Credentials rotated successfully")

# Usage
rotate_credentials(
    old_passphrase="old_key_123",
    new_passphrase="new_key_456"
)
```

## Example 9: CI/CD Integration

```yaml
# .github/workflows/deploy.yml
name: Deploy with Encrypted Credentials

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Deploy with encrypted credentials
        env:
          OPENQUANT_MASTER_KEY: ${{ secrets.MASTER_KEY }}
        run: |
          python scripts/encrypt_env.py load
          python scripts/deploy.py
```

## Example 10: Error Handling

```python
from openquant.security import load_encrypted_env
import sys

def safe_load_credentials():
    """Load credentials with proper error handling."""
    try:
        env_vars = load_encrypted_env(encrypted_path=".env.encrypted")
        return env_vars
    
    except FileNotFoundError:
        print("ERROR: .env.encrypted not found")
        print("Run: python scripts/encrypt_env.py encrypt")
        sys.exit(1)
    
    except ValueError as e:
        if "Invalid passphrase" in str(e):
            print("ERROR: Invalid passphrase")
            print("Check OPENQUANT_MASTER_KEY environment variable")
        else:
            print(f"ERROR: {e}")
        sys.exit(1)
    
    except Exception as e:
        print(f"Unexpected error loading credentials: {e}")
        sys.exit(1)

# Use in application
credentials = safe_load_credentials()
```

## Example 11: Testing with Encrypted Credentials

```python
# tests/conftest.py
import pytest
import os
from pathlib import Path
from openquant.security import encrypt_env_file

@pytest.fixture(scope="session")
def encrypted_test_credentials():
    """Provide encrypted test credentials for test suite."""
    
    test_env_content = """
MT5_LOGIN="12345678"
MT5_PASSWORD="test_password"
APCA_API_KEY_ID="test_key"
"""
    
    test_env_path = Path(".env.test")
    test_env_path.write_text(test_env_content)
    
    # Encrypt test credentials
    encrypted_path = encrypt_env_file(
        env_path=test_env_path,
        output_path=".env.test.encrypted",
        passphrase="test_passphrase_123"
    )
    
    # Set passphrase for tests
    os.environ["OPENQUANT_MASTER_KEY"] = "test_passphrase_123"
    
    yield encrypted_path
    
    # Cleanup
    test_env_path.unlink(missing_ok=True)
    encrypted_path.unlink(missing_ok=True)

# In your tests
def test_with_credentials(encrypted_test_credentials):
    from openquant.security import load_encrypted_env
    
    env_vars = load_encrypted_env(encrypted_path=encrypted_test_credentials)
    assert env_vars["MT5_LOGIN"] == "12345678"
```

## Example 12: Multi-Environment Setup

```python
# config/environments.py
import os
from pathlib import Path
from openquant.security import load_encrypted_env

class EnvironmentConfig:
    """Manage different environment configurations."""
    
    @staticmethod
    def load(env_name: str = None):
        """Load environment-specific configuration."""
        if env_name is None:
            env_name = os.getenv("OPENQUANT_ENV", "development")
        
        env_file_map = {
            "development": ".env.dev.encrypted",
            "staging": ".env.staging.encrypted",
            "production": ".env.prod.encrypted",
        }
        
        encrypted_path = env_file_map.get(env_name)
        if not encrypted_path:
            raise ValueError(f"Unknown environment: {env_name}")
        
        if not Path(encrypted_path).exists():
            raise FileNotFoundError(
                f"Environment file not found: {encrypted_path}"
            )
        
        return load_encrypted_env(encrypted_path=encrypted_path)

# Usage
from config.environments import EnvironmentConfig

# Load based on OPENQUANT_ENV variable
os.environ["OPENQUANT_ENV"] = "production"
config = EnvironmentConfig.load()
```

## Security Best Practices

1. **Never commit encrypted files without separate key management**
2. **Use strong passphrases (16+ characters)**
3. **Store OPENQUANT_MASTER_KEY in secure secret managers (AWS Secrets Manager, Azure Key Vault, etc.)**
4. **Rotate credentials regularly**
5. **Use different passphrases for different environments**
6. **Audit access to encrypted files and master keys**
7. **Use `load_encrypted_env()` instead of `decrypt_env_file()` to avoid plaintext on disk**
