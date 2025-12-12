"""CLI utility for encrypting and decrypting .env files.

Usage examples:
    # Encrypt .env file
    python scripts/encrypt_env.py encrypt
    
    # Encrypt with explicit passphrase
    python scripts/encrypt_env.py encrypt --passphrase "my_secret"
    
    # Decrypt .env.encrypted to .env
    python scripts/encrypt_env.py decrypt
    
    # Load encrypted env vars (decrypt in memory)
    python scripts/encrypt_env.py load
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.security.secrets import (
    encrypt_env_file,
    decrypt_env_file,
    load_encrypted_env,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encrypt/decrypt .env files for OpenQuant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    encrypt_parser = subparsers.add_parser("encrypt", help="Encrypt .env file")
    encrypt_parser.add_argument(
        "--input", "-i",
        default=".env",
        help="Path to plaintext .env file (default: .env)"
    )
    encrypt_parser.add_argument(
        "--output", "-o",
        help="Path for encrypted output (default: .env.encrypted)"
    )
    encrypt_parser.add_argument(
        "--passphrase", "-p",
        help="Encryption passphrase (or use OPENQUANT_MASTER_KEY env var)"
    )
    
    decrypt_parser = subparsers.add_parser("decrypt", help="Decrypt .env.encrypted file")
    decrypt_parser.add_argument(
        "--input", "-i",
        default=".env.encrypted",
        help="Path to encrypted .env file (default: .env.encrypted)"
    )
    decrypt_parser.add_argument(
        "--output", "-o",
        help="Path for decrypted output (default: .env.decrypted)"
    )
    decrypt_parser.add_argument(
        "--passphrase", "-p",
        help="Decryption passphrase (or use OPENQUANT_MASTER_KEY env var)"
    )
    
    load_parser = subparsers.add_parser(
        "load",
        help="Load encrypted env vars into memory (no file output)"
    )
    load_parser.add_argument(
        "--input", "-i",
        default=".env.encrypted",
        help="Path to encrypted .env file (default: .env.encrypted)"
    )
    load_parser.add_argument(
        "--passphrase", "-p",
        help="Decryption passphrase (or use OPENQUANT_MASTER_KEY env var)"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "encrypt":
            output = encrypt_env_file(
                env_path=args.input,
                output_path=args.output,
                passphrase=args.passphrase
            )
            print(f"✓ Successfully encrypted to: {output}")
            print(f"  To use: Set OPENQUANT_MASTER_KEY environment variable")
            print(f"  or pass passphrase to load_encrypted_env()")
            
        elif args.command == "decrypt":
            output = decrypt_env_file(
                encrypted_path=args.input,
                output_path=args.output,
                passphrase=args.passphrase
            )
            print(f"✓ Successfully decrypted to: {output}")
            print(f"  WARNING: Plaintext credentials now on disk!")
            
        elif args.command == "load":
            env_vars = load_encrypted_env(
                encrypted_path=args.input,
                passphrase=args.passphrase
            )
            print(f"✓ Successfully loaded {len(env_vars)} environment variables")
            print(f"  Variables loaded into os.environ:")
            for key in env_vars:
                print(f"    - {key}")
    
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
