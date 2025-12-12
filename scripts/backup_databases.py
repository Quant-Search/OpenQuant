#!/usr/bin/env python3
"""CLI script for database backup and recovery operations."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.storage.backup import BackupManager, create_daily_backup


def main():
    parser = argparse.ArgumentParser(
        description="Database backup and recovery CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    backup_parser = subparsers.add_parser("backup", help="Create a backup")
    backup_parser.add_argument(
        "--name",
        help="Backup name (default: timestamp-based)",
    )
    backup_parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Skip compression",
    )
    backup_parser.add_argument(
        "--backup-dir",
        default="data/backups",
        help="Backup directory (default: data/backups)",
    )

    restore_parser = subparsers.add_parser("restore", help="Restore from a backup")
    restore_parser.add_argument(
        "backup_name",
        help="Name of backup to restore",
    )
    restore_parser.add_argument(
        "--restore-dir",
        help="Directory to restore to (default: original locations)",
    )
    restore_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing databases without backup",
    )
    restore_parser.add_argument(
        "--backup-dir",
        default="data/backups",
        help="Backup directory (default: data/backups)",
    )

    list_parser = subparsers.add_parser("list", help="List available backups")
    list_parser.add_argument(
        "--backup-dir",
        default="data/backups",
        help="Backup directory (default: data/backups)",
    )

    verify_parser = subparsers.add_parser("verify", help="Verify a backup")
    verify_parser.add_argument(
        "backup_name",
        help="Name of backup to verify",
    )
    verify_parser.add_argument(
        "--backup-dir",
        default="data/backups",
        help="Backup directory (default: data/backups)",
    )

    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old backups")
    cleanup_parser.add_argument(
        "--keep-days",
        type=int,
        default=30,
        help="Keep backups from last N days (default: 30)",
    )
    cleanup_parser.add_argument(
        "--keep-minimum",
        type=int,
        default=5,
        help="Always keep at least N most recent backups (default: 5)",
    )
    cleanup_parser.add_argument(
        "--backup-dir",
        default="data/backups",
        help="Backup directory (default: data/backups)",
    )

    daily_parser = subparsers.add_parser("daily", help="Run daily backup routine")
    daily_parser.add_argument(
        "--backup-dir",
        default="data/backups",
        help="Backup directory (default: data/backups)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    manager = BackupManager(backup_dir=args.backup_dir)

    if args.command == "backup":
        print(f"Creating backup...")
        backup_path = manager.create_backup(
            backup_name=args.name,
            compress=not args.no_compress,
        )
        print(f"Backup created: {backup_path}")
        return 0

    elif args.command == "restore":
        print(f"Restoring backup: {args.backup_name}")
        try:
            restored = manager.restore_backup(
                backup_name=args.backup_name,
                restore_dir=args.restore_dir,
                overwrite=args.overwrite,
            )
            print(f"Successfully restored {len(restored)} database(s):")
            for db_name, path in restored.items():
                print(f"  {db_name} -> {path}")
            return 0
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
        except Exception as e:
            print(f"Restore failed: {e}")
            return 1

    elif args.command == "list":
        backups = manager.list_backups()
        if not backups:
            print("No backups found.")
            return 0

        print(f"Found {len(backups)} backup(s):\n")
        for backup in backups:
            compressed = backup.get("compressed", True)
            compression = "compressed" if compressed else "uncompressed"
            size_mb = backup["size_bytes"] / (1024 * 1024)
            print(f"  {backup['name']} ({compression})")
            print(f"    Created: {backup['created_at']}")
            print(f"    Size: {size_mb:.2f} MB")
            if backup.get("databases"):
                print(f"    Databases: {len(backup['databases'])}")
                for db in backup["databases"]:
                    db_size_mb = db.get("size_bytes", 0) / (1024 * 1024)
                    print(f"      - {db['name']} ({db_size_mb:.2f} MB)")
            print()
        return 0

    elif args.command == "verify":
        print(f"Verifying backup: {args.backup_name}")
        result = manager.verify_backup(args.backup_name)
        if result["valid"]:
            print("✓ Backup is valid")
            if result.get("databases"):
                print("\nDatabases:")
                for db in result["databases"]:
                    status = "✓" if db["valid"] else "✗"
                    print(f"  {status} {db['name']}")
                    if db.get("tables"):
                        print(f"      Tables: {', '.join(db['tables'])}")
            return 0
        else:
            print(f"✗ Backup is invalid: {result.get('error', 'Unknown error')}")
            return 1

    elif args.command == "cleanup":
        print(f"Cleaning up backups older than {args.keep_days} days...")
        deleted = manager.cleanup_old_backups(
            keep_days=args.keep_days,
            keep_minimum=args.keep_minimum,
        )
        print(f"Deleted {deleted} old backup(s)")
        return 0

    elif args.command == "daily":
        print("Running daily backup routine...")
        backup_path = create_daily_backup(backup_dir=args.backup_dir)
        if backup_path:
            print(f"Daily backup completed: {backup_path}")
            return 0
        else:
            print("Daily backup failed")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
