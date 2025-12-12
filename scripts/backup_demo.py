#!/usr/bin/env python3
"""Demo script showing backup and recovery operations."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.storage.backup import BackupManager, create_daily_backup


def demo_basic_backup():
    """Demonstrate basic backup operations."""
    print("=" * 60)
    print("Basic Backup Demo")
    print("=" * 60)
    
    manager = BackupManager(backup_dir="data/backups")
    
    print("\n1. Creating a backup...")
    backup_path = manager.create_backup(compress=True)
    print(f"   ✓ Backup created: {backup_path}")
    
    print("\n2. Listing all backups...")
    backups = manager.list_backups()
    print(f"   Found {len(backups)} backup(s)")
    for backup in backups[:3]:
        size_mb = backup["size_bytes"] / (1024 * 1024)
        print(f"   - {backup['name']}: {size_mb:.2f} MB")
    
    print("\n3. Verifying latest backup...")
    if backups:
        latest = backups[0]["name"]
        result = manager.verify_backup(latest)
        if result["valid"]:
            print(f"   ✓ Backup '{latest}' is valid")
            if result.get("databases"):
                for db in result["databases"]:
                    print(f"     - {db['name']}: {len(db.get('tables', []))} tables")
        else:
            print(f"   ✗ Backup verification failed: {result.get('error')}")
    
    print("\nDemo complete!")


def demo_restore():
    """Demonstrate restore operations."""
    print("=" * 60)
    print("Restore Demo")
    print("=" * 60)
    
    manager = BackupManager(backup_dir="data/backups")
    
    print("\n1. Listing available backups...")
    backups = manager.list_backups()
    if not backups:
        print("   No backups available. Create one first!")
        return
    
    print(f"   Found {len(backups)} backup(s)")
    for i, backup in enumerate(backups[:5], 1):
        print(f"   {i}. {backup['name']} - {backup['created_at']}")
    
    print("\n2. Restore example (dry run - not actually restoring):")
    latest = backups[0]["name"]
    print(f"   To restore: manager.restore_backup('{latest}')")
    print(f"   This would:")
    print(f"     - Backup existing databases to .bak files")
    print(f"     - Restore databases from archive")
    print(f"     - Return dict of restored files")
    
    print("\nRestore demo complete (no actual restore performed)!")


def demo_daily_routine():
    """Demonstrate daily backup routine."""
    print("=" * 60)
    print("Daily Backup Routine Demo")
    print("=" * 60)
    
    print("\n1. Running daily backup routine...")
    print("   (This creates a backup and cleans up old ones)")
    
    backup_path = create_daily_backup(backup_dir="data/backups")
    
    if backup_path:
        print(f"   ✓ Daily backup completed: {backup_path}")
    else:
        print("   ✗ Daily backup failed")
    
    print("\nDaily routine demo complete!")


def demo_custom_databases():
    """Demonstrate backing up custom databases."""
    print("=" * 60)
    print("Custom Database Backup Demo")
    print("=" * 60)
    
    custom_dbs = [
        "data/results.duckdb",
        "data/audit_trail.duckdb",
    ]
    
    print(f"\n1. Setting up custom backup (only {len(custom_dbs)} databases):")
    for db in custom_dbs:
        print(f"   - {db}")
    
    manager = BackupManager(
        backup_dir="data/custom_backups",
        databases=custom_dbs
    )
    
    print("\n2. Creating custom backup...")
    backup_path = manager.create_backup(backup_name="custom_backup")
    print(f"   ✓ Custom backup created: {backup_path}")
    
    print("\nCustom backup demo complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backup system demo")
    parser.add_argument(
        "demo",
        choices=["basic", "restore", "daily", "custom", "all"],
        help="Which demo to run"
    )
    
    args = parser.parse_args()
    
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 12 + "OpenQuant Backup System Demo" + " " * 18 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    if args.demo == "basic" or args.demo == "all":
        demo_basic_backup()
        print()
    
    if args.demo == "restore" or args.demo == "all":
        demo_restore()
        print()
    
    if args.demo == "daily" or args.demo == "all":
        demo_daily_routine()
        print()
    
    if args.demo == "custom" or args.demo == "all":
        demo_custom_databases()
        print()
