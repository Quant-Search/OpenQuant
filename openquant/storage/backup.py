"""Database backup and recovery module.

Provides automated daily snapshots of DuckDB databases with compression:
- results.duckdb: Research results
- portfolio tables (in results.duckdb): Trades, positions, equity
- audit_trail.duckdb: Audit trail
- tca.duckdb: Transaction cost analysis

Features:
- Compressed backup archives (tar.gz)
- Automated daily snapshots
- Point-in-time restore
- Backup rotation and cleanup
"""
from __future__ import annotations
import tarfile
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
import duckdb
import json

from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)


class BackupManager:
    """Manages database backups and recovery."""

    DEFAULT_DATABASES = [
        "data/results.duckdb",
        "data/audit_trail.duckdb",
        "data/tca.duckdb",
    ]

    def __init__(
        self,
        backup_dir: str | Path = "data/backups",
        databases: Optional[List[str | Path]] = None,
    ):
        """Initialize backup manager.

        Args:
            backup_dir: Directory to store backups
            databases: List of database paths to backup (uses defaults if None)
        """
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.databases = [Path(db) for db in (databases or self.DEFAULT_DATABASES)]

    def create_backup(
        self,
        backup_name: Optional[str] = None,
        compress: bool = True,
    ) -> Path:
        """Create a backup of all configured databases.

        Args:
            backup_name: Name for backup (default: timestamp-based)
            compress: Whether to compress the backup (default: True)

        Returns:
            Path to the backup file or directory
        """
        timestamp = datetime.now(timezone.utc)
        if backup_name is None:
            backup_name = timestamp.strftime("backup_%Y%m%d_%H%M%S")

        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "timestamp": timestamp.isoformat(),
            "databases": [],
        }

        for db_path in self.databases:
            if not db_path.exists():
                LOGGER.warning(f"Database {db_path} does not exist, skipping")
                continue

            try:
                db_name = db_path.name
                backup_db_path = backup_path / db_name

                if self._is_database_locked(db_path):
                    LOGGER.warning(f"Database {db_path} is locked, creating a checkpoint backup")
                    self._checkpoint_backup(db_path, backup_db_path)
                else:
                    shutil.copy2(db_path, backup_db_path)

                file_size = backup_db_path.stat().st_size
                metadata["databases"].append({
                    "name": db_name,
                    "original_path": str(db_path),
                    "size_bytes": file_size,
                })
                LOGGER.info(f"Backed up {db_name} ({file_size:,} bytes)")

            except Exception as e:
                LOGGER.error(f"Failed to backup {db_path}: {e}")
                continue

        metadata_path = backup_path / "backup_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if compress:
            archive_path = self.backup_dir / f"{backup_name}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(backup_path, arcname=backup_name)

            shutil.rmtree(backup_path)
            final_size = archive_path.stat().st_size
            LOGGER.info(f"Created compressed backup: {archive_path} ({final_size:,} bytes)")
            return archive_path
        else:
            LOGGER.info(f"Created uncompressed backup: {backup_path}")
            return backup_path

    def _is_database_locked(self, db_path: Path) -> bool:
        """Check if a database is currently locked by another connection."""
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            con.close()
            return False
        except duckdb.IOException as e:
            if "lock" in str(e).lower():
                return True
            return False
        except Exception:
            return False

    def _checkpoint_backup(self, source: Path, dest: Path) -> None:
        """Create a backup using DuckDB's EXPORT DATABASE functionality."""
        try:
            temp_export_dir = dest.parent / f"temp_export_{source.stem}"
            temp_export_dir.mkdir(parents=True, exist_ok=True)

            con = duckdb.connect(str(source), read_only=True)
            try:
                con.execute(f"EXPORT DATABASE '{temp_export_dir}'")
            finally:
                con.close()

            con_new = duckdb.connect(str(dest))
            try:
                con_new.execute(f"IMPORT DATABASE '{temp_export_dir}'")
            finally:
                con_new.close()

            shutil.rmtree(temp_export_dir)

        except Exception as e:
            LOGGER.error(f"Checkpoint backup failed: {e}, falling back to file copy")
            if dest.exists():
                dest.unlink()
            shutil.copy2(source, dest)

    def restore_backup(
        self,
        backup_name: str,
        restore_dir: Optional[str | Path] = None,
        overwrite: bool = False,
    ) -> Dict[str, Path]:
        """Restore databases from a backup.

        Args:
            backup_name: Name of the backup (with or without .tar.gz extension)
            restore_dir: Directory to restore to (default: original locations)
            overwrite: Whether to overwrite existing databases

        Returns:
            Dictionary mapping database names to restored paths
        """
        if not backup_name.endswith(".tar.gz"):
            backup_name = f"{backup_name}.tar.gz"

        archive_path = self.backup_dir / backup_name
        if not archive_path.exists():
            uncompressed_path = self.backup_dir / backup_name.replace(".tar.gz", "")
            if uncompressed_path.exists():
                backup_path = uncompressed_path
            else:
                raise FileNotFoundError(f"Backup not found: {archive_path}")
        else:
            temp_extract_dir = self.backup_dir / f"temp_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            temp_extract_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(temp_extract_dir)

            extracted_dirs = list(temp_extract_dir.iterdir())
            if not extracted_dirs:
                raise ValueError(f"No content found in backup archive: {archive_path}")

            backup_path = extracted_dirs[0]

        metadata_path = backup_path / "backup_metadata.json"
        if not metadata_path.exists():
            LOGGER.warning(f"No metadata found in backup: {backup_path}")
            metadata = {"databases": []}
        else:
            with open(metadata_path) as f:
                metadata = json.load(f)

        restored = {}
        for db_file in backup_path.glob("*.duckdb"):
            if restore_dir:
                target_path = Path(restore_dir) / db_file.name
                target_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                db_info = next(
                    (d for d in metadata.get("databases", []) if d["name"] == db_file.name),
                    None
                )
                if db_info:
                    target_path = Path(db_info["original_path"])
                else:
                    target_path = Path("data") / db_file.name

            if target_path.exists() and not overwrite:
                backup_existing = target_path.with_suffix(
                    f".duckdb.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                shutil.copy2(target_path, backup_existing)
                LOGGER.info(f"Backed up existing database to {backup_existing}")

            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(db_file, target_path)
            restored[db_file.name] = target_path
            LOGGER.info(f"Restored {db_file.name} to {target_path}")

        if archive_path.exists() and "temp_restore_" in str(backup_path):
            shutil.rmtree(temp_extract_dir)

        return restored

    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups with metadata.

        Returns:
            List of backup information dictionaries
        """
        backups = []

        for archive in sorted(self.backup_dir.glob("backup_*.tar.gz"), reverse=True):
            try:
                size = archive.stat().st_size
                mtime = datetime.fromtimestamp(archive.stat().st_mtime, tz=timezone.utc)

                with tarfile.open(archive, "r:gz") as tar:
                    members = tar.getmembers()
                    metadata_member = next(
                        (m for m in members if m.name.endswith("backup_metadata.json")),
                        None
                    )

                    if metadata_member:
                        metadata_file = tar.extractfile(metadata_member)
                        if metadata_file:
                            metadata = json.load(metadata_file)
                        else:
                            metadata = {}
                    else:
                        metadata = {}

                backups.append({
                    "name": archive.stem,
                    "path": str(archive),
                    "size_bytes": size,
                    "created_at": mtime.isoformat(),
                    "databases": metadata.get("databases", []),
                })
            except Exception as e:
                LOGGER.error(f"Failed to read backup {archive}: {e}")
                continue

        for backup_dir in sorted(self.backup_dir.glob("backup_*"), reverse=True):
            if backup_dir.is_dir():
                try:
                    metadata_path = backup_dir / "backup_metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                    else:
                        metadata = {}

                    total_size = sum(f.stat().st_size for f in backup_dir.rglob("*") if f.is_file())
                    mtime = datetime.fromtimestamp(backup_dir.stat().st_mtime, tz=timezone.utc)

                    backups.append({
                        "name": backup_dir.name,
                        "path": str(backup_dir),
                        "size_bytes": total_size,
                        "created_at": mtime.isoformat(),
                        "databases": metadata.get("databases", []),
                        "compressed": False,
                    })
                except Exception as e:
                    LOGGER.error(f"Failed to read backup directory {backup_dir}: {e}")
                    continue

        return backups

    def cleanup_old_backups(self, keep_days: int = 30, keep_minimum: int = 5) -> int:
        """Delete backups older than specified days, keeping a minimum number.

        Args:
            keep_days: Delete backups older than this many days
            keep_minimum: Always keep at least this many most recent backups

        Returns:
            Number of backups deleted
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=keep_days)
        backups = self.list_backups()

        if len(backups) <= keep_minimum:
            LOGGER.info(f"Only {len(backups)} backups exist, skipping cleanup")
            return 0

        deleted = 0
        for backup in backups[keep_minimum:]:
            try:
                created_at = datetime.fromisoformat(backup["created_at"])
                if created_at < cutoff_date:
                    backup_path = Path(backup["path"])
                    if backup_path.is_file():
                        backup_path.unlink()
                    elif backup_path.is_dir():
                        shutil.rmtree(backup_path)

                    deleted += 1
                    LOGGER.info(f"Deleted old backup: {backup['name']}")
            except Exception as e:
                LOGGER.error(f"Failed to delete backup {backup['name']}: {e}")
                continue

        return deleted

    def verify_backup(self, backup_name: str) -> Dict[str, Any]:
        """Verify the integrity of a backup.

        Args:
            backup_name: Name of the backup to verify

        Returns:
            Dictionary with verification results
        """
        if not backup_name.endswith(".tar.gz"):
            backup_name = f"{backup_name}.tar.gz"

        archive_path = self.backup_dir / backup_name
        if not archive_path.exists():
            return {"valid": False, "error": "Backup file not found"}

        try:
            temp_verify_dir = self.backup_dir / f"temp_verify_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            temp_verify_dir.mkdir(parents=True, exist_ok=True)

            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(temp_verify_dir)

            extracted_dirs = list(temp_verify_dir.iterdir())
            if not extracted_dirs:
                return {"valid": False, "error": "Empty backup archive"}

            backup_path = extracted_dirs[0]
            db_files = list(backup_path.glob("*.duckdb"))

            verified_dbs = []
            for db_file in db_files:
                try:
                    con = duckdb.connect(str(db_file), read_only=True)
                    tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                    con.close()

                    verified_dbs.append({
                        "name": db_file.name,
                        "valid": True,
                        "tables": [t[0] for t in tables],
                    })
                except Exception as e:
                    verified_dbs.append({
                        "name": db_file.name,
                        "valid": False,
                        "error": str(e),
                    })

            shutil.rmtree(temp_verify_dir)

            return {
                "valid": all(db["valid"] for db in verified_dbs),
                "databases": verified_dbs,
            }

        except Exception as e:
            return {"valid": False, "error": str(e)}


def create_daily_backup(backup_dir: str | Path = "data/backups") -> Optional[Path]:
    """Create a daily backup snapshot. Intended for scheduled execution.

    Args:
        backup_dir: Directory to store backups

    Returns:
        Path to created backup, or None if failed
    """
    try:
        manager = BackupManager(backup_dir=backup_dir)
        backup_path = manager.create_backup(compress=True)
        LOGGER.info(f"Daily backup created: {backup_path}")

        deleted = manager.cleanup_old_backups(keep_days=30, keep_minimum=5)
        if deleted > 0:
            LOGGER.info(f"Cleaned up {deleted} old backups")

        return backup_path
    except Exception as e:
        LOGGER.error(f"Daily backup failed: {e}")
        return None
