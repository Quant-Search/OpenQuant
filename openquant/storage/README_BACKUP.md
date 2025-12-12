# Database Backup and Recovery

Automated backup and recovery system for OpenQuant databases.

## Overview

The backup system provides automated daily snapshots of all critical databases:
- `results.duckdb` - Research results and portfolio data
- `audit_trail.duckdb` - Audit trail logs
- `tca.duckdb` - Transaction cost analysis

Backups are compressed using tar.gz compression and stored in `data/backups/`.

## Quick Start

### Manual Backup

```bash
# Create a backup now
python3 scripts/backup_databases.py backup

# Create a backup with custom name
python3 scripts/backup_databases.py backup --name my_backup

# Create uncompressed backup
python3 scripts/backup_databases.py backup --no-compress
```

### Restore from Backup

```bash
# List available backups
python3 scripts/backup_databases.py list

# Restore a backup (creates .bak files of existing databases)
python3 scripts/backup_databases.py restore backup_20240115_120000

# Restore with overwrite (no backup of existing files)
python3 scripts/backup_databases.py restore backup_20240115_120000 --overwrite

# Restore to custom location
python3 scripts/backup_databases.py restore backup_20240115_120000 --restore-dir /path/to/restore
```

### Verify Backup Integrity

```bash
# Verify a backup file
python3 scripts/backup_databases.py verify backup_20240115_120000
```

### Cleanup Old Backups

```bash
# Delete backups older than 30 days (keeps minimum 5 most recent)
python3 scripts/backup_databases.py cleanup

# Custom retention policy
python3 scripts/backup_databases.py cleanup --keep-days 60 --keep-minimum 10
```

## Automated Daily Backups

### Windows

Run as Administrator:

```batch
scripts\schedule_daily_backup.bat
```

This creates a Windows Task Scheduler task that runs daily at 2:00 AM.

**Manage the scheduled task:**
```batch
# View task
schtasks /query /tn "OpenQuant Daily Backup"

# Delete task
schtasks /delete /tn "OpenQuant Daily Backup" /f

# Run manually
python scripts\backup_databases.py daily
```

### Linux/Mac

```bash
chmod +x scripts/schedule_daily_backup.sh
./scripts/schedule_daily_backup.sh
```

This adds a cron job that runs daily at 2:00 AM.

**Manage the cron job:**
```bash
# View all cron jobs
crontab -l

# Edit cron jobs
crontab -e

# Run manually
python3 scripts/backup_databases.py daily

# View logs
tail -f data/backup.log
```

## Programmatic Usage

### Python API

```python
from openquant.storage.backup import BackupManager, create_daily_backup

# Initialize backup manager
manager = BackupManager(backup_dir="data/backups")

# Create a backup
backup_path = manager.create_backup(backup_name="my_backup", compress=True)
print(f"Backup created: {backup_path}")

# List backups
backups = manager.list_backups()
for backup in backups:
    print(f"{backup['name']} - {backup['created_at']}")

# Restore a backup
restored = manager.restore_backup("backup_20240115_120000", overwrite=False)
print(f"Restored: {restored}")

# Verify a backup
result = manager.verify_backup("backup_20240115_120000")
print(f"Valid: {result['valid']}")

# Cleanup old backups
deleted = manager.cleanup_old_backups(keep_days=30, keep_minimum=5)
print(f"Deleted {deleted} old backups")

# Daily backup routine (backup + cleanup)
backup_path = create_daily_backup(backup_dir="data/backups")
```

### Custom Database List

```python
from openquant.storage.backup import BackupManager

# Backup custom databases
custom_dbs = [
    "data/results.duckdb",
    "data/custom.duckdb",
]

manager = BackupManager(
    backup_dir="data/custom_backups",
    databases=custom_dbs
)

backup_path = manager.create_backup()
```

## Backup Format

### Archive Structure

```
backup_20240115_120000.tar.gz
├── backup_metadata.json       # Backup metadata
├── results.duckdb            # Results database
├── audit_trail.duckdb        # Audit trail
└── tca.duckdb               # TCA database
```

### Metadata Format

```json
{
  "timestamp": "2024-01-15T12:00:00+00:00",
  "databases": [
    {
      "name": "results.duckdb",
      "original_path": "data/results.duckdb",
      "size_bytes": 1048576
    }
  ]
}
```

## Features

### Locked Database Handling

The backup system can handle databases that are currently in use:
- Attempts read-only connection first
- Falls back to DuckDB EXPORT/IMPORT if locked
- Falls back to file copy if all else fails

### Safety Features

- **Automatic backup of existing files**: When restoring, existing databases are backed up to `.duckdb.bak.TIMESTAMP` files
- **Backup verification**: Verify integrity before restore
- **Retention policy**: Automatic cleanup with minimum backup count
- **Metadata tracking**: Full audit trail of what's backed up

### Compression

- Backups are compressed using tar.gz (gzip)
- Typical compression ratio: 5-10x for DuckDB files
- Can disable compression with `--no-compress` flag

## Retention Policy

Default retention policy (can be customized):
- **Keep days**: 30 days
- **Keep minimum**: 5 most recent backups

Even if backups are older than 30 days, the 5 most recent will always be kept.

## Troubleshooting

### Backup Failed: Database Locked

If a database is actively being written to, the backup system will:
1. Try checkpoint backup using DuckDB EXPORT/IMPORT
2. Fall back to file copy
3. Log warnings but continue with other databases

### Restore Failed: File Not Found

```bash
# Verify backup exists
python3 scripts/backup_databases.py list

# Check the exact name
python3 scripts/backup_databases.py list | grep backup_
```

### Permission Errors

**Windows**: Run Command Prompt or PowerShell as Administrator

**Linux/Mac**: Ensure you have write permissions to the data directory:
```bash
chmod -R u+w data/
```

### Verify After Restore

Always verify databases after restore:
```python
import duckdb

# Verify results.duckdb
con = duckdb.connect("data/results.duckdb", read_only=True)
print(con.execute("SELECT COUNT(*) FROM results").fetchone())
con.close()
```

## Best Practices

1. **Test restores regularly**: Verify backups can actually be restored
2. **Monitor backup logs**: Check `data/backup.log` for errors
3. **Offsite backups**: Copy backups to external storage or cloud
4. **Before major changes**: Create manual backup before schema changes or bulk updates
5. **Storage management**: Monitor `data/backups/` disk usage

## Integration with Paper Trading

The backup system automatically includes portfolio data stored in `results.duckdb`:
- `portfolio_trades` table
- `portfolio_positions` table  
- `portfolio_equity` table

## FAQ

**Q: Can I backup while trading is active?**  
A: Yes, the backup system handles locked databases gracefully.

**Q: How much disk space do I need?**  
A: Plan for ~10-20x the size of your databases (for 30 days of retention with compression).

**Q: Can I restore individual tables?**  
A: No, backups restore entire databases. For table-level operations, use DuckDB's EXPORT/IMPORT directly.

**Q: What happens if backup fails?**  
A: The system logs errors but continues with other databases. Check logs in `data/backup.log`.

**Q: Can I change the backup schedule?**  
A: Yes, edit the Windows Task or cron job to use your preferred schedule.
