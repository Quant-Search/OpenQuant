"""Storage module for OpenQuant.

Provides database storage and backup functionality for:
- Research results
- Portfolio data
- Audit trails
- Transaction cost analysis
"""
from openquant.storage.backup import BackupManager, create_daily_backup

__all__ = [
    "BackupManager",
    "create_daily_backup",
]
