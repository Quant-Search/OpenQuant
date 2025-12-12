#!/bin/bash
# Schedule daily database backup using cron
# Run this script once to set up automated daily backups at 2 AM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$SCRIPT_DIR/backup_databases.py"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found in PATH"
    exit 1
fi

# Verify the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Backup script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Create cron entry
CRON_JOB="0 2 * * * cd $PROJECT_ROOT && python3 $PYTHON_SCRIPT daily >> $PROJECT_ROOT/data/backup.log 2>&1"

echo "Setting up daily backup cron job..."
echo "This will run every day at 2:00 AM"
echo ""
echo "Cron entry:"
echo "$CRON_JOB"
echo ""

# Check if cron entry already exists
(crontab -l 2>/dev/null | grep -v "$PYTHON_SCRIPT daily") | crontab -

# Add new cron entry
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

if [ $? -eq 0 ]; then
    echo "✓ Daily backup cron job created successfully"
    echo ""
    echo "To view all cron jobs: crontab -l"
    echo "To remove this job: crontab -e (then delete the line)"
    echo "To run manually: python3 $PYTHON_SCRIPT daily"
    echo "Logs will be written to: $PROJECT_ROOT/data/backup.log"
else
    echo "✗ Failed to create cron job"
    exit 1
fi
