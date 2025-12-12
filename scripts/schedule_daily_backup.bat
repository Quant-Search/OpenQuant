@echo off
REM Schedule daily database backup using Windows Task Scheduler
REM Run this script once to set up automated daily backups at 2 AM

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..
set PYTHON_SCRIPT=%SCRIPT_DIR%backup_databases.py

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found in PATH
    exit /b 1
)

REM Create the scheduled task
echo Creating scheduled task for daily database backup...
schtasks /create /tn "OpenQuant Daily Backup" /tr "python \"%PYTHON_SCRIPT%\" daily" /sc daily /st 02:00 /f

if errorlevel 1 (
    echo Error: Failed to create scheduled task
    echo Please run this script as Administrator
    exit /b 1
)

echo.
echo ✓ Daily backup task created successfully
echo Task will run every day at 2:00 AM
echo.
echo To view the task: schtasks /query /tn "OpenQuant Daily Backup"
echo To delete the task: schtasks /delete /tn "OpenQuant Daily Backup" /f
echo To run manually: python "%PYTHON_SCRIPT%" daily
echo.

pause
