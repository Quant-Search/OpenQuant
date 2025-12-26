@echo off
title OpenQuant Trading Robot
color 0A

echo.
echo  ========================================
echo      OpenQuant Trading Robot
echo  ========================================
echo.
echo  Starting the trading dashboard...
echo  Please wait...
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo  [ERROR] Python virtual environment not found!
    echo  Please run setup first.
    pause
    exit /b 1
)

echo  Opening dashboard at http://localhost:8501
echo.
echo  Press Ctrl+C to stop the robot
echo.

start "" "http://localhost:8501"
.venv\Scripts\streamlit.exe run robot\dashboard.py --server.port 8501

pause

