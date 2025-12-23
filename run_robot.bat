@echo off
REM ============================================================
REM OpenQuant MVP Trading Robot - Windows Launcher
REM ============================================================
REM 
REM Usage:
REM   run_robot.bat           - Run in paper trading mode (default)
REM   run_robot.bat live      - Run in live trading mode (requires MT5)
REM   run_robot.bat backtest  - Run backtest on historical data
REM
REM Configuration:
REM   Set your MT5 credentials in .env file or as environment variables
REM
REM ============================================================

REM Set the mode (default: paper)
SET MODE=%1
IF "%MODE%"=="" SET MODE=paper

REM Activate virtual environment if exists
IF EXIST .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)

REM Run the modular robot
python main.py --mode %MODE%

pause
