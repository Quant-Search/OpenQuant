@echo off
echo ====================================
echo OpenQuant MVP Dashboard
echo ====================================
echo.

cd /d %~dp0
call .venv\Scripts\activate.bat

echo Starting dashboard at http://localhost:8501
echo Press Ctrl+C to stop
echo.

python -m streamlit run robot/dashboard.py --server.port 8501

pause

