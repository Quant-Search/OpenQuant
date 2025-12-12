@echo off
REM Install dependencies, skipping pandas_ta (incompatible with Python 3.14)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install pandas numpy pyyaml python-dotenv pytest
.venv\Scripts\python.exe -m pip install yfinance ccxt matplotlib duckdb optuna
.venv\Scripts\python.exe -m pip install streamlit pyinstaller plotly psutil
.venv\Scripts\python.exe -m pip install statsmodels arch filterpy scipy hurst
.venv\Scripts\python.exe -m pip install alpaca-py scikit-learn
echo Installation complete!
