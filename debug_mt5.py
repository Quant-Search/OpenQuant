import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

try:
    login = int(os.getenv("MT5_LOGIN"))
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_TERMINAL_PATH")
except Exception as e:
    print(f"Error reading env: {e}")
    exit(1)

print(f"Testing MT5 connection with credentials in initialize()...")
print(f"Path: {path}")
print(f"Login: {login}")
print(f"Server: {server}")

if not mt5.initialize(path=path, login=login, password=password, server=server):
    print(f"Initialize(path, creds) failed, error code = {mt5.last_error()}")
    # Try without path but with creds
    if not mt5.initialize(login=login, password=password, server=server):
        print(f"Initialize(creds) failed, error code = {mt5.last_error()}")
        quit()
    else:
        print("Initialize(creds) succeeded")
else:
    print("Initialize(path, creds) succeeded")

# Double check login status
print(f"Connected to account #{login}")
info = mt5.account_info()
if info:
    print(f"Balance: {info.balance} {info.currency}")
else:
    print("Could not get account info")

mt5.shutdown()
