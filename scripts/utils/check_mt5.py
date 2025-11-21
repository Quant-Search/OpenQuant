"""Script de vérification de la connexion MT5."""
import os
import sys
from pathlib import Path

# Add repository root to Python path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package n'est pas installé. Installer avec:")
    print("pip install MetaTrader5")
    sys.exit(1)

# Informations de connexion
terminal_path = "C:/Program Files/MetaTrader/terminal64.exe"
server = "MetaQuotes-Demo"
login = 10008295042
password = "2aR@VaBb"

print("1. Initialisation de MT5...")
if not mt5.initialize(path=terminal_path):
    print(f"ERROR: Échec de l'initialisation MT5. Erreur: {mt5.last_error()}")
    print(f"Vérifiez que le chemin du terminal est correct: {terminal_path}")
    mt5.shutdown()
    sys.exit(1)

print("2. Tentative de connexion...")
if not mt5.login(login=login, password=password, server=server):
    print(f"ERROR: Échec de la connexion. Erreur: {mt5.last_error()}")
    print("Vérifiez vos identifiants de connexion")
    mt5.shutdown()
    sys.exit(1)

# Vérifier l'état du compte
account_info = mt5.account_info()
if account_info is None:
    print("ERROR: Impossible d'obtenir les informations du compte")
    mt5.shutdown()
    sys.exit(1)

print("\nInformations du compte:")
print(f"Balance: {account_info.balance}")
print(f"Equity: {account_info.equity}")
print(f"Profit: {account_info.profit}")
print(f"Marge libre: {account_info.margin_free}")

print("\nPositions ouvertes:")
positions = mt5.positions_get()
if positions is None:
    print("Aucune position ouverte")
else:
    for position in positions:
        print(f"Symbole: {position.symbol}, Volume: {position.volume}, Type: {'Achat' if position.type == 0 else 'Vente'}")

mt5.shutdown()