"""Script pour placer un ordre test sur MT5."""
import os
import sys
from pathlib import Path
import time

# Add repository root to Python path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package n'est pas installé")
    sys.exit(1)

# Informations de connexion
terminal_path = "C:/Program Files/MetaTrader/terminal64.exe"
server = "MetaQuotes-Demo"
login = 10008295042
password = "2aR@VaBb"

print("1. Initialisation de MT5...")
if not mt5.initialize(path=terminal_path):
    print(f"ERROR: Échec de l'initialisation MT5. Erreur: {mt5.last_error()}")
    mt5.shutdown()
    sys.exit(1)

print("2. Tentative de connexion...")
if not mt5.login(login=login, password=password, server=server):
    print(f"ERROR: Échec de la connexion. Erreur: {mt5.last_error()}")
    mt5.shutdown()
    sys.exit(1)

# Symbole à trader
symbol = "EURUSD"

# Obtenir les informations du symbole
symbol_info = mt5.symbol_info(symbol)
if symbol_info is None:
    print(f"ERROR: {symbol} non trouvé")
    mt5.shutdown()
    sys.exit(1)

if not symbol_info.visible:
    print(f"Activation de {symbol}")
    if not mt5.symbol_select(symbol, True):
        print(f"ERROR: {symbol} activation échouée")
        mt5.shutdown()
        sys.exit(1)

# Obtenir le dernier prix
tick = mt5.symbol_info_tick(symbol)
if tick is None:
    print(f"ERROR: Impossible d'obtenir le prix de {symbol}")
    mt5.shutdown()
    sys.exit(1)

# Préparer l'ordre d'achat
lot = 0.01
point = symbol_info.point
price = tick.ask
deviation = 20

request = {
    "action": mt5.TRADE_ACTION_DEAL,
    "symbol": symbol,
    "volume": lot,
    "type": mt5.ORDER_TYPE_BUY,
    "price": price,
    "deviation": deviation,
    "magic": 234000,
    "comment": "ordre test python",
    "type_time": mt5.ORDER_TIME_GTC,
    "type_filling": mt5.ORDER_FILLING_FOK,
}

print(f"\nPlacement d'un ordre d'achat pour {symbol}:")
print(f"Prix: {price}, Volume: {lot} lots")

# Envoyer l'ordre
result = mt5.order_send(request)
if result.retcode != mt5.TRADE_RETCODE_DONE:
    print(f"ERROR: Ordre échoué. Code d'erreur: {result.retcode}")
    print(f"Message: {result.comment}")
else:
    print("\nOrdre exécuté avec succès!")
    print(f"Ticket: {result.order}")
    print(f"Volume: {result.volume}")
    print(f"Prix: {result.price}")

time.sleep(1)  # Attendre que la position soit mise à jour

print("\nPositions ouvertes après l'ordre:")
positions = mt5.positions_get()
if positions is None:
    print("Aucune position ouverte")
else:
    for position in positions:
        print(f"Symbole: {position.symbol}, Volume: {position.volume}, Type: {'Achat' if position.type == 0 else 'Vente'}")

mt5.shutdown()