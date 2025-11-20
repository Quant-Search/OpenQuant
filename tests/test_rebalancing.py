"""
Test for Portfolio Rebalancing.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from openquant.risk.rebalancing import Rebalancer

def test_rebalancing():
    print("\n--- Testing Portfolio Rebalancing ---")
    
    # Setup
    # Total Equity = 10,000
    # Target: 50% BTC, 50% ETH
    # Current: BTC doubled, ETH halved -> Drift!
    
    cash = 0.0
    prices = {"BTC": 50000.0, "ETH": 3000.0}
    
    # Scenario: BTC pumped, ETH dumped
    # BTC: 0.15 units * 50k = 7500
    # ETH: 0.833 units * 3k = 2500
    # Total = 10000
    # Weights: BTC 75%, ETH 25%
    # Target: 50% / 50%
    
    holdings = {"BTC": 0.15, "ETH": 0.83333333}
    targets = {"BTC": 0.5, "ETH": 0.5}
    
    rebalancer = Rebalancer(drift_threshold=0.05) # 5% tolerance
    
    # 1. Check Drift
    drifted = rebalancer.check_drift(holdings, prices, targets, cash)
    print(f"Drifted Symbols: {drifted}")
    assert "BTC" in drifted
    assert "ETH" in drifted
    
    # 2. Generate Orders
    orders = rebalancer.generate_rebalancing_orders(holdings, prices, targets, cash)
    print("Rebalancing Orders:")
    for o in orders:
        print(f"  {o['symbol']}: {o['delta_units']:.4f} units (Val: {o['delta_units']*o['price']:.2f})")
        
    # Verify BTC Sell
    btc_order = next(o for o in orders if o['symbol'] == "BTC")
    # Target BTC val = 5000. Current = 7500. Sell 2500.
    # Units = -2500 / 50000 = -0.05
    assert abs(btc_order['delta_units'] - (-0.05)) < 0.001
    
    # Verify ETH Buy
    eth_order = next(o for o in orders if o['symbol'] == "ETH")
    # Target ETH val = 5000. Current = 2500. Buy 2500.
    # Units = 2500 / 3000 = 0.8333
    assert eth_order['delta_units'] > 0
    
    print("\n✅ Rebalancing Test Passed!")

if __name__ == "__main__":
    test_rebalancing()
