"""Transaction cost model utilities and presets.

This module provides convenient presets and helper functions for the enhanced
transaction cost models in the backtest engine, including:
- Time-of-day spread modeling
- Volume-dependent slippage
- Market impact for large orders
- Funding rates for perpetual swaps
"""
from typing import Dict, Optional
import pandas as pd


# Preset time-of-day multipliers for different markets

# Typical FX major pairs (EUR/USD, GBP/USD, USD/JPY) spread patterns
TOD_MULTIPLIERS_FX_MAJOR = {
    0: 1.5,   # Asian session - lower liquidity
    1: 1.6,
    2: 1.6,
    3: 1.5,
    4: 1.4,
    5: 1.3,
    6: 1.2,
    7: 1.0,   # London open - high liquidity
    8: 0.85,  # Peak London
    9: 0.8,
    10: 0.8,
    11: 0.8,
    12: 0.8,
    13: 0.85, # NY open
    14: 0.8,  # London/NY overlap - highest liquidity
    15: 0.8,
    16: 0.85,
    17: 1.0,  # London close
    18: 1.1,
    19: 1.2,
    20: 1.3,
    21: 1.4,
    22: 1.5,
    23: 1.5,
}

# Crypto major pairs (BTC/USD, ETH/USD) - 24/7 but with patterns
TOD_MULTIPLIERS_CRYPTO_MAJOR = {
    0: 1.2,   # Lower activity
    1: 1.3,
    2: 1.3,
    3: 1.2,
    4: 1.2,
    5: 1.1,
    6: 1.0,   # Asian morning
    7: 0.95,
    8: 0.9,   # European morning
    9: 0.85,
    10: 0.85,
    11: 0.85,
    12: 0.85,
    13: 0.85, # US morning - high activity
    14: 0.8,
    15: 0.8,
    16: 0.85,
    17: 0.9,
    18: 0.95, # After hours
    19: 1.0,
    20: 1.1,
    21: 1.1,
    22: 1.2,
    23: 1.2,
}

# Crypto altcoins - generally wider spreads with more variation
TOD_MULTIPLIERS_CRYPTO_ALTCOIN = {
    0: 1.5,
    1: 1.6,
    2: 1.6,
    3: 1.5,
    4: 1.5,
    5: 1.4,
    6: 1.3,
    7: 1.2,
    8: 1.1,
    9: 1.0,
    10: 0.95,
    11: 0.95,
    12: 0.95,
    13: 0.9,  # Peak activity
    14: 0.85,
    15: 0.85,
    16: 0.9,
    17: 1.0,
    18: 1.1,
    19: 1.2,
    20: 1.3,
    21: 1.4,
    22: 1.5,
    23: 1.5,
}

# No time-of-day variation - constant spread
TOD_MULTIPLIERS_FLAT = {
    hour: 1.0 for hour in range(24)
}


# Market-specific cost presets

class CostPreset:
    """Preset configuration for transaction costs by market type."""
    
    @staticmethod
    def fx_major_ecn() -> Dict:
        """FX major pairs on ECN broker (tight spreads, low fees)."""
        return {
            "fee_bps": 0.2,
            "spread_bps": 1.0,
            "slippage_bps": 0.5,
            "use_tod_spread": True,
            "tod_multipliers": TOD_MULTIPLIERS_FX_MAJOR,
            "use_volume_slippage": False,
            "use_market_impact": False,
        }
    
    @staticmethod
    def fx_major_retail() -> Dict:
        """FX major pairs on retail broker (wider spreads)."""
        return {
            "fee_bps": 0.0,  # Spread covers costs
            "spread_bps": 2.5,
            "slippage_bps": 1.0,
            "use_tod_spread": True,
            "tod_multipliers": TOD_MULTIPLIERS_FX_MAJOR,
            "use_volume_slippage": False,
            "use_market_impact": False,
        }
    
    @staticmethod
    def crypto_spot_tier1() -> Dict:
        """Crypto spot on tier-1 exchange (Binance, Coinbase, Kraken)."""
        return {
            "fee_bps": 4.0,  # 0.04% maker, assuming taker
            "spread_bps": 2.0,
            "slippage_bps": 1.0,
            "use_tod_spread": True,
            "tod_multipliers": TOD_MULTIPLIERS_CRYPTO_MAJOR,
            "use_volume_slippage": True,
            "volume_impact_coeff": 0.15,
            "use_market_impact": True,
            "participation_rate": 0.05,
            "impact_exponent": 0.6,
        }
    
    @staticmethod
    def crypto_spot_tier2() -> Dict:
        """Crypto spot on smaller exchange (wider spreads, more slippage)."""
        return {
            "fee_bps": 8.0,
            "spread_bps": 5.0,
            "slippage_bps": 2.0,
            "use_tod_spread": True,
            "tod_multipliers": TOD_MULTIPLIERS_CRYPTO_MAJOR,
            "use_volume_slippage": True,
            "volume_impact_coeff": 0.25,
            "use_market_impact": True,
            "participation_rate": 0.03,
            "impact_exponent": 0.65,
        }
    
    @staticmethod
    def crypto_perp_tier1(avg_funding_rate_bps: float = 1.0) -> Dict:
        """Crypto perpetual swaps on tier-1 exchange."""
        return {
            "fee_bps": 2.5,  # Maker rebate possible, using taker
            "spread_bps": 1.5,
            "slippage_bps": 0.5,
            "use_tod_spread": True,
            "tod_multipliers": TOD_MULTIPLIERS_CRYPTO_MAJOR,
            "use_volume_slippage": True,
            "volume_impact_coeff": 0.1,
            "use_market_impact": True,
            "participation_rate": 0.08,
            "impact_exponent": 0.55,
            "use_dynamic_funding": True,
            "funding_rate_bps": avg_funding_rate_bps,
            "funding_interval_hours": 8,
            "premium_sensitivity": 0.15,
        }
    
    @staticmethod
    def altcoin_spot() -> Dict:
        """Altcoin spot trading (higher costs)."""
        return {
            "fee_bps": 10.0,
            "spread_bps": 10.0,
            "slippage_bps": 5.0,
            "use_tod_spread": True,
            "tod_multipliers": TOD_MULTIPLIERS_CRYPTO_ALTCOIN,
            "use_volume_slippage": True,
            "volume_impact_coeff": 0.3,
            "use_market_impact": True,
            "participation_rate": 0.02,
            "impact_exponent": 0.7,
        }
    
    @staticmethod
    def paper_trading_conservative() -> Dict:
        """Conservative cost assumptions for paper trading."""
        return {
            "fee_bps": 5.0,
            "spread_bps": 3.0,
            "slippage_bps": 2.0,
            "use_tod_spread": True,
            "use_volume_slippage": True,
            "volume_impact_coeff": 0.2,
            "use_market_impact": True,
            "participation_rate": 0.05,
            "impact_exponent": 0.6,
        }
    
    @staticmethod
    def paper_trading_optimistic() -> Dict:
        """Optimistic cost assumptions for paper trading."""
        return {
            "fee_bps": 2.0,
            "spread_bps": 1.5,
            "slippage_bps": 0.5,
            "use_tod_spread": False,
            "use_volume_slippage": False,
            "use_market_impact": False,
        }


def estimate_total_cost(
    preset: Dict,
    avg_position_changes_per_day: float = 2.0,
    avg_holding_period_days: float = 1.0,
    funding_rate_bps_if_perp: float = 1.0,
) -> Dict[str, float]:
    """Estimate total transaction costs for a trading strategy.
    
    Args:
        preset: Cost preset dictionary
        avg_position_changes_per_day: Average number of trades per day
        avg_holding_period_days: Average holding period
        funding_rate_bps_if_perp: Funding rate if trading perpetuals
    
    Returns:
        Dictionary with estimated costs breakdown
    """
    # Round-trip costs per trade
    fee_roundtrip = preset.get("fee_bps", 0) * 2  # Entry + exit
    spread_roundtrip = preset.get("spread_bps", 0) * 2
    slippage_roundtrip = preset.get("slippage_bps", 0) * 2
    
    # Estimate market impact (assuming sqrt model)
    impact_per_trade = 0.0
    if preset.get("use_market_impact", False):
        # Rough estimate: typical impact for 5% participation
        impact_per_trade = 5.0  # bps
    
    # Per-trade cost
    per_trade_cost = fee_roundtrip + spread_roundtrip + slippage_roundtrip + impact_per_trade
    
    # Daily trading cost
    daily_cost = per_trade_cost * avg_position_changes_per_day
    
    # Holding costs (funding for perps)
    holding_cost_per_day = 0.0
    if preset.get("use_dynamic_funding", False):
        # Funding every 8 hours = 3x per day
        holding_cost_per_day = funding_rate_bps_if_perp * 3
    
    # Total holding cost
    total_holding_cost = holding_cost_per_day * avg_holding_period_days
    
    # Total per trade cycle (entry + hold + exit)
    total_cost = per_trade_cost + total_holding_cost
    
    return {
        "per_trade_cost_bps": per_trade_cost,
        "daily_trading_cost_bps": daily_cost,
        "holding_cost_per_day_bps": holding_cost_per_day,
        "total_holding_cost_bps": total_holding_cost,
        "total_cycle_cost_bps": total_cost,
        "total_cycle_cost_pct": total_cost / 100.0,
        "breakeven_return_bps": total_cost,
    }


def compare_presets(*preset_names: str) -> pd.DataFrame:
    """Compare multiple cost presets side-by-side.
    
    Args:
        *preset_names: Names of presets to compare (e.g., 'fx_major_ecn', 'crypto_spot_tier1')
    
    Returns:
        DataFrame comparing the presets
    """
    presets = {}
    for name in preset_names:
        if hasattr(CostPreset, name):
            presets[name] = getattr(CostPreset, name)()
        else:
            raise ValueError(f"Unknown preset: {name}")
    
    # Get all unique keys
    all_keys = set()
    for preset in presets.values():
        all_keys.update(preset.keys())
    
    # Build comparison dataframe
    comparison = {}
    for name, preset in presets.items():
        comparison[name] = {k: preset.get(k, None) for k in sorted(all_keys)}
    
    return pd.DataFrame(comparison).T
