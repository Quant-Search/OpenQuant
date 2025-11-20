# Static Forex Correlation Matrix (Approximate)
# Source: Typical 1-year daily correlation values

# Correlation > 0.8 or < -0.8 is considered "Highly Correlated"

FOREX_CORRELATIONS = {
    "EURUSD": {
        "GBPUSD": 0.85,
        "AUDUSD": 0.75,
        "NZDUSD": 0.70,
        "USDCHF": -0.95,
        "USDCAD": -0.50,
        "USDJPY": 0.30,
    },
    "GBPUSD": {
        "EURUSD": 0.85,
        "AUDUSD": 0.65,
        "NZDUSD": 0.60,
        "USDCHF": -0.80,
        "USDCAD": -0.40,
        "USDJPY": 0.20,
    },
    "USDCHF": {
        "EURUSD": -0.95,
        "GBPUSD": -0.80,
        "USDJPY": 0.10,
    },
    "AUDUSD": {
        "EURUSD": 0.75,
        "GBPUSD": 0.65,
        "NZDUSD": 0.90,
        "USDCAD": -0.60,
    },
    "NZDUSD": {
        "EURUSD": 0.70,
        "GBPUSD": 0.60,
        "AUDUSD": 0.90,
    },
    "USDJPY": {
        "EURUSD": 0.30,
        "GBPUSD": 0.20,
        "EURJPY": 0.80,
    },
    "EURJPY": {
        "USDJPY": 0.80,
        "EURUSD": 0.70,
    }
}

def get_correlation(symbol_a: str, symbol_b: str) -> float:
    """Get correlation between two symbols. Returns 0.0 if unknown."""
    if symbol_a == symbol_b:
        return 1.0
    
    # Check A -> B
    corr = FOREX_CONFIG_CORR.get(symbol_a, {}).get(symbol_b)
    if corr is not None:
        return corr
        
    # Check B -> A
    corr = FOREX_CONFIG_CORR.get(symbol_b, {}).get(symbol_a)
    if corr is not None:
        return corr
        
    return 0.0

def check_portfolio_correlation(candidate_symbol: str, current_holdings: list[str], threshold: float = 0.8) -> bool:
    """
    Check if candidate symbol is highly correlated with any current holding.
    Returns True if correlated (should be rejected/penalized).
    """
    for holding in current_holdings:
        corr = get_correlation(candidate_symbol, holding)
        if abs(corr) >= threshold:
            return True
    return False

# Alias for internal use
FOREX_CONFIG_CORR = FOREX_CORRELATIONS
