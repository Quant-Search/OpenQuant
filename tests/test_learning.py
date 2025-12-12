"""Test Learning Mechanisms.

Verifies TradeMemory persistence and OnlineLearner updates.
"""
import pytest
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime
from openquant.storage.trade_memory import TradeMemory
from openquant.strategies.online_learner import OnlineLearner

TEST_DB = "data/test_trades.duckdb"
TEST_MODEL = "data/test_model.joblib"

@pytest.fixture
def clean_files():
    # Setup
    if Path(TEST_DB).exists(): Path(TEST_DB).unlink()
    if Path(TEST_MODEL).exists(): Path(TEST_MODEL).unlink()
    yield
    # Teardown
    if Path(TEST_DB).exists(): Path(TEST_DB).unlink()
    if Path(TEST_MODEL).exists(): Path(TEST_MODEL).unlink()

def test_trade_memory(clean_files):
    memory = TradeMemory(db_path=TEST_DB)
    
    trade = {
        "id": "123",
        "symbol": "BTC/USDT",
        "side": "LONG",
        "entry_time": datetime.now(),
        "exit_time": datetime.now(),
        "entry_price": 50000.0,
        "exit_price": 51000.0,
        "quantity": 0.1,
        "pnl": 100.0,
        "pnl_pct": 0.02,
        "exit_reason": "tp",
        "market_regime": "trending",
        "features_json": {"rsi": 70, "vol": 0.01}
    }
    
    memory.save_trade(trade)
    
    df = memory.load_recent_trades()
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "BTC/USDT"
    assert df.iloc[0]["pnl"] == 100.0
    
    memory.close()

def test_online_learner(clean_files):
    learner = OnlineLearner(model_path=TEST_MODEL)
    
    # Fake features: [rsi, vol]
    features_win = np.array([30, 0.01]) # Low RSI -> Buy -> Win
    features_loss = np.array([80, 0.05]) # High RSI -> Buy -> Loss
    
    # Initial prediction (should be 0.5 or random)
    prob_init = learner.predict_proba(features_win)
    
    # Train: Win on low RSI
    for _ in range(10):
        learner.update(features_win, 1)
        
    # Train: Loss on high RSI
    for _ in range(10):
        learner.update(features_loss, 0)
        
    # Check predictions
    prob_win = learner.predict_proba(features_win)
    prob_loss = learner.predict_proba(features_loss)
    
    assert prob_win > 0.5
    assert prob_loss < 0.5
    assert learner.is_fitted

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
