from pathlib import Path
import duckdb
from openquant.storage.results_db import upsert_results, get_best_config


def test_get_best_config_returns_params_and_score(tmp_path: Path):
    dbp = tmp_path / "res.duckdb"
    # First row (older), lower score
    rows1 = [{
        "exchange": "x", "symbol": "AAA/USDT", "timeframe": "1d", "strategy": "sma",
        "params": {"fast": 10, "slow": 50}, "bars": 100,
        "metrics": {"sharpe": 0.5, "dsr": 0.4, "max_dd": 0.1, "cvar": 0.2, "n_trades": 10, "ok": True, "wfo_mts": 0.3},
    }]
    upsert_results(rows1, db_path=dbp)
    # Second row (newer), higher score
    rows2 = [{
        "exchange": "x", "symbol": "AAA/USDT", "timeframe": "1d", "strategy": "sma",
        "params": {"fast": 20, "slow": 100}, "bars": 100,
        "metrics": {"sharpe": 0.6, "dsr": 0.7, "max_dd": 0.1, "cvar": 0.2, "n_trades": 12, "ok": True, "wfo_mts": 0.8},
    }]
    upsert_results(rows2, db_path=dbp)

    params, score = get_best_config(dbp, "x", "AAA/USDT", "1d", "sma")
    assert isinstance(score, (int, float)) and score >= 0.8 - 1e-12
    assert isinstance(params, dict)
    assert params.get("fast") == 20 and params.get("slow") == 100

