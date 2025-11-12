from pathlib import Path
import duckdb
from openquant.storage.results_db import upsert_results


def test_best_configs_view_selects_most_recent_and_highest_metric(tmp_path: Path):
    dbp = tmp_path / "res.duckdb"
    # Insert two rows for same key with different metrics and timestamps (second call is later)
    rows1 = [{
        "exchange": "x", "symbol": "AAA/USDT", "timeframe": "1d", "strategy": "sma",
        "params": {"fast": 10, "slow": 50}, "bars": 100,
        "metrics": {"sharpe": 0.5, "dsr": 0.4, "max_dd": 0.1, "cvar": 0.2, "n_trades": 10, "ok": True, "wfo_mts": 0.3},
    }]
    upsert_results(rows1, db_path=dbp)
    rows2 = [{
        "exchange": "x", "symbol": "AAA/USDT", "timeframe": "1d", "strategy": "sma",
        "params": {"fast": 20, "slow": 100}, "bars": 100,
        "metrics": {"sharpe": 0.6, "dsr": 0.7, "max_dd": 0.1, "cvar": 0.2, "n_trades": 12, "ok": True, "wfo_mts": 0.8},
    }]
    upsert_results(rows2, db_path=dbp)

    con = duckdb.connect(database=str(dbp))
    try:
        df = con.execute("SELECT * FROM best_configs").df()
    finally:
        con.close()
    assert len(df) == 1
    # Expect the second row (fast=20) to be selected due to newer ts and higher wfo_mts
    assert 'params' in df.columns
    assert 'wfo_mts' in df.columns
    assert df.iloc[0]['wfo_mts'] >= 0.8 - 1e-12

