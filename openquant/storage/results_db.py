"""DuckDB-backed results store for universe research outputs.

Schema (table results):
- ts TIMESTAMP
- run_id VARCHAR
- exchange VARCHAR
- symbol VARCHAR
- timeframe VARCHAR
- strategy VARCHAR
- params JSON
- sharpe DOUBLE
- dsr DOUBLE
- max_dd DOUBLE
- cvar DOUBLE
- n_trades INTEGER
- bars INTEGER
- ok BOOLEAN

Usage:
    upsert_results(rows, db_path="data/results.duckdb")
"""
from __future__ import annotations
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime, timezone
import json
import duckdb  # type: ignore


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_schema(con: duckdb.DuckDBPyConnection) -> None:  # type: ignore
    # Create with newest schema (no-op if exists)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            ts TIMESTAMP,
            run_id VARCHAR,
            exchange VARCHAR,
            symbol VARCHAR,
            timeframe VARCHAR,
            strategy VARCHAR,
            params JSON,
            sharpe DOUBLE,
            dsr DOUBLE,
            max_dd DOUBLE,
            cvar DOUBLE,
            n_trades INTEGER,
            bars INTEGER,
            ok BOOLEAN,
            wfo_mts DOUBLE
        );
        """
    )
    # Ensure new columns exist on older DBs
    try:
        cols = con.execute("PRAGMA table_info('results')").fetchall()
        names = {row[1] for row in cols}
        if 'wfo_mts' not in names:
            con.execute("ALTER TABLE results ADD COLUMN wfo_mts DOUBLE")
    except Exception:
        pass


def _refresh_best_view(con: duckdb.DuckDBPyConnection) -> None:  # type: ignore
    # Select the best config per (exchange, symbol, timeframe, strategy), preferring latest ts and higher WFO/DSR
    con.execute(
        """
        CREATE OR REPLACE VIEW best_configs AS
        WITH ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY exchange, symbol, timeframe, strategy
                       ORDER BY ts DESC,
                                COALESCE(wfo_mts, dsr) DESC,
                                dsr DESC,
                                sharpe DESC
                   ) AS rn
            FROM results
            WHERE ok
        )
        SELECT * FROM ranked WHERE rn = 1;
        """
    )



def get_best_params(db_path: str | Path,
                     exchange: str,
                     symbol: str,
                     timeframe: str,
                     strategy: str) -> Dict[str, Any] | None:
    """Fetch best known params for a specific (exchange, symbol, timeframe, strategy).
    Returns a dict of params or None if not found.
    """
    dbp = Path(db_path)
    if not dbp.exists():
        return None
    con = duckdb.connect(database=str(dbp))
    try:
        _ensure_schema(con)
        _refresh_best_view(con)
        df = con.execute(
            """
            SELECT params FROM best_configs
            WHERE exchange = ? AND symbol = ? AND timeframe = ? AND strategy = ?
            ORDER BY ts DESC LIMIT 1
            """,
            (exchange, symbol, timeframe, strategy),
        ).df()
        if df.empty:
            return None
        p = df.iloc[0]["params"]
        try:
            # DuckDB may return JSON type as str; ensure Python dict
            if isinstance(p, str):
                return json.loads(p)
            if isinstance(p, dict):
                return p
        except Exception:
            return None
        return None
    finally:
        con.close()



def get_best_config(db_path: str | Path,
                     exchange: str,
                     symbol: str,
                     timeframe: str,
                     strategy: str) -> tuple[Dict[str, Any] | None, float | None]:
    """Fetch best params and their score (coalesce wfo_mts, dsr, sharpe) for a key.
    Returns (params_dict or None, score or None).
    """
    dbp = Path(db_path)
    if not dbp.exists():
        return None, None
    con = duckdb.connect(database=str(dbp))
    try:
        _ensure_schema(con)
        _refresh_best_view(con)
        df = con.execute(
            """
            SELECT params, COALESCE(wfo_mts, dsr, sharpe) AS score
            FROM best_configs
            WHERE exchange = ? AND symbol = ? AND timeframe = ? AND strategy = ?
            ORDER BY ts DESC LIMIT 1
            """,
            (exchange, symbol, timeframe, strategy),
        ).df()
        if df.empty:
            return None, None
        p = df.iloc[0]["params"]
        s = df.iloc[0]["score"] if "score" in df.columns else None
        params_dict = None
        try:
            if isinstance(p, str):
                params_dict = json.loads(p)
            elif isinstance(p, dict):
                params_dict = p
        except Exception:
            params_dict = None
        try:
            score_val = float(s) if s is not None else None
        except Exception:
            score_val = None
        return params_dict, score_val
    finally:
        con.close()



def get_best_snapshot(db_path: str | Path) -> dict:
    """Return a snapshot mapping key tuples to a simple value tuple.
    Key: (exchange, symbol, timeframe, strategy)
    Value: (ts_iso, params_json, score)
    Score is COALESCE(wfo_mts, dsr, sharpe).
    """
    dbp = Path(db_path)
    if not dbp.exists():
        return {}
    con = duckdb.connect(database=str(dbp))
    try:
        _ensure_schema(con)
        _refresh_best_view(con)
        df = con.execute(
            """
            SELECT exchange, symbol, timeframe, strategy,
                   ts, params, COALESCE(wfo_mts, dsr, sharpe) AS score
            FROM best_configs
            """
        ).df()
        snap = {}
        if df.empty:
            return snap
        for _, row in df.iterrows():
            key = (row["exchange"], row["symbol"], row["timeframe"], row["strategy"])
            ts_iso = str(row["ts"]) if row.get("ts") is not None else ""
            params_json = row["params"]
            val = (ts_iso, params_json, float(row["score"]) if row.get("score") is not None else None)
            snap[key] = val
        return snap
    finally:
        con.close()


def diff_best_snapshots(prev: dict, curr: dict) -> tuple[set, set, set]:
    """Return (added, removed, changed) key sets comparing snapshots."""
    prev_keys, curr_keys = set(prev.keys()), set(curr.keys())
    added = curr_keys - prev_keys
    removed = prev_keys - curr_keys
    common = prev_keys & curr_keys
    changed = {k for k in common if prev.get(k) != curr.get(k)}
    return added, removed, changed


def upsert_results(rows: List[Dict[str, Any]], db_path: str | Path = "data/results.duckdb", run_id: str | None = None) -> Path:
    if not rows:
        return Path(db_path)
    db_path = Path(db_path)
    _ensure_dir(db_path)
    ts = datetime.now(timezone.utc)
    rid = run_id or ts.strftime("%Y%m%d_%H%M%S")
    # Prepare rows for insertion
    prepared = []
    for r in rows:
        m = r.get("metrics", {})
        prepared.append(
            (
                ts,
                rid,
                r.get("exchange", ""),
                r.get("symbol", ""),
                r.get("timeframe", ""),
                r.get("strategy", ""),
                json.dumps(r.get("params", {})),
                float(m.get("sharpe", 0.0)),
                float(m.get("dsr", 0.0)),
                float(m.get("max_dd", 0.0)),
                float(m.get("cvar", 0.0)),
                int(m.get("n_trades", 0)),
                int(r.get("bars", 0)),
                bool(m.get("ok", True)),
                float(m.get("wfo_mts", 0.0)),
            )
        )
    con = duckdb.connect(database=str(db_path))
    try:
        _ensure_schema(con)
        con.executemany(
            """
            INSERT INTO results (ts, run_id, exchange, symbol, timeframe, strategy, params,
                                 sharpe, dsr, max_dd, cvar, n_trades, bars, ok, wfo_mts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            prepared,
        )
        _refresh_best_view(con)
    finally:
        con.close()
    return db_path

