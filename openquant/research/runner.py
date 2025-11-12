"""Simple research runner: grid search SMA crossover for one asset."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import itertools
import pandas as pd

from ..utils.config import load_config
from ..utils.logging import get_logger
from ..data.loader import DataLoader
from ..strategies.rule_based.momentum import SMACrossoverStrategy
from ..backtest.engine import backtest_signals
from ..backtest.metrics import sharpe, max_drawdown
from ..reporting.markdown import write_report

LOGGER = get_logger(__name__)


def _freq_from_timeframe(tf: str) -> str:
    return tf.lower()


def run_from_config(path: str | Path) -> Path:
    cfg = load_config(path)

    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source", "yfinance")
    symbol = data_cfg.get("symbol", "AAPL")
    timeframe = data_cfg.get("timeframe", "1d")
    start = data_cfg.get("start")
    end = data_cfg.get("end")

    bt_cfg = cfg.get("backtest", {})
    fee_bps = float(bt_cfg.get("fee_bps", 1.0))
    # Position sizing parameters (single-asset for now)
    initial_capital = float(bt_cfg.get("initial_capital", 100_000.0))  # not yet used in normalized equity
    position_weight = float(bt_cfg.get("position_weight", 1.0))
    max_leverage = float(bt_cfg.get("max_leverage", 1.0))
    max_weight = float(bt_cfg.get("max_weight", 1.0))
    # Effective weight cannot exceed caps and must be non-negative
    eff_weight = max(0.0, min(position_weight, max_leverage, max_weight))

    strat_cfg = cfg.get("strategy", {})
    fast_list = strat_cfg.get("fast", [10, 20])
    slow_list = strat_cfg.get("slow", [50, 100])

    cons = cfg.get("constraints", {})
    max_dd_allowed = float(cons.get("max_drawdown", 0.2))  # 0.2 => 20%
    cvar_alpha = float(cons.get("cvar_alpha", 0.95))
    max_cvar_allowed = float(cons.get("max_cvar", 0.2))

    dl = DataLoader()
    LOGGER.info(f"Fetching data {source} {symbol} {timeframe} {start=} {end=}")
    df = dl.get_ohlcv(source, symbol, timeframe=timeframe, start=start, end=end)
    if df.empty:
        raise RuntimeError("No data returned; check symbol/source/config")

    # Evaluate grid; collect both constrained-ok and all results so we can still report if none meet constraints
    results_all: List[Dict[str, Any]] = []
    results_ok: List[Dict[str, Any]] = []
    for fast, slow in itertools.product(fast_list, slow_list):
        if fast >= slow:
            continue
        strat = SMACrossoverStrategy(fast=fast, slow=slow)
        sig = strat.generate_signals(df)
        res = backtest_signals(df, sig, fee_bps=fee_bps, weight=eff_weight)

        freq = _freq_from_timeframe(timeframe)
        s = sharpe(res.returns, freq=freq)
        dd = -float(max_drawdown(res.equity_curve))  # convert to positive fraction
        dd = abs(dd)  # sanitize negative zero
        # Simple CVaR estimate on returns at alpha
        losses = -res.returns.dropna().values
        if losses.size:
            import numpy as np
            var = float(np.quantile(losses, cvar_alpha))
            tail = losses[losses >= var]
            cvar_val = float(tail.mean()) if tail.size else var
        else:
            cvar_val = 0.0

        # Sum trades via numpy to get a plain scalar and avoid pandas deprecation warnings
        n_trades = int(res.trades.to_numpy().sum())
        ok = (dd <= max_dd_allowed) and (cvar_val <= max_cvar_allowed)

        row = {
            "params": {"fast": fast, "slow": slow},
            "metrics": {"sharpe": s, "max_dd": dd, "cvar": cvar_val, "n_trades": n_trades, "ok": ok},
            "equity": res.equity_curve,
        }
        results_all.append(row)
        if ok:
            results_ok.append(row)

    pool = results_ok if results_ok else results_all
    if not results_ok:
        LOGGER.warning("No strategies satisfied constraints; proceeding with unconstrained ranking for reporting.")

    # Pick top by Sharpe
    results_sorted = sorted(pool, key=lambda x: x["metrics"]["sharpe"], reverse=True)
    top = results_sorted[0]

    report_path = write_report(results_sorted, cfg, out_dir=cfg.get("report", {}).get("out_dir", "reports"), top_equity=top["equity"]) 
    LOGGER.info(f"Report written: {report_path}")
    return report_path


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "configs/aapl_sma.yaml"
    run_from_config(path)

