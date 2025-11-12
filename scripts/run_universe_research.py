#!/usr/bin/env python
"""Run multi-asset/multi-timeframe research for Binance USDT symbols (real network).
Usage:
  python scripts/run_universe_research.py [top_n]
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

# Ensure repo root on path
import sys as _sys, os as _os
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in _sys.path:
    _sys.path.insert(0, str(_repo_root))

from openquant.research.universe_runner import run_universe
from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _run_once(top_n: int, dd_limit, cvar_limit, daily_loss_cap, max_per_symbol: int | None, max_per_strategy_per_symbol: int | None, alert_webhook: str | None) -> None:
    workers = min(8, (os.cpu_count() or 4))
    # Strategy grids per strategy
    param_grids = {
        "sma": {"fast": (10, 20), "slow": (50, 100)},
        "ema": {"fast": (10, 20), "slow": (50, 100)},
        "rsi": {"length": (14, 21), "threshold": (50.0, 55.0)},
        "macd": {"fast": (10, 12), "slow": (20, 26), "signal": (9,)},
        "bollinger": {"length": (20,), "k": (2.0, 2.5)},
    }

    report_path = run_universe(
        exchange="binance",
        timeframes=("1d","4h","1h","30m","15m"),
        top_n=top_n,
        strategies=("sma","ema","rsi","macd","bollinger"),
        param_grids=param_grids,
        fast_list=(10,20),  # fallback for sma/ema if param_grids omitted keys
        slow_list=(50,100),
        fee_bps=2.0,
        weight=1.0,
        out_dir="reports",
        max_workers=workers,      # per-DF grid
        fetch_workers=min(4, workers),  # across symbols/timeframes
        optimize=True,
        optuna_trials=15,
        run_wfo=True,
        dd_limit=dd_limit,
        cvar_limit=cvar_limit,
        daily_loss_cap=daily_loss_cap,
        max_ok_per_symbol=max_per_symbol,
        max_ok_per_strategy_per_symbol=max_per_strategy_per_symbol,
        alert_webhook=alert_webhook,
    )
    LOGGER.info(f"Done. Report: {report_path.as_posix()}")


def main():
    import argparse, time, subprocess
    p = argparse.ArgumentParser()
    p.add_argument("top_n", nargs="?", type=int, default=50)
    p.add_argument("--repeat-mins", type=int, default=0, help="Repeat run every N minutes (0 = run once)")
    p.add_argument("--dd-limit", type=float, default=None)
    p.add_argument("--cvar-limit", type=float, default=None)
    p.add_argument("--daily-loss-cap", type=float, default=None)
    p.add_argument("--max-per-symbol", type=int, default=20, help="Max OK results to keep per symbol (default=20)")
    p.add_argument("--max-per-strategy-per-symbol", type=int, default=5, help="Max OK results per (symbol,strategy) (default=5)")
    p.add_argument("--alert-webhook", type=str, default=None, help="Optional alert webhook URL; if omitted, uses env OPENQUANT_ALERT_WEBHOOK if set")
    args = p.parse_args()

    LOGGER.info(f"Starting universe research: top_n={args.top_n}")
    launched_dashboard = False

    def launch_dashboard_once():
        nonlocal launched_dashboard
        if launched_dashboard:
            return
        try:
            subprocess.Popen([sys.executable, "-m", "streamlit", "run", "openquant/gui/dashboard.py"], close_fds=True)
            launched_dashboard = True
            LOGGER.info("Streamlit dashboard launched (http://localhost:8501)")
        except Exception as e:
            LOGGER.warning(f"Could not launch Streamlit dashboard automatically: {e}")

    if args.repeat_mins and args.repeat_mins > 0:
        LOGGER.info(f"Scheduler mode: repeating every {args.repeat_mins} minutes. Press Ctrl+C to stop.")
        while True:
            _run_once(
                args.top_n,
                args.dd_limit,
                args.cvar_limit,
                args.daily_loss_cap,
                (None if args.max_per_symbol is None else int(args.max_per_symbol)),
                (None if args.max_per_strategy_per_symbol is None else int(args.max_per_strategy_per_symbol)),
                args.alert_webhook,
            )
            launch_dashboard_once()
            time.sleep(max(60, int(args.repeat_mins) * 60))
    else:
        _run_once(
            args.top_n,
            args.dd_limit,
            args.cvar_limit,
            args.daily_loss_cap,
            (None if args.max_per_symbol is None else int(args.max_per_symbol)),
            (None if args.max_per_strategy_per_symbol is None else int(args.max_per_strategy_per_symbol)),
            args.alert_webhook,
        )
        launch_dashboard_once()


if __name__ == "__main__":
    main()

