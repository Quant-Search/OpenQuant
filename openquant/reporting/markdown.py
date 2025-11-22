"""Markdown report writer for research runs."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import yaml
import pandas as pd


def write_report(
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    out_dir: str | Path = "reports",
    top_k: int = 10,
    top_equity: Optional[pd.Series] = None,
) -> Path:
    """Write a Markdown report and return its path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = out / f"report_{ts}.md"

    # Persist top equity curve if provided
    equity_path: Optional[Path] = None
    if top_equity is not None and not top_equity.empty:
        equity_path = out / f"equity_{ts}.csv"
        top_equity.to_csv(equity_path, header=["equity"])  # index as UTC timestamps

    # Sort by objective (Sharpe) descending
    ranked = sorted(results, key=lambda x: x.get("metrics", {}).get("sharpe", 0.0), reverse=True)[:top_k]

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# OpenQuant Research Report\\n\\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}Z\n\n")
        f.write("## Config\\n")
        f.write("```yaml\n" + yaml.safe_dump(config, sort_keys=False) + "\n```\n\n")

        f.write("## Top Results (by Sharpe)\\n")
        f.write("| rank | params | sharpe | max_dd | cvar | n_trades | ok |\n")
        f.write("|---:|---|---:|---:|---:|---:|:---:|\n")
        for i, r in enumerate(ranked, 1):
            m = r.get("metrics", {})
            ok_flag = "Y" if m.get("ok", True) else "N"
            f.write(
                f"| {i} | {r.get('params')} | {m.get('sharpe',0):.3f} | {m.get('max_dd',0):.3f} | {m.get('cvar',0):.3f} | {m.get('n_trades',0)} | {ok_flag} |\n"
            )

        if equity_path is not None:
            f.write("\n## Equity Curve (CSV)\\n")
            f.write(f"Saved to: {equity_path.as_posix()}\\n")

    return report_path




def write_universe_summary(
    rows: List[Dict[str, Any]],
    out_dir: str | Path = "reports",
    top_k: int = 100,
) -> Path:
    """Write a Markdown summary for multi-asset/multi-timeframe research.
    Each row should include: symbol, timeframe, params, metrics{sharpe,max_dd,cvar,n_trades,ok}.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = out / f"universe_{ts}.md"

    ranked = sorted(rows, key=lambda x: x.get("metrics", {}).get("sharpe", 0.0), reverse=True)[:top_k]

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"# OpenQuant Universe Research Summary\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}Z\n\n")
        f.write("## Top Results (by Sharpe)\n")
        f.write("| rank | strategy | symbol | timeframe | params | sharpe | dsr | max_dd | cvar | n_trades | ok |\n")
        f.write("|---:|:---:|---|:---:|---|---:|---:|---:|---:|---:|:---:|\n")
        for i, r in enumerate(ranked, 1):
            m = r.get("metrics", {})
            ok_flag = "Y" if m.get("ok", True) else "N"
            f.write(
                f"| {i} | {r.get('strategy','sma')} | {r.get('symbol')} | {r.get('timeframe')} | {r.get('params')} | "
                f"{m.get('sharpe',0):.3f} | {m.get('dsr',0):.3f} | {m.get('max_dd',0):.3f} | {m.get('cvar',0):.3f} | {m.get('n_trades',0)} | {ok_flag} |\n"
            )

    return report_path
