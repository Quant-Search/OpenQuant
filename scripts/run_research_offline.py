"""Offline smoke test for the research runner using synthetic data.
This avoids network calls and validates end-to-end report generation.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

# Ensure repository root is on sys.path when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openquant.research.runner import run_from_config
import openquant.data.loader as loader_mod


def synthetic_ohlcv(n: int = 200) -> pd.DataFrame:
    # UTC index, 1-day bars
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    # Create a gentle uptrend with small noise
    close = pd.Series(100.0).repeat(n).reset_index(drop=True)
    close = pd.Series(100.0 + (pd.Series(range(n)) * 0.1)).astype(float)
    # Build OHLCV from Close
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.001
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.999
    vol = pd.Series(1_000, index=range(n))
    df = pd.DataFrame({"Open": open_.values, "High": high.values, "Low": low.values, "Close": close.values, "Volume": vol.values}, index=idx)
    return df


def main():
    # Monkeypatch DataLoader.get_ohlcv to return synthetic data regardless of source
    original = loader_mod.DataLoader.get_ohlcv

    def _patched(self, source: str, symbol: str, timeframe: str = "1d", **kwargs):
        return synthetic_ohlcv(300)

    loader_mod.DataLoader.get_ohlcv = _patched  # type: ignore
    try:
        cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/aapl_sma.yaml"
        p = run_from_config(cfg)
        print(p)
        # Persist path for CI/agent visibility
        out_flag = Path("reports") / "last_offline_report.txt"
        out_flag.parent.mkdir(parents=True, exist_ok=True)
        out_flag.write_text(str(p), encoding="utf-8")
    finally:
        # restore
        loader_mod.DataLoader.get_ohlcv = original  # type: ignore


if __name__ == "__main__":
    main()

