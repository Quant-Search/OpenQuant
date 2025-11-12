import yaml
import pandas as pd
from pathlib import Path

from openquant.research.runner import run_from_config
import openquant.data.loader as loader_mod


def _synthetic_ohlcv(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(100.0 + (pd.Series(range(n)) * 0.1), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.001
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.999
    vol = pd.Series(1_000, index=idx)
    return pd.DataFrame({"Open": open_.values, "High": high.values, "Low": low.values, "Close": close.values, "Volume": vol.values}, index=idx)


def test_research_offline_report(tmp_path, monkeypatch):
    # Monkeypatch DataLoader.get_ohlcv to avoid network
    monkeypatch.setattr(loader_mod.DataLoader, "get_ohlcv", lambda self, *a, **k: _synthetic_ohlcv(300))

    # Prepare a temporary config file
    cfg = {
        "name": "offline_sma",
        "data": {
            "source": "yfinance",  # ignored due to monkeypatch
            "symbol": "FAKE",
            "timeframe": "1d",
        },
        "strategy": {"fast": [10, 20], "slow": [50, 100]},
        "backtest": {"fee_bps": 0.5},
        "constraints": {"max_drawdown": 0.5, "cvar_alpha": 0.95, "max_cvar": 0.5},
        "report": {"out_dir": str(tmp_path / "reports")},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    # Run research
    report_path = run_from_config(str(cfg_path))

    # Verify outputs
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Top Results" in content
    assert "| ok |" in content  # report includes constraint satisfaction flag

