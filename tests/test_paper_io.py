from __future__ import annotations
from pathlib import Path
from openquant.paper.state import PortfolioState
from openquant.paper.io import save_state, load_state


def test_save_load_roundtrip(tmp_path: Path):
    state = PortfolioState(cash=1234.5)
    key = ("BINANCE","BTC/USDT","1h","ema")
    state.set_position(key, 7.5)
    p = tmp_path / "state.json"
    save_state(state, p)
    loaded = load_state(p)
    assert abs(loaded.cash - 1234.5) < 1e-9
    assert abs(loaded.position(key) - 7.5) < 1e-9

