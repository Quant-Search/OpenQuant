from __future__ import annotations
from openquant.paper.mt5_bridge import is_available, init, shutdown


def test_mt5_bridge_smoke():
    # Should not raise and should return a boolean availability flag
    avail = is_available()
    assert isinstance(avail, bool)
    # init() should return False if not available, or bool if available; never raise
    ok = init()
    assert isinstance(ok, bool)
    shutdown()

