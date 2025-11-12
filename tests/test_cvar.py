import numpy as np
import pandas as pd
from openquant.backtest.metrics import cvar


def test_cvar_monotone_tail():
    # Loss distribution: small gains, occasional large losses
    r = pd.Series([0.001]*100 + [-0.05]*5 + [-0.10]*2)
    cv = cvar(r, alpha=0.95)
    # CVaR should be >= VaR tail average and positive
    assert cv > 0
    # Heuristic: CVaR should be at least as large as the 95th percentile loss
    losses = -r.values
    var = float(np.quantile(losses, 0.95))
    assert cv >= var - 1e-12

