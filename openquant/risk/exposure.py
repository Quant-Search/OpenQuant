from __future__ import annotations
"""Portfolio exposure capping and allocation helpers for paper-trading phase.

- propose_portfolio_weights: Greedy ranking-based allocation under caps.

All inputs are pure-Python data; no side effects. This module does not place orders.
"""
from typing import List, Dict, Any, Tuple, Optional
import math
import numpy as _np



def _score(row: Dict[str, Any]) -> Tuple[int, float, float, float]:
    """Ranking tuple: has_wfo, wfo_mts, dsr, sharpe (desc)."""
    m = (row.get("metrics") or {})
    w = m.get("wfo_mts")
    d = m.get("dsr")
    s = m.get("sharpe")
    def fin(x):
        try:
            return float(x)
        except Exception:
            return -math.inf
    return (1 if w is not None else 0, fin(w), fin(d), fin(s))


def propose_portfolio_weights(
    rows: List[Dict[str, Any]],
    *,
    max_total_weight: float = 1.0,
    max_symbol_weight: float = 0.2,
    slot_weight: float = 0.05,
    volatility_adjusted: bool = True,  # New parameter
    regime_bias: str | None = None,
) -> List[Tuple[int, float]]:
    """Propose portfolio weights (index, weight) for OK rows under exposure caps.

    - rows: list of research rows as produced by the runner
    - max_total_weight: total portfolio cap (e.g., 1.0 = 100%)
    - max_symbol_weight: per-symbol cap (e.g., 0.2 = 20%)
    - slot_weight: base weight to assign per accepted candidate (e.g., 0.05 = 5%)
    - volatility_adjusted: if True, adjust weights by inverse volatility

    Returns a list of (row_index, weight) for rows that receive non-zero weight,
    in the order they are accepted by the greedy allocator.
    """
    if not rows:
        return []
    # Filter to OK rows and order by descending score
    candidates = [(i, r) for i, r in enumerate(rows) if (r.get("metrics") or {}).get("ok")]
    candidates.sort(key=lambda t: _score(t[1]), reverse=True)
    
    # Correlation filter disabled for performance (returns not stored)
    # from .correlation import filter_correlated_candidates
    # candidates = filter_correlated_candidates(candidates, threshold=0.8)
    
    # Calculate volatility adjustments if enabled
    # Uses inverse volatility: lower volatility assets get higher weights
    # This implements risk parity-like allocation
    vol_factors = {}
    if volatility_adjusted:
        # Step 1: Collect volatility for each candidate
        for idx, row in candidates:
            symbol = row.get("symbol")
            if not symbol:
                continue

            metrics = row.get("metrics") or {}

            # Try multiple sources for volatility:
            # 1. Direct max_dd (drawdown as proxy for risk)
            # 2. Returns std if available
            # 3. Default to 1.0
            max_dd = metrics.get("max_dd")
            returns = metrics.get("returns")

            vol = None
            if max_dd is not None and max_dd > 0:
                # Use max drawdown as risk proxy (higher DD = higher risk)
                vol = float(max_dd)
            elif returns is not None and hasattr(returns, 'std'):
                vol = float(returns.std())

            if vol is not None and vol > 0:
                # Inverse volatility: lower vol/dd -> higher weight
                vol_factors[symbol] = 1.0 / max(vol, 0.01)
            else:
                vol_factors[symbol] = 1.0

        # Step 2: Normalize to median = 1.0 for stability
        if vol_factors:
            sorted_factors = sorted(vol_factors.values())
            median_factor = sorted_factors[len(sorted_factors) // 2]
            if median_factor > 0:
                for sym in vol_factors:
                    vol_factors[sym] /= median_factor
                    # Clamp to prevent extreme weights (0.5x to 2x of base)
                    vol_factors[sym] = max(0.5, min(2.0, vol_factors[sym]))

    assigned_total = 0.0
    per_symbol: Dict[str, float] = {}
    out: List[Tuple[int, float]] = []

    # Get current holdings symbols for correlation check
    # Note: holdings is not passed to this function currently. 
    # We should update the signature or assume we only check within the proposed batch for now.
    # For now, we check within the proposed batch.
    
    from .forex_correlation import check_portfolio_correlation

    def _regime_factor(m: Dict[str, Any], regime: str | None) -> float:
        if not regime:
            return 1.0
        try:
            key = {
                "bull": "bull_sharpe",
                "bear": "bear_sharpe",
                "volatile": "volatile_sharpe",
                "calm": "calm_sharpe",
            }.get(regime, None)
            if not key:
                return 1.0
            s = float(m.get(key, 0.0) or 0.0)
            f = 1.0 + (s / 5.0)
            return max(0.7, min(1.5, f))
        except Exception:
            return 1.0

    for idx, row in candidates:
        if assigned_total >= max_total_weight - 1e-12:
            break
        sym = row.get("symbol")
        if not isinstance(sym, str):
            continue

        # Check Correlation with *other* symbols in this batch (not the same symbol)
        # We allow multiple entries for the same symbol (different strategies)
        # as long as per-symbol cap is respected
        other_symbols = [s for s in per_symbol.keys() if s != sym]
        if check_portfolio_correlation(sym, other_symbols, threshold=0.8):
            continue
            
        sym_used = per_symbol.get(sym, 0.0)
        remaining_total = max(0.0, max_total_weight - assigned_total)
        remaining_sym = max(0.0, max_symbol_weight - sym_used)
        
        # Apply volatility adjustment
        base_w = slot_weight
        if volatility_adjusted and sym in vol_factors:
            base_w *= vol_factors[sym]
        base_w *= _regime_factor(row.get("metrics") or {}, regime_bias)
        
        w = min(base_w, remaining_total, remaining_sym)
        if w <= 1e-12:
            continue
        out.append((idx, w))
        assigned_total += w
        per_symbol[sym] = sym_used + w

    return out


def _project_to_simplex_with_caps(w: _np.ndarray, cap: float) -> _np.ndarray:
    """Projette le vecteur w sur le simplexe {w>=0, sum(w)=1, w_i<=cap}."""
    w = _np.clip(w, 0.0, cap)
    s = w.sum()
    if s == 0:
        # répartir uniformément sous la contrainte cap
        n = w.shape[0]
        base = min(cap, 1.0 / n)
        return _np.full(n, base)
    return w / s


def mean_variance_optimize(
    exp_returns: _np.ndarray,
    cov: _np.ndarray,
    *,
    risk_aversion: float = 1.0,
    cap_per_asset: float = 0.2,
    allow_short: bool = False,
) -> _np.ndarray:
    """
    Optimisation de Markowitz (mean-variance) paramétrique.
    Résout: argmin_w [ (risk_aversion) * w^T Σ w - exp_returns^T w ]
    s.c. sum(w)=1, w>=0, w_i<=cap_per_asset (si allow_short=False).
    Retourne les poids w.

    Équations:
    - Fonction objectif: f(w) = λ wᵀΣw - μᵀw
    - Contrainte de budget: ∑ w_i = 1
    - Contrainte de non-négativité: w_i ≥ 0 (désactivée si allow_short=True)
    - Cap par actif: w_i ≤ c
    """
    n = exp_returns.shape[0]
    Σ = _np.array(cov, dtype=float)
    μ = _np.array(exp_returns, dtype=float)
    # Solution fermée sans contraintes: w* ∝ Σ^{-1} μ
    try:
        Σ_inv = _np.linalg.pinv(Σ)
    except Exception:
        Σ_inv = _np.eye(n)
    w = Σ_inv.dot(μ)
    if w.sum() != 0:
        w = w / w.sum()
    else:
        w = _np.full(n, 1.0 / n)
    if not allow_short:
        w = _project_to_simplex_with_caps(w, cap_per_asset)
    return w


def efficient_frontier(
    exp_returns: _np.ndarray,
    cov: _np.ndarray,
    *,
    n_points: int = 20,
    cap_per_asset: float = 0.2,
    allow_short: bool = False,
) -> List[Dict[str, float]]:
    """
    Frontière efficiente (approximation) en balayant λ (aversion au risque).
    Retourne une liste dicts {"lambda": λ, "return": μᵀw, "risk": sqrt(wᵀΣw)}.
    """
    out: List[Dict[str, float]] = []
    lambdas = _np.linspace(0.1, 5.0, n_points)
    for lam in lambdas:
        w = mean_variance_optimize(exp_returns, cov, risk_aversion=lam, cap_per_asset=cap_per_asset, allow_short=allow_short)
        ret = float(_np.dot(exp_returns, w))
        risk = float(_np.sqrt(float(w.T.dot(cov).dot(w))))
        out.append({"lambda": float(lam), "return": ret, "risk": risk})
    return out


def parametric_var_cvar(portfolio_mean: float, portfolio_std: float, alpha: float = 0.95) -> Tuple[float, float]:
    """
    VaR/CVaR paramétriques sous hypothèse normale.
    VaR_α = μ + σ Φ^{-1}(α)
    CVaR_α = μ + σ * (ϕ(z) / (1-α)), où z = Φ^{-1}(α)
    """
    from math import sqrt, exp, pi
    # Inversion de la CDF normale standard (approximation)
    # Utilise l’approximation de Wichura via scipy absente -> approximation simple
    import statistics
    # Approche: pour α dans (0.5, 0.999), z ≈ statistics.NormalDist().inv_cdf(alpha)
    z = statistics.NormalDist().inv_cdf(alpha)
    phi = (1.0 / sqrt(2.0 * pi)) * exp(-(z**2) / 2.0)
    var = portfolio_mean + portfolio_std * z
    cvar = portfolio_mean + portfolio_std * (phi / (1.0 - alpha))
    return var, cvar


def build_input_from_rows(rows: List[Dict[str, Any]]) -> Tuple[_np.ndarray, _np.ndarray]:
    """
    Construit (μ, Σ) à partir des métriques de recherche.
    Fallback: μ à partir de sharpe estimé et Σ diagonale calibrée sur max_dd.
    """
    sharpe_list = []
    risk_diag = []
    for r in rows:
        m = r.get("metrics", {})
        sharpe_list.append(float(m.get("sharpe", 0.0) or 0.0))
        dd = float(m.get("max_dd", 0.0) or 0.0)
        risk_diag.append(max(dd, 1e-3))
    μ = _np.array(sharpe_list, dtype=float)
    Σ = _np.diag(_np.array(risk_diag, dtype=float))
    return μ, Σ

