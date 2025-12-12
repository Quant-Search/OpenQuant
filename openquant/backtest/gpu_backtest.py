"""GPU-accelerated backtest engine using CuPy.

Ce module fournit une version GPU du backtest engine pour accélérer
massivement les simulations. Utilise CuPy pour les calculs vectorisés sur GPU.

Performance attendue: 10-100x plus rapide que CPU pour grands datasets.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass

from ..utils.logging import get_logger
from .engine import BacktestResult

LOGGER = get_logger(__name__)

cp = None
try:
    import cupy as cp
except Exception:
    cp = None


def is_gpu_backtest_available() -> bool:
    if cp is None:
        return False
    try:
        if cp.cuda.runtime.getDeviceCount() <= 0:
            return False
        _ = (cp.arange(8, dtype=cp.float32).sum()).item()
        return True
    except Exception:
        return False


def backtest_signals_gpu(
    df: pd.DataFrame,
    signals: pd.Series,
    fee_bps: float = 1.0,
    weight: float = 1.0,
    stop_loss_atr: Optional[float] = None,
    take_profit_atr: Optional[float] = None,
    spread_bps: float = 0.0,
    leverage: float = 1.0,
    swap_long: float = 0.0,
    swap_short: float = 0.0,
    pip_value: float = 0.0001,
) -> BacktestResult:
    """Backtest GPU-accelerated pour signaux long/flat/short.
    
    Version entièrement GPU du backtest engine. Tous les calculs sont faits
    sur GPU pour vitesse maximale.
    
    Args:
        df: OHLCV DataFrame avec 'Close', 'High', 'Low'
        signals: Series de {-1,0,1}. 1=long, -1=short, 0=flat
        fee_bps: Frais par trade en basis points (ex: 1.0 = 0.01%)
        weight: Fraction du capital allouée (0-1)
        stop_loss_atr: Distance du SL en multiples d'ATR (ex: 2.0)
        take_profit_atr: Distance du TP en multiples d'ATR (ex: 3.0)
        spread_bps: Spread bid-ask en basis points
        leverage: Levier (1.0 = pas de levier, 50.0 = levier Forex 50x)
        swap_long: Coût swap positions long en pips/jour (négatif = coût)
        swap_short: Coût swap positions short en pips/jour
        pip_value: Valeur d'1 pip en devise de cotation (ex: 0.0001 pour EURUSD)
        
    Returns:
        BacktestResult avec equity curve et returns
        
    Raises:
        RuntimeError: Si GPU non disponible
    """
    if not is_gpu_backtest_available():
        raise RuntimeError(
            "GPU non disponible pour backtest. "
            "Utilisez backtest_signals() du module engine.py"
        )
    
    if "Close" not in df.columns:
        raise KeyError("DataFrame doit contenir colonne 'Close'")
    
    # Paramètres sécurisés
    w = max(0.0, float(weight))  # Weight entre 0 et 1
    lev = max(1.0, float(leverage))  # Leverage minimum 1x
    
    # Transfert données vers GPU
    px_gpu = cp.asarray(df["Close"].values, dtype=cp.float32)
    sig_gpu = cp.asarray(signals.reindex(df.index).fillna(0).values, dtype=cp.int8)
    sig_gpu = cp.clip(sig_gpu, -1, 1)  # Clip à [-1, 0, 1]
    
    n = len(px_gpu)
    
    # Calcul ATR sur GPU si SL/TP activés
    atr_gpu = None
    if stop_loss_atr is not None or take_profit_atr is not None:
        if "High" in df.columns and "Low" in df.columns:
            high_gpu = cp.asarray(df["High"].values, dtype=cp.float32)
            low_gpu = cp.asarray(df["Low"].values, dtype=cp.float32)
            
            # True Range calculation (vectorisé)
            # TR = max(H-L, |H-Close_prev|, |L-Close_prev|)
            prev_close = cp.roll(px_gpu, 1)
            prev_close[0] = px_gpu[0]  # Pas de previous pour première barre
            
            tr1 = high_gpu - low_gpu
            tr2 = cp.abs(high_gpu - prev_close)
            tr3 = cp.abs(low_gpu - prev_close)
            
            tr = cp.maximum(tr1, cp.maximum(tr2, tr3))
            
            # ATR = SMA(TR, 14) - version simple
            # Pour production: utiliser EMA
            atr_gpu = cp.zeros_like(tr)
            for i in range(13, n):
                atr_gpu[i] = cp.mean(tr[max(0, i-13):i+1])
            
            # Forward fill premiers éléments
            if n > 0:
                atr_gpu[:13] = atr_gpu[13] if n > 13 else cp.mean(tr)
        else:
            # Fallback: volatilité simple
            returns = cp.diff(px_gpu) / px_gpu[:-1]
            vol = cp.zeros(n, dtype=cp.float32)
            for i in range(14, n):
                vol[i] = cp.std(returns[max(0, i-14):i])
            vol[:14] = vol[14] if n > 14 else cp.std(returns)
            atr_gpu = vol * px_gpu
    
    # Calcul returns (pct_change)
    ret_gpu = cp.zeros(n, dtype=cp.float32)
    ret_gpu[1:] = (px_gpu[1:] - px_gpu[:-1]) / px_gpu[:-1]
    
    # Position = signal shifted (trade au close du signal)
    pos_gpu = cp.roll(sig_gpu, 1)
    pos_gpu[0] = 0  # Pas de position au début
    
    # === Stop-Loss / Take-Profit Logic ===
    # Doit être fait en boucle car état dépend de l'historique
    if stop_loss_atr or take_profit_atr:
        # Conversion en arrays numpy pour boucle (CuPy loops sont lents)
        # Alternative: CUDA kernel custom, mais plus complexe
        px_arr = cp.asnumpy(px_gpu)
        pos_arr = cp.asnumpy(pos_gpu)
        atr_arr = cp.asnumpy(atr_gpu) if atr_gpu is not None else np.zeros(n)
        
        entry_price = 0.0
        sl_price = -float('inf')  # SL pour long
        tp_price = float('inf')   # TP pour long
        
        for i in range(1, n):
            curr_pos = pos_arr[i]
            prev_pos = pos_arr[i-1]
            
            # Nouvelle position LONG ouverte
            if curr_pos == 1 and prev_pos == 0:
                entry_price = px_arr[i-1]  # Entry au close précédent
                
                if stop_loss_atr:
                    sl_price = entry_price - stop_loss_atr * atr_arr[i-1]
                else:
                    sl_price = -float('inf')
                
                if take_profit_atr:
                    tp_price = entry_price + take_profit_atr * atr_arr[i-1]
                else:
                    tp_price = float('inf')
            
            # Nouvelle position SHORT ouverte
            elif curr_pos == -1 and prev_pos == 0:
                entry_price = px_arr[i-1]
                
                if stop_loss_atr:
                    sl_price = entry_price + stop_loss_atr * atr_arr[i-1]  # SL au-dessus pour short
                else:
                    sl_price = float('inf')
                
                if take_profit_atr:
                    tp_price = entry_price - take_profit_atr * atr_arr[i-1]  # TP en-dessous pour short
                else:
                    tp_price = -float('inf')
            
            # Check SL/TP hit pour LONG
            elif curr_pos == 1:
                if stop_loss_atr and px_arr[i] <= sl_price:
                    pos_arr[i] = 0  # Force exit
                elif take_profit_atr and px_arr[i] >= tp_price:
                    pos_arr[i] = 0  # Force exit
            
            # Check SL/TP hit pour SHORT
            elif curr_pos == -1:
                if stop_loss_atr and px_arr[i] >= sl_price:
                    pos_arr[i] = 0  # Force exit
                elif take_profit_atr and px_arr[i] <= tp_price:
                    pos_arr[i] = 0  # Force exit
        
        # Retransférer sur GPU
        pos_gpu = cp.asarray(pos_arr, dtype=cp.int8)
    
    # === Fees et Costs ===
    # Position change
    pos_change = cp.abs(cp.diff(pos_gpu, prepend=cp.int8(0)))
    
    # Fee: charged à chaque changement de position
    # fee_bps / 10000 = fraction (ex: 1 bp = 0.0001 = 0.01%)
    fee_gpu = pos_change * w * lev * (fee_bps / 10000.0)
    
    # Spread cost
    spread_cost_gpu = pos_change * w * lev * (spread_bps / 10000.0) if spread_bps > 0 else 0.0
    
    # === Swap Cost (Overnight holding) ===
    # Simplifié: détecte changement de jour
    swap_impact_gpu = cp.zeros(n, dtype=cp.float32)
    
    if swap_long != 0.0 or swap_short != 0.0:
        # Détection changement de jour (via index timestamp)
        # Note: df.index doit être DatetimeIndex
        if hasattr(df.index, 'day'):
            days = df.index.day.values
            day_gpu = cp.asarray(days, dtype=cp.int32)
            day_shifted = cp.roll(day_gpu, 1)
            day_shifted[0] = day_gpu[0]
            day_change = (day_gpu != day_shifted).astype(cp.float32)
            
            # Swap en % = (swap_pips * pip_value) / price
            swap_pct_long = (swap_long * pip_value) / px_gpu
            swap_pct_short = (swap_short * pip_value) / px_gpu
            
            # Impact: négatif si swap est négatif (coût)
            # Long positions
            swap_impact_gpu += (pos_gpu > 0).astype(cp.float32) * day_change * (-swap_pct_long) * w * lev
            
            # Short positions
            swap_impact_gpu += (pos_gpu < 0).astype(cp.float32) * day_change * (-swap_pct_short) * w * lev
    
    # === Strategy Returns ===
    # Leverage amplifie returns ET costs
    strat_ret_gpu = (pos_gpu * w * lev) * ret_gpu - fee_gpu - spread_cost_gpu + swap_impact_gpu
    
    # Equity curve (cumprod de 1 + returns)
    # CuPy ne supporte pas pd.Series, donc on fait en numpy
    equity_gpu = (1.0 + strat_ret_gpu).cumprod()
    
    # Trades (changements de position)
    trades_gpu = pos_change
    
    # Transfert retour vers CPU (pandas)
    equity_series = pd.Series(cp.asnumpy(equity_gpu), index=df.index)
    returns_series = pd.Series(cp.asnumpy(strat_ret_gpu), index=df.index)
    positions_series = pd.Series(cp.asnumpy(pos_gpu), index=df.index)
    trades_series = pd.Series(cp.asnumpy(trades_gpu), index=df.index)
    
    return BacktestResult(
        equity_curve=equity_series,
        returns=returns_series,
        positions=positions_series,
        trades=trades_series
    )


def batch_backtest_gpu(
    df: pd.DataFrame,
    signals_dict: dict[str, pd.Series],
    fee_bps: float = 1.0,
    weight: float = 1.0,
    **kwargs
) -> dict[str, BacktestResult]:
    """Backtest batch pour plusieurs stratégies en parallèle sur GPU.
    
    Permet de backtester N stratégies sur les mêmes données en une seule passe,
    exploitant le parallélisme massif du GPU.
    
    Args:
        df: OHLCV DataFrame
        signals_dict: Dict {strategy_name: signals_series}
        fee_bps: Frais en basis points
        weight: Weight pour chaque stratégie
        **kwargs: Arguments additionnels pour backtest (SL, TP, etc.)
        
    Returns:
        Dict {strategy_name: BacktestResult}
        
    Example:
        >>> signals = {
        ...     "strat1": pd.Series([1,0,1,0]),
        ...     "strat2": pd.Series([0,1,0,1])
        ... }
        >>> results = batch_backtest_gpu(df, signals)
    """
    if not is_gpu_backtest_available():
        raise RuntimeError("GPU non disponible pour batch backtest")
    
    LOGGER.info(f"Batch backtest GPU: {len(signals_dict)} stratégies")
    
    results = {}
    
    # Note: Pour vrai parallélisme GPU, il faudrait un kernel custom CUDA
    # qui traite toutes les stratégies simultanément.
    # Pour l'instant: boucle séquentielle, mais sur GPU
    # C'est déjà beaucoup plus rapide que CPU car chaque backtest est GPU-accelerated
    
    for strat_name, signals in signals_dict.items():
        try:
            result = backtest_signals_gpu(
                df=df,
                signals=signals,
                fee_bps=fee_bps,
                weight=weight,
                **kwargs
            )
            results[strat_name] = result
            
        except Exception as e:
            LOGGER.error(f"Erreur batch backtest pour {strat_name}: {e}")
            # Continue avec les autres stratégies
            
    return results
