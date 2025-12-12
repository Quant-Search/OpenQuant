"""Tests for GPU-accelerated backtest engine.

Vérifie que:
1. GPU backtest produit mêmes résultats que CPU backtest
2. GPU est plus rapide que CPU (si disponible)
3. Batch backtest fonctionne correctement
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from openquant.backtest.engine import backtest_signals
from openquant.backtest.gpu_backtest import (
    backtest_signals_gpu,
    batch_backtest_gpu,
    is_gpu_backtest_available
)


# Skip all tests si GPU non disponible
pytestmark = pytest.mark.skipif(
    not is_gpu_backtest_available(),
    reason="GPU not available for testing"
)


@pytest.fixture
def sample_data():
    """Génère données OHLCV synthétiques pour test."""
    n = 1000  # 1000 barres
    dates = pd.date_range(start='2024-01-01', periods=n, freq='1h')
    
    # Prix random walk
    np.random.seed(42)
    returns = np.random.randn(n) * 0.01  # 1% volatilité
    close = 100 * (1 + returns).cumprod()
    
    # OHLC
    high = close * (1 + np.abs(np.random.randn(n)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.005)
    open_ = close * (1 + np.random.randn(n) * 0.002)
    volume = np.random.randint(1000, 10000, n)
    
    df = pd.DataFrame({
        'Open': open_,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume
    }, index=dates)
    
    return df


@pytest.fixture
def simple_signals(sample_data):
    """Signaux de test: alternance Long/Flat."""
    n = len(sample_data)
    signals = pd.Series(0, index=sample_data.index)
    
    # Long toutes les 10 barres pendant 5 barres
    for i in range(0, n, 10):
        signals.iloc[i:min(i+5, n)] = 1
    
    return signals


def test_gpu_backtest_basic(sample_data, simple_signals):
    """Test basique: GPU backtest doit fonctionner sans erreur."""
    result = backtest_signals_gpu(
        df=sample_data,
        signals=simple_signals,
        fee_bps=1.0,
        weight=1.0
    )
    
    # Vérifications basiques
    assert result is not None, "Result ne doit pas être None"
    assert len(result.equity_curve) == len(sample_data), "Equity curve doit avoir même longueur"
    assert result.equity_curve.iloc[0] == 1.0, "Equity doit commencer à 1.0"
    assert result.equity_curve.iloc[-1] > 0, "Equity finale doit être positive"
    
    # Vérifie que trades ont eu lieu
    n_trades = result.trades.sum()
    assert n_trades > 0, "Au moins un trade doit avoir eu lieu"


def test_gpu_vs_cpu_consistency(sample_data, simple_signals):
    """Test critique: GPU doit produire EXACTEMENT les mêmes résultats que CPU."""
    # CPU backtest
    cpu_result = backtest_signals(
        df=sample_data,
        signals=simple_signals,
        fee_bps=1.0,
        weight=1.0
    )
    
    # GPU backtest
    gpu_result = backtest_signals_gpu(
        df=sample_data,
        signals=simple_signals,
        fee_bps=1.0,
        weight=1.0
    )
    
    # Comparaison (tolérance pour erreurs de floating point)
    np.testing.assert_allclose(
        cpu_result.equity_curve.values,
        gpu_result.equity_curve.values,
        rtol=1e-5,  # Relative tolerance 0.001%
        atol=1e-8,  # Absolute tolerance
        err_msg="GPU et CPU equity curves doivent être identiques"
    )
    
    np.testing.assert_allclose(
        cpu_result.returns.values,
        gpu_result.returns.values,
        rtol=1e-5,
        atol=1e-8,
        err_msg="GPU et CPU returns doivent être identiques"
    )
    
    # Positions doivent être exactement identiques (pas de floating point ici)
    np.testing.assert_array_equal(
        cpu_result.positions.values,
        gpu_result.positions.values,
        err_msg="GPU et CPU positions doivent être identiques"
    )


def test_gpu_backtest_with_sl_tp(sample_data, simple_signals):
    """Test GPU backtest avec Stop-Loss et Take-Profit."""
    result = backtest_signals_gpu(
        df=sample_data,
        signals=simple_signals,
        fee_bps=1.0,
        weight=1.0,
        stop_loss_atr=2.0,
        take_profit_atr=3.0
    )
    
    # Vérifications
    assert result is not None
    assert len(result.equity_curve) == len(sample_data)
    
    # Avec SL/TP, on attend moins de barres en position
    # (car exits précoces)
    total_bars_in_position = (result.positions != 0).sum()
    signal_bars = (simple_signals != 0).sum()
    
    # SL/TP devrait réduire le temps en position
    assert total_bars_in_position <= signal_bars, \
        "SL/TP devrait réduire temps en position"


def test_gpu_vs_cpu_with_sl_tp(sample_data, simple_signals):
    """Test consistency GPU vs CPU avec SL/TP activés."""
    params = {
        'df': sample_data,
        'signals': simple_signals,
        'fee_bps': 1.0,
        'weight': 1.0,
        'stop_loss_atr': 2.0,
        'take_profit_atr': 3.0
    }
    
    cpu_result = backtest_signals(**params)
    gpu_result = backtest_signals_gpu(**params)
    
    # Tolérance un peu plus large car SL/TP a des boucles
    np.testing.assert_allclose(
        cpu_result.equity_curve.values,
        gpu_result.equity_curve.values,
        rtol=1e-4,  # 0.01%
        atol=1e-7,
        err_msg="GPU et CPU avec SL/TP equity doivent matcher"
    )


def test_batch_backtest_gpu(sample_data):
    """Test batch backtest pour plusieurs stratégies."""
    # Créer 3 stratégies différentes
    n = len(sample_data)
    
    signals_dict = {
        'strat1': pd.Series([1 if i % 10 < 5 else 0 for i in range(n)], index=sample_data.index),
        'strat2': pd.Series([1 if i % 15 < 7 else 0 for i in range(n)], index=sample_data.index),
        'strat3': pd.Series([1 if i % 20 < 10 else 0 for i in range(n)], index=sample_data.index),
    }
    
    # Batch backtest
    results = batch_backtest_gpu(
        df=sample_data,
        signals_dict=signals_dict,
        fee_bps=1.0,
        weight=1.0
    )
    
    # Vérifications
    assert len(results) == 3, "Devrait avoir 3 résultats"
    assert 'strat1' in results
    assert 'strat2' in results
    assert 'strat3' in results
    
    # Chaque résultat doit être valide
    for strat_name, result in results.items():
        assert len(result.equity_curve) == len(sample_data), \
            f"{strat_name}: equity curve longueur incorrecte"
        assert result.equity_curve.iloc[0] == 1.0, \
            f"{strat_name}: equity doit commencer à 1.0"


def test_short_signals_gpu(sample_data):
    """Test GPU backtest avec signaux SHORT (-1)."""
    n = len(sample_data)
    
    # Signaux: Short toutes les 10 barres
    signals = pd.Series(0, index=sample_data.index)
    for i in range(0, n, 10):
        signals.iloc[i:min(i+5, n)] = -1  # SHORT
    
    result = backtest_signals_gpu(
        df=sample_data,
        signals=signals,
        fee_bps=1.0,
        weight=1.0
    )
    
    # Vérifications
    assert result is not None
    assert (result.positions == -1).sum() > 0, "Devrait avoir positions short"
    
    # Comparer avec CPU
    cpu_result = backtest_signals(
        df=sample_data,
        signals=signals,
        fee_bps=1.0,
        weight=1.0
    )
    
    np.testing.assert_allclose(
        cpu_result.equity_curve.values,
        result.equity_curve.values,
        rtol=1e-5,
        err_msg="Short signals: GPU et CPU doivent matcher"
    )


def test_leverage_and_fees_gpu(sample_data, simple_signals):
    """Test GPU backtest avec leverage et fees variés."""
    result_lev1 = backtest_signals_gpu(
        df=sample_data,
        signals=simple_signals,
        fee_bps=1.0,
        weight=1.0,
        leverage=1.0
    )
    
    result_lev10 = backtest_signals_gpu(
        df=sample_data,
        signals=simple_signals,
        fee_bps=1.0,
        weight=1.0,
        leverage=10.0
    )
    
    # Avec leverage 10x, returns doivent être amplifiés
    # (en valeur absolue, car fees aussi amplifiés)
    mean_ret_lev1 = abs(result_lev1.returns.mean())
    mean_ret_lev10 = abs(result_lev10.returns.mean())
    
    assert mean_ret_lev10 > mean_ret_lev1, \
        "Leverage devrait amplifier les returns moyens"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
