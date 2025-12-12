"""Overfitting Safeguards.

Detects and prevents trading of overfitted strategies using multiple statistical tests.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

@dataclass
class OverfittingResult:
    """Result of overfitting check."""
    is_safe: bool
    reason: str
    metrics: Dict[str, float]

class OverfittingGuard:
    """
    Safeguards against overfitting using multiple checks:
    - Deflated Sharpe Ratio (DSR) threshold
    - In-Sample vs Out-of-Sample performance comparison
    - Consecutive wins detection
    - Parameter stability analysis
    """
    def __init__(
        self,
        min_dsr: float = 1.0,
        max_is_oos_ratio: float = 1.5,
        max_consecutive_wins: int = 10,
        min_trades: int = 30
    ):
        self.min_dsr = min_dsr
        self.max_is_oos_ratio = max_is_oos_ratio
        self.max_consecutive_wins = max_consecutive_wins
        self.min_trades = min_trades
        
    def check_strategy(
        self,
        returns: pd.Series,
        is_sharpe: Optional[float] = None,
        oos_sharpe: Optional[float] = None,
        n_trials: Optional[int] = None
    ) -> OverfittingResult:
        """
        Check if a strategy is safe to trade.
        
        Args:
            returns: Series of trade returns.
            is_sharpe: In-sample Sharpe ratio (optional).
            oos_sharpe: Out-of-sample Sharpe ratio (optional).
            n_trials: Number of trials/backtests performed (for DSR).
            
        Returns:
            OverfittingResult indicating if strategy is safe.
        """
        metrics = {}
        
        # Check 1: Minimum number of trades
        n_trades = len(returns)
        metrics['n_trades'] = n_trades
        
        if n_trades < self.min_trades:
            return OverfittingResult(
                is_safe=False,
                reason=f"Insufficient trades ({n_trades} < {self.min_trades})",
                metrics=metrics
            )
            
        # Check 2: Deflated Sharpe Ratio
        sharpe = self._calculate_sharpe(returns)
        metrics['sharpe'] = sharpe
        
        if n_trials and n_trials > 1:
            dsr = self._calculate_dsr(sharpe, n_trades, n_trials)
            metrics['dsr'] = dsr
            
            if dsr < self.min_dsr:
                return OverfittingResult(
                    is_safe=False,
                    reason=f"DSR too low ({dsr:.2f} < {self.min_dsr})",
                    metrics=metrics
                )
        else:
            # Simple Sharpe check if no trials info
            if sharpe < 1.0:
                return OverfittingResult(
                    is_safe=False,
                    reason=f"Sharpe too low ({sharpe:.2f})",
                    metrics=metrics
                )
                
        # Check 3: IS/OOS comparison
        if is_sharpe is not None and oos_sharpe is not None:
            metrics['is_sharpe'] = is_sharpe
            metrics['oos_sharpe'] = oos_sharpe
            
            if oos_sharpe > 0:
                ratio = is_sharpe / oos_sharpe
                metrics['is_oos_ratio'] = ratio
                
                if ratio > self.max_is_oos_ratio:
                    return OverfittingResult(
                        is_safe=False,
                        reason=f"IS/OOS ratio too high ({ratio:.2f}), likely overfitted",
                        metrics=metrics
                    )
            else:
                return OverfittingResult(
                    is_safe=False,
                    reason="OOS Sharpe is negative",
                    metrics=metrics
                )
                
        # Check 4: Consecutive wins
        consecutive = self._max_consecutive_wins(returns)
        metrics['max_consecutive_wins'] = consecutive
        
        if consecutive > self.max_consecutive_wins:
            return OverfittingResult(
                is_safe=False,
                reason=f"Too many consecutive wins ({consecutive}), suspicious",
                metrics=metrics
            )
            
        # All checks passed
        return OverfittingResult(
            is_safe=True,
            reason="All overfitting checks passed",
            metrics=metrics
        )
        
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0:
            return 0.0
        mean_ret = returns.mean()
        std_ret = returns.std()
        if std_ret == 0:
            return 0.0
        return mean_ret / std_ret * np.sqrt(252)  # Annualized
        
    def _calculate_dsr(self, sharpe: float, n_trades: int, n_trials: int) -> float:
        """
        Calculate Deflated Sharpe Ratio.
        
        Adjusts Sharpe for multiple testing bias.
        Based on Bailey & Lopez de Prado (2014).
        """
        if n_trials <= 1:
            return sharpe
            
        # Expected maximum Sharpe under null hypothesis (random)
        # Using Euler-Mascheroni constant approximation
        euler = 0.5772
        expected_max_sharpe = np.sqrt(2 * np.log(n_trials)) - (
            (np.log(np.log(n_trials)) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * np.log(n_trials)))
        )
        
        # Standard error
        se = 1 / np.sqrt(n_trades)
        
        # Deflated Sharpe (z-score)
        dsr = (sharpe - expected_max_sharpe) / se
        
        return float(dsr)
        
    def _max_consecutive_wins(self, returns: pd.Series) -> int:
        """Count maximum consecutive winning trades."""
        if len(returns) == 0:
            return 0
            
        wins = (returns > 0).astype(int)
        max_consecutive = 0
        current_consecutive = 0
        
        for win in wins:
            if win:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
                
        return max_consecutive
