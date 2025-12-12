"""Adaptive Position Sizing Module.

Provides methods to calculate optimal position size based on:
- Kelly Criterion
- Volatility Targeting
- Account Risk %
"""
import numpy as np
import pandas as pd
from typing import Optional

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def kelly_criterion(
    win_rate: float,
    win_loss_ratio: float,
    fraction: float = 1.0
) -> float:
    """Calculate Kelly Criterion fraction.
    
    f* = p - q/b
    where:
    p = win probability
    q = loss probability (1-p)
    b = win/loss ratio
    
    Args:
        win_rate: Probability of winning (0.0 to 1.0)
        win_loss_ratio: Avg Win / Avg Loss
        fraction: Kelly fraction to use (e.g. 0.5 for Half Kelly)
        
    Returns:
        Optimal position size (0.0 to 1.0)
    """
    if win_loss_ratio <= 0:
        return 0.0
        
    kelly = win_rate - (1 - win_rate) / win_loss_ratio
    return max(0.0, kelly * fraction)


def volatility_target_sizing(
    volatility: float,
    target_volatility: float = 0.20,
    max_leverage: float = 1.0
) -> float:
    """Calculate size to achieve target volatility.
    
    Size = Target Vol / Current Vol
    
    Args:
        volatility: Current annualized volatility (e.g. 0.30)
        target_volatility: Target annualized volatility (e.g. 0.20)
        max_leverage: Maximum allowed leverage
        
    Returns:
        Position size (capped at max_leverage)
    """
    if volatility <= 0:
        return 0.0
        
    size = target_volatility / volatility
    return min(size, max_leverage)


class AdaptiveSizer:
    """Helper class to manage adaptive sizing state."""
    
    def __init__(
        self, 
        method: str = "volatility", 
        target_risk: float = 0.01,
        max_drawdown: float = 0.50,
        aggressive_mode: bool = False
    ):
        self.method = method
        self.target_risk = target_risk
        self.max_drawdown = max_drawdown
        self.aggressive_mode = aggressive_mode
        
        # Trade statistics
        self.wins = 0
        self.losses = 0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        
        # Equity tracking for drawdown
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_drawdown = 0.0
        
        # Winning/losing streak tracking
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
    def update(self, pnl: float, current_equity: Optional[float] = None):
        """Update stats with trade PnL."""
        if pnl > 0:
            self.wins += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            # Running average
            self.avg_win = (self.avg_win * (self.wins - 1) + pnl) / self.wins
        elif pnl < 0:
            self.losses += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.avg_loss = (self.avg_loss * (self.losses - 1) + abs(pnl)) / self.losses
            
        # Update equity tracking
        if current_equity is not None:
            self.current_equity = current_equity
            if current_equity > self.peak_equity:
                self.peak_equity = current_equity
            if self.peak_equity > 0:
                self.current_drawdown = (self.peak_equity - current_equity) / self.peak_equity
                
    def get_drawdown_multiplier(self) -> float:
        """
        Get position size multiplier based on drawdown.
        
        In aggressive mode:
        - Scale UP during winning streaks (up to 2x)
        - Scale DOWN during losing streaks (down to 0.3x)
        - Reduce significantly when approaching max drawdown
        """
        multiplier = 1.0
        
        # Drawdown protection
        if self.current_drawdown > 0:
            # Linear reduction as we approach max drawdown
            dd_ratio = self.current_drawdown / self.max_drawdown
            
            if dd_ratio > 0.8:
                # Very close to max DD - reduce to minimum
                multiplier = 0.25
            elif dd_ratio > 0.5:
                # Getting risky - reduce significantly
                multiplier = 0.5
            elif dd_ratio > 0.3:
                # Moderate caution
                multiplier = 0.75
                
        # Streak-based adjustment (aggressive mode only)
        if self.aggressive_mode:
            # Increase size on winning streaks (martingale-lite)
            if self.consecutive_wins >= 3:
                multiplier *= min(1.5, 1 + (self.consecutive_wins - 2) * 0.1)
            elif self.consecutive_wins >= 2:
                multiplier *= 1.2
                
            # Decrease on losing streaks (anti-martingale)
            if self.consecutive_losses >= 3:
                multiplier *= max(0.3, 1 - (self.consecutive_losses - 2) * 0.15)
            elif self.consecutive_losses >= 2:
                multiplier *= 0.6
                
        return max(0.2, min(2.0, multiplier))
            
    def get_size(
        self, 
        volatility: Optional[float] = None,
        probability: Optional[float] = None
    ) -> float:
        """
        Get recommended position size.
        
        Args:
            volatility: Current volatility estimate
            probability: ML model probability (0.5 to 1.0)
        """
        base_size = 1.0
        
        if self.method == "kelly":
            total = self.wins + self.losses
            if total < 10:
                base_size = 0.2 if self.aggressive_mode else 0.1
            else:
                win_rate = self.wins / total
                ratio = self.avg_win / self.avg_loss if self.avg_loss > 0 else 1.0
                
                # Use full Kelly in aggressive mode, half Kelly otherwise
                fraction = 0.75 if self.aggressive_mode else 0.5
                base_size = kelly_criterion(win_rate, ratio, fraction=fraction)
                
        elif self.method == "volatility":
            if volatility is not None:
                # Higher target vol in aggressive mode
                target_vol = 0.35 if self.aggressive_mode else 0.20
                base_size = volatility_target_sizing(
                    volatility, 
                    target_volatility=target_vol,
                    max_leverage=2.0 if self.aggressive_mode else 1.0
                )
                
        # Apply probability boost
        if probability is not None and probability > 0.5:
            # Scale by confidence: 0.5 -> 1.0x, 0.7 -> 1.4x, 0.9 -> 1.8x
            prob_multiplier = 1.0 + (probability - 0.5) * 2.0
            base_size *= prob_multiplier
            
        # Apply drawdown multiplier
        base_size *= self.get_drawdown_multiplier()
        
        # Cap at reasonable maximum
        max_size = 2.0 if self.aggressive_mode else 1.0
        return max(0.05, min(max_size, base_size))
        
    def get_stats(self) -> dict:
        """Get current statistics."""
        total = self.wins + self.losses
        return {
            "win_rate": self.wins / total if total > 0 else 0.5,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "expectancy": (self.avg_win * self.wins - self.avg_loss * self.losses) / total if total > 0 else 0,
            "current_drawdown": self.current_drawdown,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "total_trades": total
        }

