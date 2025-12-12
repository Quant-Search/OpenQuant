"""Kelly Criterion with Adaptive Position Sizing.

Implements Kelly Criterion for optimal position sizing with:
- Win-rate and payoff ratio estimation from historical trades
- Volatility-adjusted scaling
- Drawdown-based position reduction
- Conservative fractional Kelly (default 0.5x)
- Safety caps and minimum trade history requirements
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class TradeRecord:
    """Single trade outcome for Kelly statistics."""
    pnl: float
    entry_price: float
    exit_price: float
    size: float
    duration_bars: int = 0


@dataclass
class KellyStats:
    """Kelly Criterion statistics."""
    win_rate: float = 0.5
    avg_win: float = 0.0
    avg_loss: float = 0.0
    payoff_ratio: float = 1.0  # avg_win / avg_loss
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    kelly_fraction: float = 0.0
    
    def expectancy(self) -> float:
        """Expected value per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.win_rate * self.avg_win - (1 - self.win_rate) * self.avg_loss


class KellyCriterion:
    """Kelly Criterion position sizer with volatility and drawdown adjustments.
    
    Core formula: f* = p - q/b
    where:
        p = win probability
        q = loss probability (1-p)
        b = payoff ratio (avg_win / avg_loss)
        
    Additional features:
    - Fractional Kelly (default 0.5x for safety)
    - Volatility scaling (inverse volatility weighting)
    - Drawdown-based reduction (reduce size as drawdown increases)
    - Minimum trade history requirement (conservative sizing until sufficient data)
    """
    
    def __init__(
        self,
        kelly_fraction: float = 0.5,
        min_trades: int = 20,
        volatility_target: float = 0.20,
        max_drawdown_threshold: float = 0.15,
        drawdown_scale_factor: float = 2.0,
        max_position_size: float = 1.0,
        min_position_size: float = 0.0,
    ):
        """Initialize Kelly Criterion sizer.
        
        Args:
            kelly_fraction: Fraction of Kelly to use (0.5 = Half Kelly, safer)
            min_trades: Minimum trades before using full Kelly (use conservative sizing before)
            volatility_target: Target annualized volatility (e.g., 0.20 = 20%)
            max_drawdown_threshold: Drawdown threshold to start reducing size (e.g., 0.15 = 15%)
            drawdown_scale_factor: How aggressively to reduce on drawdown (higher = more aggressive)
            max_position_size: Maximum position size as fraction of capital
            min_position_size: Minimum position size threshold (below this, no trade)
        """
        self.kelly_fraction = kelly_fraction
        self.min_trades = min_trades
        self.volatility_target = volatility_target
        self.max_drawdown_threshold = max_drawdown_threshold
        self.drawdown_scale_factor = drawdown_scale_factor
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        
        # Trade history for statistics
        self.trade_history: List[TradeRecord] = []
        
        # Equity tracking for drawdown
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.current_drawdown: float = 0.0
        
        # Cached statistics
        self._stats: Optional[KellyStats] = None
        self._stats_dirty: bool = True
        
    def record_trade(
        self,
        pnl: float,
        entry_price: float,
        exit_price: float,
        size: float,
        duration_bars: int = 0,
    ) -> None:
        """Record a completed trade for Kelly statistics.
        
        Args:
            pnl: Profit/loss in currency units
            entry_price: Entry price
            exit_price: Exit price
            size: Position size (units)
            duration_bars: Trade duration in bars
        """
        trade = TradeRecord(
            pnl=pnl,
            entry_price=entry_price,
            exit_price=exit_price,
            size=size,
            duration_bars=duration_bars,
        )
        self.trade_history.append(trade)
        self._stats_dirty = True
        
        LOGGER.debug(
            f"Recorded trade: PnL={pnl:.2f}, Entry={entry_price:.4f}, "
            f"Exit={exit_price:.4f}, Size={size:.4f}"
        )
        
    def update_equity(self, equity: float) -> None:
        """Update equity for drawdown tracking.
        
        Args:
            equity: Current account equity
        """
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
            
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
        else:
            self.current_drawdown = 0.0
            
    def _compute_stats(self) -> KellyStats:
        """Compute Kelly statistics from trade history."""
        if len(self.trade_history) == 0:
            return KellyStats()
            
        wins = [t for t in self.trade_history if t.pnl > 0]
        losses = [t for t in self.trade_history if t.pnl < 0]
        
        total_trades = len(self.trade_history)
        n_wins = len(wins)
        n_losses = len(losses)
        
        win_rate = n_wins / total_trades if total_trades > 0 else 0.5
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 1.0
        
        # Payoff ratio (b in Kelly formula)
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Kelly formula: f* = p - q/b = p - (1-p)/b
        kelly_optimal = win_rate - (1 - win_rate) / payoff_ratio if payoff_ratio > 0 else 0.0
        kelly_optimal = max(0.0, kelly_optimal)  # Can't be negative
        
        # Apply fractional Kelly
        kelly_frac = kelly_optimal * self.kelly_fraction
        
        return KellyStats(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            payoff_ratio=payoff_ratio,
            total_trades=total_trades,
            wins=n_wins,
            losses=n_losses,
            kelly_fraction=kelly_frac,
        )
        
    def get_stats(self) -> KellyStats:
        """Get current Kelly statistics (cached).
        
        Returns:
            KellyStats object with current statistics
        """
        if self._stats_dirty or self._stats is None:
            self._stats = self._compute_stats()
            self._stats_dirty = False
        return self._stats
        
    def compute_base_kelly_size(self) -> float:
        """Compute base Kelly position size from trade history.
        
        Returns:
            Base Kelly fraction (0.0 to max_position_size)
        """
        stats = self.get_stats()
        
        # If insufficient trades, use conservative sizing
        if stats.total_trades < self.min_trades:
            # Linear ramp from 10% to full Kelly as trades accumulate
            ramp_factor = stats.total_trades / self.min_trades
            conservative_size = 0.1 * ramp_factor
            LOGGER.debug(
                f"Insufficient trades ({stats.total_trades}/{self.min_trades}), "
                f"using conservative size: {conservative_size:.3f}"
            )
            return conservative_size
            
        # Use Kelly fraction
        kelly_size = stats.kelly_fraction
        
        # Safety cap
        kelly_size = min(kelly_size, self.max_position_size)
        
        LOGGER.debug(
            f"Kelly size: {kelly_size:.3f} (win_rate={stats.win_rate:.3f}, "
            f"payoff_ratio={stats.payoff_ratio:.3f})"
        )
        
        return kelly_size
        
    def apply_volatility_adjustment(
        self,
        base_size: float,
        volatility: float,
    ) -> float:
        """Adjust position size based on volatility.
        
        Higher volatility -> smaller position (inverse volatility weighting)
        
        Args:
            base_size: Base position size from Kelly
            volatility: Current annualized volatility (e.g., 0.30 = 30%)
            
        Returns:
            Volatility-adjusted position size
        """
        if volatility <= 0:
            LOGGER.warning("Volatility <= 0, returning base size")
            return base_size
            
        # Volatility scaling factor: target_vol / current_vol
        vol_adjustment = self.volatility_target / volatility
        
        # Don't scale up too much (cap at 2x)
        vol_adjustment = min(vol_adjustment, 2.0)
        
        adjusted_size = base_size * vol_adjustment
        
        LOGGER.debug(
            f"Volatility adjustment: {vol_adjustment:.3f} "
            f"(target={self.volatility_target:.3f}, current={volatility:.3f})"
        )
        
        return adjusted_size
        
    def apply_drawdown_scaling(
        self,
        size: float,
    ) -> float:
        """Apply drawdown-based position size reduction.
        
        As drawdown increases beyond threshold, reduce position size quadratically.
        
        Args:
            size: Current position size
            
        Returns:
            Drawdown-scaled position size
        """
        if self.current_drawdown <= self.max_drawdown_threshold:
            return size
            
        # Excess drawdown beyond threshold
        excess_dd = self.current_drawdown - self.max_drawdown_threshold
        
        # Quadratic reduction: scale_factor = 1 - (excess_dd * drawdown_scale_factor)^2
        # This creates aggressive reduction as drawdown increases
        reduction = (excess_dd * self.drawdown_scale_factor) ** 2
        scale_factor = max(0.1, 1.0 - reduction)  # Keep at least 10% size
        
        scaled_size = size * scale_factor
        
        LOGGER.info(
            f"Drawdown scaling: {scale_factor:.3f} "
            f"(DD={self.current_drawdown:.3%}, threshold={self.max_drawdown_threshold:.3%})"
        )
        
        return scaled_size
        
    def compute_position_size(
        self,
        volatility: Optional[float] = None,
    ) -> float:
        """Compute optimal position size with all adjustments.
        
        Args:
            volatility: Current annualized volatility (optional)
            
        Returns:
            Final position size as fraction of capital (0.0 to max_position_size)
        """
        # Step 1: Base Kelly size from trade statistics
        size = self.compute_base_kelly_size()
        
        # Step 2: Volatility adjustment (if provided)
        if volatility is not None and volatility > 0:
            size = self.apply_volatility_adjustment(size, volatility)
            
        # Step 3: Drawdown scaling
        size = self.apply_drawdown_scaling(size)
        
        # Step 4: Apply caps
        size = max(self.min_position_size, min(size, self.max_position_size))
        
        LOGGER.info(f"Final position size: {size:.3f}")
        
        return size
        
    def get_summary(self) -> Dict:
        """Get summary of Kelly Criterion state.
        
        Returns:
            Dictionary with statistics and current state
        """
        stats = self.get_stats()
        return {
            "kelly_fraction": stats.kelly_fraction,
            "win_rate": stats.win_rate,
            "payoff_ratio": stats.payoff_ratio,
            "avg_win": stats.avg_win,
            "avg_loss": stats.avg_loss,
            "expectancy": stats.expectancy(),
            "total_trades": stats.total_trades,
            "wins": stats.wins,
            "losses": stats.losses,
            "current_drawdown": self.current_drawdown,
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
        }


def compute_rolling_volatility(
    prices: np.ndarray,
    window: int = 20,
    annualization_factor: float = 252.0,
) -> float:
    """Compute rolling annualized volatility from price series.
    
    Args:
        prices: Price array (most recent last)
        window: Lookback window for volatility calculation
        annualization_factor: Annualization factor (252 for daily, 252*24 for hourly, etc.)
        
    Returns:
        Annualized volatility
    """
    if len(prices) < window:
        window = len(prices)
        
    if window < 2:
        return 0.0
        
    # Use last 'window' prices
    recent_prices = prices[-window:]
    
    # Log returns
    returns = np.diff(np.log(recent_prices))
    
    # Annualized volatility
    vol = np.std(returns) * np.sqrt(annualization_factor)
    
    return float(vol)


def estimate_win_rate_from_signals(
    signals: np.ndarray,
    returns: np.ndarray,
) -> Tuple[float, float, float]:
    """Estimate win rate and payoff ratio from signal/return history.
    
    Args:
        signals: Array of signals (-1, 0, 1)
        returns: Array of forward returns
        
    Returns:
        Tuple of (win_rate, avg_win, avg_loss)
    """
    # Filter to only periods with signals
    mask = signals != 0
    if not np.any(mask):
        return 0.5, 0.0, 0.0
        
    # Signal returns (signal * forward_return)
    signal_returns = signals[mask] * returns[mask]
    
    # Wins and losses
    wins = signal_returns[signal_returns > 0]
    losses = signal_returns[signal_returns < 0]
    
    total = len(signal_returns)
    if total == 0:
        return 0.5, 0.0, 0.0
        
    win_rate = len(wins) / total
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0.0
    
    return win_rate, avg_win, avg_loss
