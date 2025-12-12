from __future__ import annotations
"""Simple paper order simulator using target weights from allocation JSON.

For now, we operate in "notional weights" per key and pretend we can
rebalance immediately at a given price snapshot (no slippage model yet).

Later we can add: next-bar-open fills, slippage, fees, partial fills.
"""
from typing import Dict, List, Tuple, Iterable, Optional
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .state import PortfolioState, Key
from ..risk.kill_switch import KILL_SWITCH
from ..risk.circuit_breaker import CIRCUIT_BREAKER
from ..risk.kelly_criterion import KellyCriterion, compute_rolling_volatility
from ..risk.trade_validator import TRADE_VALIDATOR
from ..risk.adaptive_sizing import AdaptiveSizer
from ..quant.regime_detector import RegimeDetector, RegimeType
from ..reporting.performance_tracker import PerformanceTracker
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class MarketSnapshot:
    prices: Dict[Key, float]  # last trade price per key (quote currency units)
    next_prices: Dict[Key, float] = field(default_factory=dict)  # next-bar prices for fills (optional)
    price_history: Dict[Key, np.ndarray] = field(default_factory=dict)  # historical prices for volatility calculation


class PortfolioOptimizer:
    """Portfolio optimizer for optimal allocation using mean-variance optimization."""
    
    def __init__(self, method: str = "sharpe", risk_free_rate: float = 0.0):
        """
        Args:
            method: Optimization method ('sharpe', 'min_variance', 'max_return')
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.method = method
        self.risk_free_rate = risk_free_rate
        
    def optimize_allocation(
        self, 
        returns: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Calculate optimal portfolio weights.
        
        Args:
            returns: DataFrame with returns for each asset (columns are assets)
            constraints: Optional constraints dict with 'min_weight', 'max_weight', 'max_assets'
            
        Returns:
            Dictionary mapping asset keys to optimal weights
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            LOGGER.warning("scipy not installed, using equal weights")
            n = len(returns.columns)
            return {str(col): 1.0 / n for col in returns.columns}
            
        if returns.empty or len(returns.columns) == 0:
            return {}
            
        # Calculate expected returns and covariance
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        n_assets = len(returns.columns)
        
        # Set default constraints
        if constraints is None:
            constraints = {}
        min_weight = constraints.get('min_weight', 0.0)
        max_weight = constraints.get('max_weight', 0.5)
        
        def portfolio_stats(weights):
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            return portfolio_return, portfolio_std
            
        def neg_sharpe(weights):
            p_ret, p_std = portfolio_stats(weights)
            if p_std == 0:
                return 0.0
            return -(p_ret - self.risk_free_rate) / p_std
            
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
            
        # Objective function
        if self.method == "sharpe":
            objective = neg_sharpe
        elif self.method == "min_variance":
            objective = portfolio_variance
        else:  # max_return
            objective = lambda w: -np.dot(w, mean_returns)
            
        # Constraints: weights sum to 1
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: min_weight <= weight <= max_weight
        bounds = tuple((min_weight, max_weight) for _ in range(n_assets))
        
        # Initial guess: equal weights
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        try:
            result = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = result.x
                # Map back to asset keys
                allocation = {}
                for i, col in enumerate(returns.columns):
                    if weights[i] > 1e-6:  # Filter out tiny weights
                        allocation[str(col)] = float(weights[i])
                return allocation
            else:
                LOGGER.warning(f"Optimization failed: {result.message}, using equal weights")
                return {str(col): 1.0 / n_assets for col in returns.columns}
        except Exception as e:
            LOGGER.error(f"Optimization error: {e}, using equal weights")
            return {str(col): 1.0 / n_assets for col in returns.columns}


class EnhancedSimulator:
    """
    Enhanced paper trading simulator with integrated:
    - Portfolio optimizer for allocation calculation
    - Adaptive sizer for position sizing
    - Regime detector for exposure adjustment
    - Performance tracker for degradation alerts
    """
    
    def __init__(
        self,
        optimizer_method: str = "sharpe",
        sizer_method: str = "volatility",
        regime_lookback: int = 100,
        performance_path: Optional[str] = None
    ):
        """
        Args:
            optimizer_method: Portfolio optimization method ('sharpe', 'min_variance', 'max_return')
            sizer_method: Adaptive sizing method ('volatility', 'kelly')
            regime_lookback: Lookback period for regime detection
            performance_path: Path for performance tracker storage
        """
        self.portfolio_optimizer = PortfolioOptimizer(method=optimizer_method)
        self.adaptive_sizer = AdaptiveSizer(method=sizer_method)
        self.regime_detector = RegimeDetector(lookback=regime_lookback)
        
        if performance_path:
            from pathlib import Path
            self.performance_tracker = PerformanceTracker(data_path=Path(performance_path))
        else:
            self.performance_tracker = PerformanceTracker()
            
        self._trading_halted = False
        self._halt_reason = ""
        
    def check_profitability_constraints(
        self,
        lookback_days: int = 30,
        min_sharpe: float = 1.0,
        max_drawdown: float = 0.20
    ) -> Tuple[bool, str]:
        """
        Check if trading should be halted due to poor performance.
        
        Halts trading if:
        - Rolling Sharpe ratio drops below min_sharpe (default 1.0)
        - Current drawdown exceeds max_drawdown (default 20%)
        
        Args:
            lookback_days: Period for rolling metrics calculation
            min_sharpe: Minimum acceptable Sharpe ratio
            max_drawdown: Maximum acceptable drawdown (as decimal, e.g., 0.20 = 20%)
            
        Returns:
            Tuple of (should_halt, reason)
        """
        stats = self.performance_tracker.get_stats(lookback_days=lookback_days)
        
        # Check Sharpe ratio
        sharpe = stats.get('sharpe_estimate', 0.0)
        if sharpe < min_sharpe:
            reason = f"Rolling Sharpe ({sharpe:.2f}) below minimum ({min_sharpe:.2f})"
            LOGGER.warning(f"PROFITABILITY CONSTRAINT VIOLATED: {reason}")
            return True, reason
            
        # Check drawdown
        current_dd = stats.get('current_drawdown', 0.0)
        if current_dd > max_drawdown:
            reason = f"Current drawdown ({current_dd:.2%}) exceeds maximum ({max_drawdown:.2%})"
            LOGGER.warning(f"PROFITABILITY CONSTRAINT VIOLATED: {reason}")
            return True, reason
            
        return False, ""
        
    def halt_trading(self, reason: str):
        """Halt all trading operations."""
        self._trading_halted = True
        self._halt_reason = reason
        LOGGER.critical(f"TRADING HALTED: {reason}")
        
    def resume_trading(self):
        """Resume trading operations."""
        self._trading_halted = False
        self._halt_reason = ""
        LOGGER.info("Trading resumed")
        
    def is_trading_halted(self) -> Tuple[bool, str]:
        """Check if trading is currently halted."""
        return self._trading_halted, self._halt_reason
        
    def compute_optimized_allocation(
        self,
        historical_prices: Dict[Key, pd.Series],
        base_targets: Optional[List[Tuple[Key, float]]] = None,
        regime_data: Optional[pd.DataFrame] = None
    ) -> List[Tuple[Key, float]]:
        """
        Compute optimized portfolio allocation with regime adjustment.
        
        Args:
            historical_prices: Dict mapping keys to price Series for optimization
            base_targets: Optional base target weights to adjust
            regime_data: Optional DataFrame for regime detection
            
        Returns:
            List of (key, weight) tuples with optimized allocations
        """
        if not historical_prices:
            return base_targets or []
            
        # Calculate returns for each asset
        returns_dict = {}
        for key, prices in historical_prices.items():
            if len(prices) > 1:
                returns_dict[str(key)] = prices.pct_change().dropna()
                
        if not returns_dict:
            return base_targets or []
            
        # Align all returns to common index
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            return base_targets or []
            
        # Optimize allocation
        optimized_weights = self.portfolio_optimizer.optimize_allocation(returns_df)
        
        # Apply regime adjustment if regime data provided
        regime_multiplier = 1.0
        if regime_data is not None and not regime_data.empty:
            try:
                regime_info = self.regime_detector.detect_regime(regime_data)
                
                # Adjust exposure based on regime
                trend_regime = regime_info.get('trend_regime')
                vol_regime = regime_info.get('volatility_regime')
                
                # Reduce exposure in high volatility or ranging markets
                if vol_regime == RegimeType.HIGH_VOLATILITY:
                    regime_multiplier *= 0.6
                    LOGGER.info("High volatility regime detected, reducing exposure by 40%")
                    
                if trend_regime == RegimeType.RANGING:
                    regime_multiplier *= 0.7
                    LOGGER.info("Ranging market detected, reducing exposure by 30%")
                    
                # Store regime info for logging
                LOGGER.info(f"Regime: {trend_regime.value if hasattr(trend_regime, 'value') else trend_regime}, "
                          f"Hurst: {regime_info.get('hurst_exponent', 0):.2f}, "
                          f"Vol: {regime_info.get('volatility', 0):.4f}")
            except Exception as e:
                LOGGER.error(f"Regime detection failed: {e}")
                
        # Convert to list of tuples and apply regime multiplier
        allocation = []
        for key_str, weight in optimized_weights.items():
            try:
                # Parse key string back to tuple if needed
                key = eval(key_str) if isinstance(key_str, str) and key_str.startswith('(') else key_str
                adjusted_weight = weight * regime_multiplier
                if adjusted_weight > 1e-6:
                    allocation.append((key, adjusted_weight))
            except Exception as e:
                LOGGER.warning(f"Failed to parse key {key_str}: {e}")
                
        return allocation
        
    def apply_adaptive_sizing(
        self,
        targets: List[Tuple[Key, float]],
        volatility: Optional[float] = None,
        current_equity: Optional[float] = None
    ) -> List[Tuple[Key, float]]:
        """
        Apply adaptive position sizing to target allocations.
        
        Args:
            targets: List of (key, weight) tuples
            volatility: Current portfolio volatility estimate
            current_equity: Current portfolio equity for tracking
            
        Returns:
            List of (key, adjusted_weight) tuples
        """
        # Update sizer with current equity if provided
        if current_equity is not None:
            self.adaptive_sizer.current_equity = current_equity
            
        # Get sizing multiplier
        size_multiplier = self.adaptive_sizer.get_size(volatility=volatility)
        
        LOGGER.info(f"Adaptive sizing multiplier: {size_multiplier:.2f}")
        
        # Apply sizing to all targets
        adjusted_targets = []
        for key, weight in targets:
            adjusted_weight = weight * size_multiplier
            if adjusted_weight > 1e-6:
                adjusted_targets.append((key, adjusted_weight))
                
        # Log sizing stats
        stats = self.adaptive_sizer.get_stats()
        LOGGER.debug(f"Sizer stats: WinRate={stats['win_rate']:.2%}, "
                    f"Expectancy={stats['expectancy']:.4f}, "
                    f"DD={stats['current_drawdown']:.2%}")
                    
        return adjusted_targets
        
    def update_performance(
        self,
        state: PortfolioState,
        snap: MarketSnapshot,
        filled_orders: List[Tuple[Key, float, float, float]]
    ):
        """
        Update performance tracker with latest equity and trade fills.
        
        Args:
            state: Current portfolio state
            snap: Market snapshot with current prices
            filled_orders: List of (key, delta_units, exec_price, fee_paid) from execute_orders
        """
        # Calculate current equity
        equity = state.cash
        for k, u in state.holdings.items():
            p = float(snap.prices.get(k, 0.0))
            equity += float(u) * p
            
        # Update equity curve
        if self.performance_tracker.initial_equity == 0:
            self.performance_tracker.set_initial_equity(equity)
        else:
            self.performance_tracker.update_equity(equity)
            
        # Record closed trades (simplified - detect full position closes)
        for key, delta_units, exec_price, fee_paid in filled_orders:
            cur_units = state.position(key)
            
            # Check if this was a position close (went to zero or flipped)
            old_units = cur_units - delta_units
            
            if (old_units != 0 and 
                ((old_units > 0 and cur_units <= 0) or (old_units < 0 and cur_units >= 0))):
                
                # Calculate P&L for the closed portion
                avg_price = state.avg_price.get(key, exec_price)
                closed_units = abs(old_units) if abs(cur_units) < abs(old_units) else abs(old_units)
                
                if old_units > 0:
                    # Closed long
                    pnl = (exec_price - avg_price) * closed_units - fee_paid
                    side = "LONG"
                else:
                    # Closed short
                    pnl = (avg_price - exec_price) * closed_units - fee_paid
                    side = "SHORT"
                    
                # Extract symbol from key
                symbol = key[1] if isinstance(key, tuple) and len(key) > 1 else str(key)
                
                self.performance_tracker.record_trade(
                    symbol=symbol,
                    side=side,
                    entry_price=avg_price,
                    exit_price=exec_price,
                    quantity=closed_units,
                    strategy=key[3] if isinstance(key, tuple) and len(key) > 3 else "unknown"
                )
                
                # Update adaptive sizer
                self.adaptive_sizer.update(pnl, equity)


def compute_target_units(state: PortfolioState, targets: List[Tuple[Key, float]], snap: MarketSnapshot) -> Dict[Key, float]:
    """Convert target weights (fraction of equity) into target units per key.

    - state.cash + sum(holdings * price) defines equity.
    - For each key with target weight w, target notional = w * equity.
    - Units = notional / price.
    """
    # compute equity
    equity = state.cash
    for k, u in state.holdings.items():
        p = float(snap.prices.get(k, 0.0))
        equity += float(u) * p
    # compute target units per key
    out: Dict[Key, float] = {}
    for (k, w) in targets:
        price = float(snap.prices.get(k, 0.0))
        if price <= 0.0 or w <= 0.0:
            out[k] = 0.0 if w <= 0.0 else 0.0
            continue
        notional = float(w) * equity
        out[k] = notional / price
    return out


def compute_target_units_with_kelly(
    state: PortfolioState,
    targets: List[Tuple[Key, float]],
    snap: MarketSnapshot,
    kelly_sizers: Dict[Key, KellyCriterion],
    volatility_window: int = 20,
    annualization_factor: float = 252.0,
) -> Dict[Key, float]:
    """Convert target weights into units with Kelly Criterion position sizing.
    
    Applies adaptive position sizing based on:
    - Kelly Criterion (win rate and payoff ratio)
    - Volatility adjustment (inverse volatility weighting)
    - Drawdown scaling (reduce size during drawdowns)
    
    Args:
        state: Current portfolio state
        targets: List of (key, signal_weight) where signal_weight is the raw strategy signal
        snap: Market snapshot with prices and price history
        kelly_sizers: Dictionary of Kelly sizers per key
        volatility_window: Lookback window for volatility calculation
        annualization_factor: Factor to annualize volatility (252 for daily bars)
        
    Returns:
        Dictionary of target units per key
    """
    # Compute equity
    equity = state.cash
    for k, u in state.holdings.items():
        p = float(snap.prices.get(k, 0.0))
        equity += float(u) * p
        
    # Update Kelly sizers with current equity
    for kelly in kelly_sizers.values():
        kelly.update_equity(equity)
        
    # Compute target units per key with Kelly sizing
    out: Dict[Key, float] = {}
    for (k, signal_weight) in targets:
        price = float(snap.prices.get(k, 0.0))
        
        # Skip if no price or zero signal
        if price <= 0.0 or abs(signal_weight) < 1e-9:
            out[k] = 0.0
            continue
            
        # Get Kelly sizer for this key (create if doesn't exist)
        if k not in kelly_sizers:
            kelly_sizers[k] = KellyCriterion()
            kelly_sizers[k].update_equity(equity)
            
        kelly = kelly_sizers[k]
        
        # Compute volatility if price history available
        volatility = None
        if k in snap.price_history and len(snap.price_history[k]) >= 2:
            volatility = compute_rolling_volatility(
                snap.price_history[k],
                window=volatility_window,
                annualization_factor=annualization_factor,
            )
            
        # Get Kelly-adjusted position size
        kelly_fraction = kelly.compute_position_size(volatility=volatility)
        
        # Apply signal direction and Kelly sizing
        # signal_weight is typically -1, 0, or 1, but could be continuous
        # Kelly fraction is 0 to max_position_size (e.g., 0 to 1.0)
        adjusted_weight = signal_weight * kelly_fraction
        
        # Compute notional and units
        notional = float(adjusted_weight) * equity
        out[k] = notional / price
        
    return out


def compute_rebalance_orders(state: PortfolioState, targets: List[Tuple[Key, float]], snap: MarketSnapshot) -> List[Tuple[Key, float, float, Optional[float], Optional[float]]]:
    """Return list of (key, delta_units, price, sl, tp) required to reach targets."""
    target_units = compute_target_units(state, targets, snap)
    orders: List[Tuple[Key, float, float, Optional[float], Optional[float]]] = []
    for k, tgt_u in target_units.items():
        cur_u = state.position(k)
        if abs(tgt_u - cur_u) <= 1e-9:
            continue
        price = float(snap.prices.get(k, 0.0))
        # SL/TP are not determined here (unless we pass them in targets? yes we should)
        # But targets is List[Tuple[Key, float]] (weight).
        # We need to change targets to include SL/TP or pass a separate map.
        # For now, return None, None. The caller (paper_apply_allocation) should inject them if available.
        orders.append((k, tgt_u - cur_u, price, None, None))
    return orders


def compute_rebalance_orders_with_kelly(
    state: PortfolioState,
    targets: List[Tuple[Key, float]],
    snap: MarketSnapshot,
    kelly_sizers: Dict[Key, KellyCriterion],
    volatility_window: int = 20,
    annualization_factor: float = 252.0,
) -> List[Tuple[Key, float, float, Optional[float], Optional[float]]]:
    """Return list of orders with Kelly position sizing applied.
    
    Args:
        state: Current portfolio state
        targets: List of (key, signal_weight) tuples
        snap: Market snapshot
        kelly_sizers: Dictionary of Kelly sizers per key
        volatility_window: Lookback window for volatility
        annualization_factor: Annualization factor for volatility
        
    Returns:
        List of (key, delta_units, price, sl, tp) orders
    """
    target_units = compute_target_units_with_kelly(
        state, targets, snap, kelly_sizers, volatility_window, annualization_factor
    )
    orders: List[Tuple[Key, float, float, Optional[float], Optional[float]]] = []
    for k, tgt_u in target_units.items():
        cur_u = state.position(k)
        if abs(tgt_u - cur_u) <= 1e-9:
            continue
        price = float(snap.prices.get(k, 0.0))
        orders.append((k, tgt_u - cur_u, price, None, None))
    return orders


def record_closed_trades(
    state: PortfolioState,
    fills: List[Tuple[Key, float, float, float]],
    kelly_sizers: Dict[Key, KellyCriterion],
) -> None:
    """Record closed trades to Kelly sizers for statistics tracking.
    
    When a position is reduced or closed, record the trade outcome.
    
    Args:
        state: Current portfolio state
        fills: List of (key, delta_units, exec_price, fee_paid) from execute_orders
        kelly_sizers: Dictionary of Kelly sizers to update
    """
    for k, delta_u, exec_price, fee_paid in fills:
        # Only record when closing a position (delta_u has opposite sign to current position)
        # or when flipping position
        cur_u = state.position(k)
        prev_u = cur_u - delta_u  # Position before this fill
        
        # Check if we closed or reduced a position
        is_closing = (prev_u > 0 and delta_u < 0) or (prev_u < 0 and delta_u > 0)
        
        if not is_closing:
            continue
            
        # Get entry price
        entry_price = state.avg_price.get(k)
        if entry_price is None or entry_price <= 0:
            continue
            
        # Calculate PnL for the closed portion
        closed_units = min(abs(prev_u), abs(delta_u))
        
        if prev_u > 0:
            # Closing long: profit if exit > entry
            pnl = closed_units * (exec_price - entry_price) - fee_paid
        else:
            # Closing short: profit if exit < entry
            pnl = closed_units * (entry_price - exec_price) - fee_paid
            
        # Record to Kelly sizer
        if k not in kelly_sizers:
            kelly_sizers[k] = KellyCriterion()
            
        kelly_sizers[k].record_trade(
            pnl=pnl,
            entry_price=entry_price,
            exit_price=exec_price,
            size=closed_units,
        )


def execute_orders(
    state: PortfolioState,
    orders: Iterable[Tuple[Key, float, float, Optional[float], Optional[float]]], # Updated signature: (key, delta, price, sl, tp)
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    next_bar_fill: bool = False,
    max_fill_fraction: float = 1.0,
    snap: Optional[MarketSnapshot] = None,
) -> Tuple[Dict[str, float], List[Tuple[Key, float, float, float]]]:
    """Execute orders and update state.cash and holdings.

    Supports next-bar fills (use next_prices if available) and partial fills (max_fill_fraction < 1.0).

    Orders tuple now expects: (key, delta_units, ref_price, sl, tp)
    If sl/tp are None, they are ignored (or kept if existing).

    Returns (summary, fills) where fills are (key, delta_units, exec_price, fee_paid).

    SAFETY: Checks kill switch and circuit breaker before executing.
    If either is active, returns empty fills.
    """
    # KILL SWITCH CHECK - Critical safety mechanism
    # If kill switch is active, refuse to execute any orders
    if KILL_SWITCH.is_active():
        return {"orders": 0, "turnover": 0.0, "kill_switch_blocked": True}, []

    # CIRCUIT BREAKER CHECK - Automatic risk-based halt
    # If circuit breaker is tripped, refuse to execute any orders
    if CIRCUIT_BREAKER.is_tripped():
        return {"orders": 0, "turnover": 0.0, "circuit_breaker_blocked": True}, []

    orders_count = 0
    turnover = 0.0
    blocked_count = 0
    fills: List[Tuple[Key, float, float, float]] = []
    
    # Calculate current equity and positions for validation
    equity = state.cash
    for k, u in state.holdings.items():
        p = float(snap.prices.get(k, 0.0)) if snap else 0.0
        equity += float(u) * p
    
    current_positions = {}
    for k, u in state.holdings.items():
        p = float(snap.prices.get(k, 0.0)) if snap else 0.0
        current_positions[str(k)] = abs(float(u)) * p

    # Normalize input: if orders is list of 3-tuples, pad with None
    # This is a bit hacky to maintain backward compat if caller passes 3-tuples,
    # but better to enforce 5-tuples or check len.
    # Let's assume caller adapts or we check.

    for item in orders:
        if len(item) == 3:
            k, delta_u, ref_price = item # type: ignore
            sl, tp = None, None
        else:
            k, delta_u, ref_price, sl, tp = item # type: ignore

        if abs(delta_u) <= 1e-12:
            continue
        
        side = 1.0 if delta_u > 0 else -1.0
        
        # COMPREHENSIVE RISK VALIDATION
        validation_price = ref_price
        if next_bar_fill and snap and k in snap.next_prices:
            validation_price = snap.next_prices[k]
        
        result = TRADE_VALIDATOR.validate_trade(
            symbol=str(k),
            quantity=abs(delta_u),
            price=validation_price,
            side="buy" if side > 0 else "sell",
            portfolio_value=equity,
            current_positions=current_positions,
            current_equity=equity,
        )
        
        if not result.allowed:
            blocked_count += 1
            from openquant.utils.logging import get_logger
            logger = get_logger(__name__)
            logger.warning(f"Paper trade blocked for {k}: {result.reason}")
            continue
        
        # Use next-bar price if enabled and available
        exec_price = ref_price
        if next_bar_fill and snap and k in snap.next_prices:
            exec_price = snap.next_prices[k]
            
        # execution price includes slippage in direction of trade
        exec_price = exec_price * (1.0 + side * (slippage_bps / 1e4))
        
        # Partial fill: fill only up to max_fill_fraction of the order
        fill_fraction = min(max_fill_fraction, 1.0)
        filled_delta_u = delta_u * fill_fraction
        
        if abs(filled_delta_u) <= 1e-12:
            continue
            
        notional = abs(filled_delta_u) * exec_price
        fee_paid = notional * (fee_bps / 1e4)
        
        # cash update: buy consumes cash; sell releases cash; fees always reduce cash
        state.cash += (-notional if side > 0 else notional)
        state.cash -= fee_paid
        
        # update position and average price
        cur_u = state.position(k)
        new_u = cur_u + filled_delta_u
        
        # Update Avg Price if increasing position
        # If changing side (flip), avg price resets to exec price
        # If reducing, avg price stays same
        
        is_opening = (cur_u == 0) or (cur_u > 0 and filled_delta_u > 0) or (cur_u < 0 and filled_delta_u < 0)
        is_flip = (cur_u > 0 and new_u < 0) or (cur_u < 0 and new_u > 0)
        
        if is_flip:
            # Treated as close all + open new
            state.avg_price[k] = exec_price
        elif is_opening:
            # Weighted average
            # old_val = cur_u * old_price
            # new_part = delta * exec_price
            # total_val = old_val + new_part
            # new_avg = total_val / new_u
            old_avg = state.avg_price.get(k, exec_price)
            total_val = abs(cur_u) * old_avg + abs(filled_delta_u) * exec_price
            state.avg_price[k] = total_val / abs(new_u)
        
        # Update SL/TP if provided
        if sl is not None:
            state.sl_levels[k] = sl
        if tp is not None:
            state.tp_levels[k] = tp
            
        state.set_position(k, new_u)
        
        orders_count += 1
        turnover += notional
        fills.append((k, filled_delta_u, exec_price, fee_paid))
    
    summary = {
        "orders": float(orders_count), 
        "turnover": float(turnover),
        "blocked": float(blocked_count)
    }
    return summary, fills


def check_exits(state: PortfolioState, snap: MarketSnapshot) -> List[Tuple[Key, float, float, Optional[float], Optional[float]]]:
    """Check all positions against SL/TP levels and generate close orders if hit.
    
    Returns list of orders: (key, -units, price, None, None)
    """
    exit_orders = []
    for k, units in state.holdings.items():
        if abs(units) < 1e-9:
            continue
            
        price = snap.prices.get(k)
        if not price:
            continue
            
        sl = state.sl_levels.get(k)
        tp = state.tp_levels.get(k)
        
        hit_exit = False
        reason = ""
        
        # Long Position
        if units > 0:
            if sl and price <= sl:
                hit_exit = True
                reason = "SL"
            elif tp and price >= tp:
                hit_exit = True
                reason = "TP"
        # Short Position
        elif units < 0:
            if sl and price >= sl:
                hit_exit = True
                reason = "SL"
            elif tp and price <= tp:
                hit_exit = True
                reason = "TP"
                
        if hit_exit:
            # Close entire position
            # Order: (key, -units, price, None, None)
            # We use current price as execution price (slippage will be applied in execute)
            exit_orders.append((k, -units, price, None, None))
            
    return exit_orders


def check_daily_loss(state: PortfolioState, snap: MarketSnapshot, limit_pct: float = None) -> bool:
    """Check if daily loss exceeds limit.
    
    Returns True if trading should stop (limit hit).
    """
    if limit_pct is None:
        from openquant.config.manager import get_config
        config = get_config()
        limit_pct = config.get("paper_trading.daily_loss_limit", 0.05)
    
    if limit_pct <= 0:
        return False
        
    # Calculate current equity
    equity = state.cash
    for k, u in state.holdings.items():
        p = float(snap.prices.get(k, 0.0))
        equity += float(u) * p
        
    # Update valuation in state for other components
    state.update_valuation(equity - state.cash)
        
    # If start equity is 0 (first run), init it
    if state.daily_start_equity <= 0:
        state.daily_start_equity = equity
        
    # Calculate PnL
    start = state.daily_start_equity
    if start <= 0:
        return False
        
    pnl = equity - start
    pnl_pct = pnl / start
    
    if pnl_pct < -limit_pct:
        return True
        
    return False


def rebalance_to_targets(state: PortfolioState, targets: List[Tuple[Key, float]], snap: MarketSnapshot, *, fee_bps: float = 0.0, slippage_bps: float = 0.0, next_bar_fill: bool = False, max_fill_fraction: float = 1.0) -> Dict[str, float]:
    """Rebalance holdings to target weights at snapshot prices.

    Supports next-bar fills and partial fills.

    Returns summary dict with 'orders' (count) and 'turnover' (notional traded).
    """
    # derive orders and execute them using optional fee/slippage, next-bar, partial fills
    orders = compute_rebalance_orders(state, targets, snap)
    summary, _fills = execute_orders(state, orders, fee_bps=fee_bps, slippage_bps=slippage_bps, next_bar_fill=next_bar_fill, max_fill_fraction=max_fill_fraction, snap=snap)
    return summary


def rebalance_to_targets_with_kelly(
    state: PortfolioState,
    targets: List[Tuple[Key, float]],
    snap: MarketSnapshot,
    kelly_sizers: Dict[Key, KellyCriterion],
    *,
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    next_bar_fill: bool = False,
    max_fill_fraction: float = 1.0,
    volatility_window: int = 20,
    annualization_factor: float = 252.0,
) -> Tuple[Dict[str, float], Dict[Key, Dict]]:
    """Rebalance with Kelly Criterion adaptive position sizing.
    
    Applies Kelly-based position sizing that accounts for:
    - Historical win rate and payoff ratio
    - Current volatility (inverse volatility weighting)
    - Current drawdown (reduce size during drawdowns)
    
    Args:
        state: Current portfolio state
        targets: List of (key, signal_weight) tuples
        snap: Market snapshot with prices and price history
        kelly_sizers: Dictionary of Kelly sizers per key
        fee_bps: Fee in basis points
        slippage_bps: Slippage in basis points
        next_bar_fill: Whether to use next bar prices
        max_fill_fraction: Maximum fill fraction
        volatility_window: Lookback window for volatility
        annualization_factor: Annualization factor for volatility
        
    Returns:
        Tuple of (summary dict, kelly stats dict per key)
    """
    # Compute orders with Kelly sizing
    orders = compute_rebalance_orders_with_kelly(
        state, targets, snap, kelly_sizers, volatility_window, annualization_factor
    )
    
    # Execute orders
    summary, fills = execute_orders(
        state, orders,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
        next_bar_fill=next_bar_fill,
        max_fill_fraction=max_fill_fraction,
        snap=snap,
    )
    
    # Record closed trades for Kelly statistics
    record_closed_trades(state, fills, kelly_sizers)
    
    # Gather Kelly statistics
    kelly_stats = {}
    for k, kelly in kelly_sizers.items():
        kelly_stats[k] = kelly.get_summary()
        
    return summary, kelly_stats
