"""
Trade-Level Risk Validator Module.

Provides pre-trade risk checks that integrate:
- Asset limits (position size, concentration, leverage)
- Kill switch (emergency stop)
- Circuit breaker (automatic halt on losses/drawdown)
- Correlation constraints (prevent over-concentration in correlated assets)

This validator should be called before every order execution to ensure
trades comply with all risk management rules.

Usage:
    validator = TradeValidator()
    
    allowed, reason = validator.validate_trade(
        symbol="EURUSD",
        quantity=10000,
        price=1.10,
        side="buy",
        portfolio_value=100000,
        current_positions={"EURUSD": 5000, "GBPUSD": 3000}
    )
    
    if not allowed:
        logger.error(f"Trade rejected: {reason}")
        return
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from openquant.utils.logging import get_logger
from openquant.risk.asset_limits import ASSET_LIMITS, AssetLimitsManager
from openquant.risk.kill_switch import KILL_SWITCH, KillSwitch
from openquant.risk.circuit_breaker import CIRCUIT_BREAKER, CircuitBreaker
from openquant.risk.forex_correlation import get_correlation, check_portfolio_correlation

LOGGER = get_logger(__name__)


@dataclass
class TradeValidationResult:
    """Result of trade validation with detailed information."""
    allowed: bool
    reason: str
    failed_checks: List[str]
    warnings: List[str]
    
    def __bool__(self) -> bool:
        return self.allowed


class TradeValidator:
    """
    Comprehensive trade-level risk validator.
    
    Integrates multiple risk checks:
    1. Kill Switch: Emergency stop mechanism
    2. Circuit Breaker: Automatic halt on losses/drawdown
    3. Asset Limits: Position size, concentration, leverage limits
    4. Correlation Constraints: Prevent over-concentration in correlated assets
    
    All checks must pass for a trade to be allowed.
    """
    
    def __init__(
        self,
        asset_limits: Optional[AssetLimitsManager] = None,
        kill_switch: Optional[KillSwitch] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        correlation_threshold: float = 0.8,
    ):
        """Initialize trade validator with risk managers.
        
        Args:
            asset_limits: Asset limits manager (uses global ASSET_LIMITS if None)
            kill_switch: Kill switch instance (uses global KILL_SWITCH if None)
            circuit_breaker: Circuit breaker instance (uses global CIRCUIT_BREAKER if None)
            correlation_threshold: Correlation threshold for portfolio checks (default: 0.8)
        """
        self.asset_limits = asset_limits or ASSET_LIMITS
        self.kill_switch = kill_switch or KILL_SWITCH
        self.circuit_breaker = circuit_breaker or CIRCUIT_BREAKER
        self.correlation_threshold = correlation_threshold
        
    def validate_trade(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        portfolio_value: float,
        current_positions: Dict[str, float],
        asset_class: Optional[str] = None,
        current_equity: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> TradeValidationResult:
        """Validate a proposed trade against all risk checks.
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity (absolute value)
            price: Current/execution price
            side: 'buy' or 'sell'
            portfolio_value: Total portfolio value
            current_positions: Dict mapping {symbol: notional_value}
            asset_class: Optional asset class for symbol-specific limits
            current_equity: Optional current equity for circuit breaker update
            volatility: Optional volatility measure for circuit breaker
            
        Returns:
            TradeValidationResult with allowed flag and detailed reason
        """
        failed_checks = []
        warnings = []
        
        # Convert side to signed quantity for calculations
        signed_quantity = quantity if side.lower() == "buy" else -quantity
        
        # Check 1: Kill Switch
        if self.kill_switch.is_active():
            LOGGER.critical(f"Trade rejected for {symbol}: KILL SWITCH ACTIVE")
            return TradeValidationResult(
                allowed=False,
                reason="KILL SWITCH ACTIVE - All trading halted. Remove trigger file to resume.",
                failed_checks=["kill_switch"],
                warnings=[]
            )
        
        # Check 2: Circuit Breaker
        # Update circuit breaker state if equity provided
        if current_equity is not None:
            self.circuit_breaker.update(current_equity, volatility)
        
        if self.circuit_breaker.is_tripped():
            status = self.circuit_breaker.get_status()
            reasons = []
            if status["daily_loss_tripped"]:
                reasons.append("daily loss limit exceeded")
            if status["drawdown_tripped"]:
                reasons.append("drawdown limit exceeded")
            if status["volatility_tripped"]:
                reasons.append("volatility spike detected")
            
            reason_str = ", ".join(reasons)
            LOGGER.critical(f"Trade rejected for {symbol}: CIRCUIT BREAKER TRIPPED ({reason_str})")
            return TradeValidationResult(
                allowed=False,
                reason=f"CIRCUIT BREAKER TRIPPED - Trading halted: {reason_str}",
                failed_checks=["circuit_breaker"],
                warnings=[]
            )
        
        # Check 3: Asset Limits
        allowed, reason = self.asset_limits.check_trade(
            symbol=symbol,
            quantity=signed_quantity,
            price=price,
            portfolio_value=portfolio_value,
            current_positions=current_positions,
            asset_class=asset_class,
        )
        
        if not allowed:
            LOGGER.warning(f"Trade rejected for {symbol}: Asset limit violation - {reason}")
            failed_checks.append("asset_limits")
            return TradeValidationResult(
                allowed=False,
                reason=f"Asset limit violation: {reason}",
                failed_checks=failed_checks,
                warnings=warnings
            )
        
        # Check 4: Correlation Constraints (only for new positions or increases)
        if side.lower() == "buy":
            # Get list of current holdings (symbols with non-zero positions)
            current_holdings = [sym for sym, notional in current_positions.items() 
                              if notional > 0 and sym != symbol]
            
            if check_portfolio_correlation(symbol, current_holdings, self.correlation_threshold):
                # Find which symbols are highly correlated
                correlated_symbols = []
                for holding in current_holdings:
                    corr = get_correlation(symbol, holding)
                    if abs(corr) >= self.correlation_threshold:
                        correlated_symbols.append(f"{holding} ({corr:+.2f})")
                
                corr_str = ", ".join(correlated_symbols)
                warning_msg = f"High correlation detected: {symbol} with {corr_str}"
                LOGGER.warning(f"Trade warning for {symbol}: {warning_msg}")
                warnings.append(warning_msg)
                
                # For now, we log a warning but don't block the trade
                # In a stricter regime, you could set allowed=False here
                # failed_checks.append("correlation")
                # return TradeValidationResult(
                #     allowed=False,
                #     reason=f"Correlation constraint violation: {warning_msg}",
                #     failed_checks=failed_checks,
                #     warnings=warnings
                # )
        
        # All checks passed
        LOGGER.debug(f"Trade validated for {symbol}: {quantity} @ {price} ({side})")
        return TradeValidationResult(
            allowed=True,
            reason="All risk checks passed",
            failed_checks=[],
            warnings=warnings
        )
    
    def validate_order_batch(
        self,
        orders: List[Tuple[str, float, float, str]],
        portfolio_value: float,
        current_positions: Dict[str, float],
        asset_class_map: Optional[Dict[str, str]] = None,
        current_equity: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> Tuple[List[Tuple[str, float, float, str]], List[Tuple[str, str]]]:
        """Validate a batch of orders and return allowed orders and rejections.
        
        Args:
            orders: List of (symbol, quantity, price, side) tuples
            portfolio_value: Total portfolio value
            current_positions: Dict mapping {symbol: notional_value}
            asset_class_map: Optional dict mapping {symbol: asset_class}
            current_equity: Optional current equity for circuit breaker
            volatility: Optional volatility measure for circuit breaker
            
        Returns:
            Tuple of (allowed_orders, rejected_orders)
            - allowed_orders: List of (symbol, quantity, price, side) that passed validation
            - rejected_orders: List of (symbol, reason) for rejected trades
        """
        allowed_orders = []
        rejected_orders = []
        
        # Create a simulated position state for sequential validation
        simulated_positions = current_positions.copy()
        
        for symbol, quantity, price, side in orders:
            asset_class = asset_class_map.get(symbol) if asset_class_map else None
            
            result = self.validate_trade(
                symbol=symbol,
                quantity=quantity,
                price=price,
                side=side,
                portfolio_value=portfolio_value,
                current_positions=simulated_positions,
                asset_class=asset_class,
                current_equity=current_equity,
                volatility=volatility,
            )
            
            if result.allowed:
                allowed_orders.append((symbol, quantity, price, side))
                
                # Update simulated positions for next order validation
                trade_notional = abs(quantity * price)
                current_notional = simulated_positions.get(symbol, 0.0)
                
                if side.lower() == "buy":
                    simulated_positions[symbol] = current_notional + trade_notional
                else:
                    simulated_positions[symbol] = max(0, current_notional - trade_notional)
            else:
                rejected_orders.append((symbol, result.reason))
                LOGGER.info(f"Order rejected in batch: {symbol} - {result.reason}")
        
        return allowed_orders, rejected_orders
    
    def get_validation_status(self) -> Dict[str, any]:
        """Get current status of all risk validators.
        
        Returns:
            Dict with status of kill switch, circuit breaker, and limits
        """
        return {
            "kill_switch_active": self.kill_switch.is_active(),
            "circuit_breaker": self.circuit_breaker.get_status(),
            "asset_limits": {
                "max_total_positions": self.asset_limits.config.max_total_positions,
                "max_leverage": self.asset_limits.config.max_leverage,
                "correlation_threshold": self.correlation_threshold,
            }
        }


# Global trade validator instance
TRADE_VALIDATOR = TradeValidator()
