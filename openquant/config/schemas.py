"""Pydantic schemas for configuration validation."""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class TradingSessionConfig(BaseModel):
    """Trading session configuration."""
    start: int = Field(ge=0, le=23, description="Session start hour (UTC)")
    end: int = Field(ge=0, le=24, description="Session end hour (UTC)")

    @validator('end')
    def validate_end_after_start(cls, v, values):
        if 'start' in values and v < values['start']:
            raise ValueError('end must be after start')
        return v


class ForexSymbolConfig(BaseModel):
    """Forex symbol configuration."""
    spread_bps: float = Field(gt=0, description="Spread in basis points")
    swap_long: float = Field(description="Daily swap cost for long positions (pips)")
    swap_short: float = Field(description="Daily swap cost for short positions (pips)")
    optimal_sessions: List[str] = Field(description="List of optimal trading sessions")
    pip_value: float = Field(gt=0, description="Value of 1 pip")


class ForexConfig(BaseModel):
    """Forex trading configuration."""
    symbols: Dict[str, ForexSymbolConfig] = Field(description="Symbol-specific configurations")
    sessions: Dict[str, TradingSessionConfig] = Field(description="Trading session definitions")


class RiskLimitsConfig(BaseModel):
    """Risk management limits."""
    dd_limit: float = Field(gt=0, le=1, description="Maximum drawdown limit (fraction)")
    daily_loss_cap: float = Field(gt=0, le=1, description="Daily loss cap (fraction)")
    cvar_limit: float = Field(gt=0, le=1, description="CVaR limit (fraction)")
    max_exposure_per_symbol: float = Field(gt=0, le=1, description="Max exposure per symbol (fraction)")


class CircuitBreakerConfig(BaseModel):
    """Circuit breaker configuration."""
    daily_loss_limit: float = Field(default=0.02, gt=0, le=1, description="Daily loss limit (fraction)")
    drawdown_limit: float = Field(default=0.10, gt=0, le=1, description="Drawdown limit (fraction)")
    volatility_limit: float = Field(default=0.05, gt=0, description="Volatility spike threshold")


class BacktestConfig(BaseModel):
    """Backtest engine configuration."""
    fee_bps: float = Field(default=1.0, ge=0, description="Fee per trade in basis points")
    slippage_bps: float = Field(default=0.0, ge=0, description="Slippage in basis points")
    spread_bps: float = Field(default=0.0, ge=0, description="Bid-ask spread in basis points")
    leverage: float = Field(default=1.0, ge=1, description="Leverage multiplier")
    weight: float = Field(default=1.0, gt=0, le=1, description="Fraction of capital allocated")
    impact_coeff: float = Field(default=0.0, ge=0, description="Market impact coefficient")


class StrategyMixerConfig(BaseModel):
    """Strategy mixer configuration."""
    threshold: float = Field(default=0.2, ge=0, le=1, description="Signal threshold for long/short")
    equal_weights: bool = Field(default=True, description="Use equal weights for strategies")


class AdaptiveSizingConfig(BaseModel):
    """Adaptive position sizing configuration."""
    method: str = Field(default="volatility", description="Sizing method: kelly or volatility")
    target_risk: float = Field(default=0.01, gt=0, le=1, description="Target risk per trade")
    max_drawdown: float = Field(default=0.50, gt=0, le=1, description="Maximum drawdown threshold")
    aggressive_mode: bool = Field(default=False, description="Enable aggressive sizing")
    target_volatility: float = Field(default=0.20, gt=0, description="Target annualized volatility")
    max_leverage: float = Field(default=1.0, ge=1, description="Maximum leverage")


class StationarityConfig(BaseModel):
    """Stationarity testing configuration."""
    adf_threshold: float = Field(default=0.05, gt=0, le=1, description="ADF test p-value threshold")
    kpss_threshold: float = Field(default=0.05, gt=0, le=1, description="KPSS test p-value threshold")
    hurst_mean_reverting: float = Field(default=0.45, gt=0, lt=0.5, description="Hurst threshold for mean-reverting")
    hurst_trending: float = Field(default=0.55, gt=0.5, lt=1, description="Hurst threshold for trending")
    hurst_high_confidence: float = Field(default=0.15, gt=0, description="Hurst deviation for high confidence")


class ConcentrationLimitsConfig(BaseModel):
    """Concentration limits for portfolio."""
    max_per_symbol: Optional[int] = Field(default=None, ge=1, description="Max configs per symbol")
    max_per_strategy_per_symbol: Optional[int] = Field(default=None, ge=1, description="Max configs per strategy-symbol pair")


class PaperTradingConfig(BaseModel):
    """Paper trading simulator configuration."""
    fee_bps: float = Field(default=0.0, ge=0, description="Fee per trade in basis points")
    slippage_bps: float = Field(default=0.0, ge=0, description="Slippage in basis points")
    next_bar_fill: bool = Field(default=False, description="Fill orders at next bar open")
    max_fill_fraction: float = Field(default=1.0, gt=0, le=1, description="Maximum fill fraction per order")
    daily_loss_limit: float = Field(default=0.05, gt=0, le=1, description="Daily loss limit (fraction)")


class Config(BaseModel):
    """Main configuration schema."""
    forex: Optional[ForexConfig] = None
    risk_limits: RiskLimitsConfig = Field(default_factory=RiskLimitsConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    strategy_mixer: StrategyMixerConfig = Field(default_factory=StrategyMixerConfig)
    adaptive_sizing: AdaptiveSizingConfig = Field(default_factory=AdaptiveSizingConfig)
    stationarity: StationarityConfig = Field(default_factory=StationarityConfig)
    concentration_limits: ConcentrationLimitsConfig = Field(default_factory=ConcentrationLimitsConfig)
    paper_trading: PaperTradingConfig = Field(default_factory=PaperTradingConfig)
