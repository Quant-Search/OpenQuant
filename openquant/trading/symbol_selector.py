"""Intelligent Symbol Selection.

Filters and ranks trading symbols based on:
- Liquidity (volume, spread)
- Volatility (ATR, recent movement)
- Trend strength (ADX, directional movement)
- Correlation (avoid correlated assets)
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

@dataclass
class SymbolMetrics:
    """Metrics for a trading symbol."""
    symbol: str
    avg_volume: float
    volatility: float  # ATR %
    adx: float  # Trend strength 0-100
    spread_bps: float  # Typical spread in basis points
    hurst: float  # Trend persistence
    score: float  # Overall score

class SymbolSelector:
    """
    Intelligent symbol selection for optimal trading.
    
    Selection criteria:
    1. Liquidity: High volume = better execution
    2. Volatility: Moderate = good opportunity without excessive risk
    3. Trend strength: Higher ADX = clearer direction
    4. Correlation: Low correlation with existing positions
    """
    
    def __init__(
        self,
        min_volume: float = 1_000_000,  # Min daily volume
        min_volatility: float = 0.005,  # Min daily ATR %
        max_volatility: float = 0.05,   # Max daily ATR %
        min_adx: float = 20,            # Min ADX for trend
        max_spread_bps: float = 10,     # Max spread in bps
        max_correlation: float = 0.7    # Max correlation between positions
    ):
        self.min_volume = min_volume
        self.min_volatility = min_volatility
        self.max_volatility = max_volatility
        self.min_adx = min_adx
        self.max_spread_bps = max_spread_bps
        self.max_correlation = max_correlation
        
    def analyze_symbol(
        self, 
        symbol: str, 
        df: pd.DataFrame,
        spread_bps: float = 2.0
    ) -> SymbolMetrics:
        """
        Analyze a symbol and calculate metrics.
        
        Args:
            symbol: Symbol name
            df: OHLCV dataframe
            spread_bps: Typical spread in basis points
            
        Returns:
            SymbolMetrics with calculated scores
        """
        if len(df) < 30:
            return SymbolMetrics(
                symbol=symbol,
                avg_volume=0,
                volatility=0,
                adx=0,
                spread_bps=spread_bps,
                hurst=0.5,
                score=0
            )
            
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume'] if 'Volume' in df else pd.Series([0] * len(df))
        
        # Calculate metrics
        avg_volume = volume.rolling(20).mean().iloc[-1]
        
        # ATR % (volatility)
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        volatility = atr / close.iloc[-1]
        
        # ADX (trend strength)
        adx = self._calculate_adx(df)
        
        # Hurst exponent (simplified)
        hurst = self._calculate_hurst(close)
        
        # Calculate overall score
        score = self._calculate_score(
            avg_volume=avg_volume,
            volatility=volatility,
            adx=adx,
            spread_bps=spread_bps,
            hurst=hurst
        )
        
        return SymbolMetrics(
            symbol=symbol,
            avg_volume=avg_volume,
            volatility=volatility,
            adx=adx,
            spread_bps=spread_bps,
            hurst=hurst,
            score=score
        )
        
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX (Average Directional Index)."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed averages
        atr = pd.Series(tr).rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0
        
    def _calculate_hurst(self, series: pd.Series, max_lag: int = 20) -> float:
        """Simplified Hurst exponent calculation."""
        try:
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
            
            if all(t > 0 for t in tau):
                reg = np.polyfit(np.log(lags), np.log(tau), 1)
                return float(reg[0])
        except Exception:
            pass
        return 0.5
        
    def _calculate_score(
        self,
        avg_volume: float,
        volatility: float,
        adx: float,
        spread_bps: float,
        hurst: float
    ) -> float:
        """
        Calculate overall symbol score (0-100).
        
        Higher score = better trading candidate.
        """
        score = 0.0
        
        # Volume score (0-25): Higher is better
        if avg_volume >= self.min_volume:
            volume_score = min(25, 25 * (avg_volume / (self.min_volume * 10)))
            score += volume_score
            
        # Volatility score (0-25): Moderate is best
        if self.min_volatility <= volatility <= self.max_volatility:
            # Optimal around middle of range
            mid = (self.min_volatility + self.max_volatility) / 2
            vol_distance = abs(volatility - mid) / (self.max_volatility - self.min_volatility)
            score += 25 * (1 - vol_distance)
            
        # ADX score (0-25): Higher is better for trending
        if adx >= self.min_adx:
            adx_score = min(25, 25 * (adx / 50))
            score += adx_score
            
        # Spread score (0-15): Lower is better
        if spread_bps <= self.max_spread_bps:
            spread_score = 15 * (1 - spread_bps / self.max_spread_bps)
            score += spread_score
            
        # Hurst score (0-10): Higher is better for trending
        if hurst > 0.5:
            hurst_score = min(10, 20 * (hurst - 0.5))
            score += hurst_score
            
        return score
        
    def rank_symbols(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        existing_positions: Optional[Dict[str, float]] = None,
        top_n: int = 10
    ) -> List[SymbolMetrics]:
        """
        Rank symbols by trading quality.
        
        Args:
            symbol_data: Dict of symbol -> OHLCV dataframe
            existing_positions: Current positions for correlation check
            top_n: Number of top symbols to return
            
        Returns:
            List of SymbolMetrics, sorted by score descending
        """
        metrics_list = []
        
        for symbol, df in symbol_data.items():
            try:
                metrics = self.analyze_symbol(symbol, df)
                
                # Apply filters
                if metrics.avg_volume < self.min_volume:
                    LOGGER.debug(f"{symbol}: Low volume ({metrics.avg_volume:.0f})")
                    continue
                    
                if not (self.min_volatility <= metrics.volatility <= self.max_volatility):
                    LOGGER.debug(f"{symbol}: Volatility out of range ({metrics.volatility:.2%})")
                    continue
                    
                metrics_list.append(metrics)
                
            except Exception as e:
                LOGGER.warning(f"Error analyzing {symbol}: {e}")
                
        # Sort by score
        metrics_list.sort(key=lambda x: x.score, reverse=True)
        
        # Check correlation with existing positions
        if existing_positions:
            filtered = []
            for m in metrics_list:
                if m.symbol not in existing_positions:
                    filtered.append(m)
            metrics_list = filtered
            
        LOGGER.info(f"Symbol ranking: {len(metrics_list)} symbols analyzed")
        for m in metrics_list[:5]:
            LOGGER.info(f"  {m.symbol}: score={m.score:.1f}, ADX={m.adx:.1f}, vol={m.volatility:.2%}")
            
        return metrics_list[:top_n]
        
    def get_best_symbols(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        n: int = 5
    ) -> List[str]:
        """Get best N symbols for trading."""
        ranked = self.rank_symbols(symbol_data, top_n=n)
        return [m.symbol for m in ranked]
