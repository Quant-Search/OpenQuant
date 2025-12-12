"""Performance Attribution Module.

Decomposes strategy returns into timing, selection, and sizing effects
for post-trade analysis using AUDIT_TRAIL data.

Attribution Framework:
- Timing: Returns from being in/out of market at right times
- Selection: Returns from choosing the right assets
- Sizing: Returns from optimal position sizing
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from openquant.storage.audit_trail import AUDIT_TRAIL, EventType
from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class AttributionResult:
    """Results of performance attribution analysis."""
    total_return: float
    timing_effect: float
    selection_effect: float
    sizing_effect: float
    interaction_effect: float
    benchmark_return: float
    active_return: float
    information_ratio: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "timing_effect": self.timing_effect,
            "selection_effect": self.selection_effect,
            "sizing_effect": self.sizing_effect,
            "interaction_effect": self.interaction_effect,
            "benchmark_return": self.benchmark_return,
            "active_return": self.active_return,
            "information_ratio": self.information_ratio,
            "details": self.details
        }


@dataclass
class TradeAnalysis:
    """Analysis of a single trade."""
    symbol: str
    strategy: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    return_pct: float
    holding_period_days: float


class PerformanceAttribution:
    """
    Performance attribution analyzer.
    
    Decomposes returns into:
    1. Timing Effect: Profit from being in market vs out
    2. Selection Effect: Profit from choosing right assets
    3. Sizing Effect: Profit from position sizing decisions
    
    Usage:
        attr = PerformanceAttribution()
        
        # Analyze last 30 days
        result = attr.analyze(
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now()
        )
        
        print(f"Total Return: {result.total_return:.2%}")
        print(f"Timing Effect: {result.timing_effect:.2%}")
        print(f"Selection Effect: {result.selection_effect:.2%}")
        print(f"Sizing Effect: {result.sizing_effect:.2%}")
    """
    
    def __init__(self, audit_trail=None):
        """Initialize with optional audit trail instance."""
        self.audit_trail = audit_trail or AUDIT_TRAIL
        
    def analyze(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> AttributionResult:
        """
        Perform performance attribution analysis.
        
        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            strategy: Filter by strategy name
            symbol: Filter by symbol
            
        Returns:
            AttributionResult with decomposed returns
        """
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)
        if end_time is None:
            end_time = datetime.now()
            
        LOGGER.info(f"Running attribution analysis from {start_time} to {end_time}")
        
        trades = self._extract_trades(start_time, end_time, strategy, symbol)
        
        if not trades:
            LOGGER.warning("No trades found for attribution analysis")
            return self._empty_result()
            
        total_return = self._calculate_total_return(trades)
        
        timing_effect = self._calculate_timing_effect(trades, start_time, end_time)
        selection_effect = self._calculate_selection_effect(trades)
        sizing_effect = self._calculate_sizing_effect(trades)
        
        benchmark_return = self._calculate_benchmark_return(trades, start_time, end_time)
        active_return = total_return - benchmark_return
        
        interaction_effect = total_return - (timing_effect + selection_effect + sizing_effect)
        
        tracking_error = self._calculate_tracking_error(trades)
        information_ratio = active_return / tracking_error if tracking_error > 0 else 0.0
        
        details = self._generate_details(trades, start_time, end_time)
        
        return AttributionResult(
            total_return=total_return,
            timing_effect=timing_effect,
            selection_effect=selection_effect,
            sizing_effect=sizing_effect,
            interaction_effect=interaction_effect,
            benchmark_return=benchmark_return,
            active_return=active_return,
            information_ratio=information_ratio,
            details=details
        )
    
    def analyze_by_strategy(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, AttributionResult]:
        """Analyze attribution for each strategy separately."""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)
        if end_time is None:
            end_time = datetime.now()
            
        trades = self._extract_trades(start_time, end_time)
        
        strategies = set(t.strategy for t in trades)
        
        results = {}
        for strat in strategies:
            results[strat] = self.analyze(start_time, end_time, strategy=strat)
            
        return results
    
    def analyze_by_symbol(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, AttributionResult]:
        """Analyze attribution for each symbol separately."""
        if start_time is None:
            start_time = datetime.now() - timedelta(days=30)
        if end_time is None:
            end_time = datetime.now()
            
        trades = self._extract_trades(start_time, end_time)
        
        symbols = set(t.symbol for t in trades)
        
        results = {}
        for sym in symbols:
            results[sym] = self.analyze(start_time, end_time, symbol=sym)
            
        return results
    
    def get_attribution_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> str:
        """Generate a formatted attribution report."""
        result = self.analyze(start_time, end_time)
        
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE ATTRIBUTION REPORT")
        report.append("=" * 60)
        report.append(f"Period: {start_time} to {end_time}")
        report.append("")
        report.append(f"Total Return:        {result.total_return:>10.2%}")
        report.append(f"Benchmark Return:    {result.benchmark_return:>10.2%}")
        report.append(f"Active Return:       {result.active_return:>10.2%}")
        report.append("")
        report.append("Attribution Breakdown:")
        report.append(f"  Timing Effect:     {result.timing_effect:>10.2%}")
        report.append(f"  Selection Effect:  {result.selection_effect:>10.2%}")
        report.append(f"  Sizing Effect:     {result.sizing_effect:>10.2%}")
        report.append(f"  Interaction:       {result.interaction_effect:>10.2%}")
        report.append("")
        report.append(f"Information Ratio:   {result.information_ratio:>10.2f}")
        report.append("")
        report.append("Details:")
        for key, value in result.details.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.4f}")
            else:
                report.append(f"  {key}: {value}")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _extract_trades(
        self,
        start_time: datetime,
        end_time: datetime,
        strategy: Optional[str] = None,
        symbol: Optional[str] = None
    ) -> List[TradeAnalysis]:
        """Extract and match trades from audit trail."""
        signals = self.audit_trail.query(
            event_type=EventType.SIGNAL,
            strategy=strategy,
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        executions = self.audit_trail.query(
            event_type=EventType.ORDER_EXECUTION,
            strategy=strategy,
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        trades = self._match_trades(executions)
        
        return trades
    
    def _match_trades(self, executions: List[Dict[str, Any]]) -> List[TradeAnalysis]:
        """Match entry and exit executions into complete trades."""
        trades = []
        positions = {}
        
        for exec_event in executions:
            symbol = exec_event.get("symbol")
            strategy = exec_event.get("strategy")
            side = exec_event.get("side", "").upper()
            quantity = exec_event.get("quantity", 0)
            price = exec_event.get("price", 0)
            timestamp_str = exec_event.get("timestamp")
            
            if not all([symbol, strategy, side, timestamp_str]):
                continue
                
            timestamp = pd.to_datetime(timestamp_str)
            
            key = (symbol, strategy)
            
            if side in ["BUY", "LONG"]:
                if key not in positions:
                    positions[key] = {
                        "symbol": symbol,
                        "strategy": strategy,
                        "entry_time": timestamp,
                        "entry_price": price,
                        "quantity": quantity,
                        "side": "LONG"
                    }
            elif side in ["SELL", "SHORT"]:
                if key in positions and positions[key]["side"] == "LONG":
                    pos = positions[key]
                    holding_period = (timestamp - pos["entry_time"]).total_seconds() / 86400.0
                    return_pct = (price - pos["entry_price"]) / pos["entry_price"]
                    
                    trades.append(TradeAnalysis(
                        symbol=symbol,
                        strategy=strategy,
                        entry_time=pos["entry_time"],
                        exit_time=timestamp,
                        entry_price=pos["entry_price"],
                        exit_price=price,
                        quantity=pos["quantity"],
                        side="LONG",
                        return_pct=return_pct,
                        holding_period_days=holding_period
                    ))
                    
                    del positions[key]
                elif key not in positions:
                    positions[key] = {
                        "symbol": symbol,
                        "strategy": strategy,
                        "entry_time": timestamp,
                        "entry_price": price,
                        "quantity": quantity,
                        "side": "SHORT"
                    }
            elif side == "COVER":
                if key in positions and positions[key]["side"] == "SHORT":
                    pos = positions[key]
                    holding_period = (timestamp - pos["entry_time"]).total_seconds() / 86400.0
                    return_pct = (pos["entry_price"] - price) / pos["entry_price"]
                    
                    trades.append(TradeAnalysis(
                        symbol=symbol,
                        strategy=strategy,
                        entry_time=pos["entry_time"],
                        exit_time=timestamp,
                        entry_price=pos["entry_price"],
                        exit_price=price,
                        quantity=pos["quantity"],
                        side="SHORT",
                        return_pct=return_pct,
                        holding_period_days=holding_period
                    ))
                    
                    del positions[key]
        
        return trades
    
    def _calculate_total_return(self, trades: List[TradeAnalysis]) -> float:
        """Calculate total portfolio return."""
        if not trades:
            return 0.0
            
        total_weights = sum(abs(t.quantity * t.entry_price) for t in trades)
        
        if total_weights == 0:
            return 0.0
            
        weighted_return = sum(
            (abs(t.quantity * t.entry_price) / total_weights) * t.return_pct
            for t in trades
        )
        
        return weighted_return
    
    def _calculate_timing_effect(
        self,
        trades: List[TradeAnalysis],
        start_time: datetime,
        end_time: datetime
    ) -> float:
        """
        Calculate timing effect: returns from being in/out at right times.
        
        Compares actual returns during holding periods to hypothetical
        buy-and-hold returns.
        """
        if not trades:
            return 0.0
            
        total_days = (end_time - start_time).days
        if total_days == 0:
            return 0.0
            
        holding_days = sum(t.holding_period_days for t in trades)
        exposure_ratio = holding_days / (total_days * len(set((t.symbol, t.strategy) for t in trades)))
        
        if exposure_ratio > 1.0:
            exposure_ratio = 1.0
            
        avg_return_per_day = self._calculate_total_return(trades) / holding_days if holding_days > 0 else 0
        
        bah_return = avg_return_per_day * total_days
        
        timing_effect = self._calculate_total_return(trades) - (bah_return * exposure_ratio)
        
        return timing_effect
    
    def _calculate_selection_effect(self, trades: List[TradeAnalysis]) -> float:
        """
        Calculate selection effect: returns from choosing right assets.
        
        Measures alpha from security selection vs equal-weighted portfolio.
        """
        if not trades:
            return 0.0
            
        by_symbol = {}
        for trade in trades:
            if trade.symbol not in by_symbol:
                by_symbol[trade.symbol] = []
            by_symbol[trade.symbol].append(trade)
        
        equal_weight_return = sum(
            np.mean([t.return_pct for t in symbol_trades])
            for symbol_trades in by_symbol.values()
        ) / len(by_symbol)
        
        actual_return = self._calculate_total_return(trades)
        
        selection_effect = actual_return - equal_weight_return
        
        return selection_effect
    
    def _calculate_sizing_effect(self, trades: List[TradeAnalysis]) -> float:
        """
        Calculate sizing effect: returns from position sizing.
        
        Measures impact of varying position sizes vs equal sizing.
        """
        if not trades:
            return 0.0
            
        equal_sized_return = np.mean([t.return_pct for t in trades])
        
        actual_return = self._calculate_total_return(trades)
        
        sizing_effect = actual_return - equal_sized_return
        
        return sizing_effect
    
    def _calculate_benchmark_return(
        self,
        trades: List[TradeAnalysis],
        start_time: datetime,
        end_time: datetime
    ) -> float:
        """
        Calculate benchmark return.
        
        Uses equal-weighted buy-and-hold of all traded symbols.
        """
        if not trades:
            return 0.0
            
        by_symbol = {}
        for trade in trades:
            if trade.symbol not in by_symbol:
                by_symbol[trade.symbol] = []
            by_symbol[trade.symbol].append(trade)
        
        symbol_returns = []
        for symbol, symbol_trades in by_symbol.items():
            first_trade = min(symbol_trades, key=lambda t: t.entry_time)
            last_trade = max(symbol_trades, key=lambda t: t.exit_time)
            
            bah_return = (last_trade.exit_price - first_trade.entry_price) / first_trade.entry_price
            symbol_returns.append(bah_return)
        
        benchmark_return = np.mean(symbol_returns) if symbol_returns else 0.0
        
        return benchmark_return
    
    def _calculate_tracking_error(self, trades: List[TradeAnalysis]) -> float:
        """Calculate tracking error vs benchmark."""
        if len(trades) < 2:
            return 0.0
            
        returns = [t.return_pct for t in trades]
        
        return float(np.std(returns, ddof=1))
    
    def _generate_details(
        self,
        trades: List[TradeAnalysis],
        start_time: datetime,
        end_time: datetime
    ) -> Dict[str, Any]:
        """Generate detailed statistics."""
        if not trades:
            return {}
            
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.return_pct > 0]
        losing_trades = [t for t in trades if t.return_pct < 0]
        
        by_strategy = {}
        for trade in trades:
            if trade.strategy not in by_strategy:
                by_strategy[trade.strategy] = []
            by_strategy[trade.strategy].append(trade)
        
        by_symbol = {}
        for trade in trades:
            if trade.symbol not in by_symbol:
                by_symbol[trade.symbol] = []
            by_symbol[trade.symbol].append(trade)
        
        return {
            "total_trades": total_trades,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": len(winning_trades) / total_trades if total_trades > 0 else 0,
            "avg_return_per_trade": np.mean([t.return_pct for t in trades]),
            "avg_holding_period_days": np.mean([t.holding_period_days for t in trades]),
            "strategies_traded": list(by_strategy.keys()),
            "symbols_traded": list(by_symbol.keys()),
            "best_trade_return": max(t.return_pct for t in trades),
            "worst_trade_return": min(t.return_pct for t in trades),
            "strategy_returns": {
                strat: np.mean([t.return_pct for t in strat_trades])
                for strat, strat_trades in by_strategy.items()
            },
            "symbol_returns": {
                sym: np.mean([t.return_pct for t in sym_trades])
                for sym, sym_trades in by_symbol.items()
            }
        }
    
    def _empty_result(self) -> AttributionResult:
        """Return empty result when no data available."""
        return AttributionResult(
            total_return=0.0,
            timing_effect=0.0,
            selection_effect=0.0,
            sizing_effect=0.0,
            interaction_effect=0.0,
            benchmark_return=0.0,
            active_return=0.0,
            information_ratio=0.0,
            details={}
        )


def analyze_recent_performance(days: int = 30) -> AttributionResult:
    """
    Convenience function to analyze recent performance.
    
    Args:
        days: Number of days to look back
        
    Returns:
        AttributionResult for the period
    """
    attr = PerformanceAttribution()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    return attr.analyze(start_time=start_time, end_time=end_time)


def generate_attribution_report(days: int = 30) -> str:
    """
    Generate a formatted attribution report.
    
    Args:
        days: Number of days to look back
        
    Returns:
        Formatted report string
    """
    attr = PerformanceAttribution()
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    return attr.get_attribution_report(start_time=start_time, end_time=end_time)
