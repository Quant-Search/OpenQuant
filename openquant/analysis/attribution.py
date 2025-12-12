"""Performance Attribution Analysis Module.

Decomposes returns into components:
- Timing Effect: Quality of entry/exit timing
- Selection Effect: Instrument/strategy choice contribution
- Sizing Effect: Position sizing contribution to returns
- Cost Drag: Impact of fees, slippage, and funding costs

Integrates with audit_trail for trade-level analysis.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import json

from ..storage.audit_trail import AuditTrail, EventType
from ..analysis.tca import TCAMonitor
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class AttributionResult:
    """Performance attribution breakdown for a period."""
    period_start: datetime
    period_end: datetime
    total_return: float
    timing_effect: float
    selection_effect: float
    sizing_effect: float
    cost_drag: float
    residual: float
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result['period_start'] = self.period_start.isoformat()
        result['period_end'] = self.period_end.isoformat()
        return result
        
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Performance Attribution ({self.period_start.date()} to {self.period_end.date()})",
            f"Total Return: {self.total_return:+.2%}",
            f"  Timing Effect: {self.timing_effect:+.2%}",
            f"  Selection Effect: {self.selection_effect:+.2%}",
            f"  Sizing Effect: {self.sizing_effect:+.2%}",
            f"  Cost Drag: {self.cost_drag:+.2%}",
            f"  Residual: {self.residual:+.2%}"
        ]
        return "\n".join(lines)


@dataclass
class TradeAttribution:
    """Attribution for a single trade."""
    symbol: str
    strategy: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: str
    pnl: float
    pnl_pct: float
    timing_quality: float
    execution_quality: float
    cost_impact: float
    holding_period_hours: float
    
    
class PerformanceAttributor:
    """
    Comprehensive performance attribution engine.
    
    Analyzes trading performance by decomposing returns into:
    1. Timing Effect: How well entries and exits were timed relative to price extremes
    2. Selection Effect: How different instruments/strategies contributed to returns
    3. Sizing Effect: How position sizing decisions impacted returns
    4. Cost Drag: Impact of transaction costs (fees, slippage, funding)
    
    Usage:
        attributor = PerformanceAttributor()
        
        # Analyze last 30 days
        result = attributor.analyze(days=30)
        print(result.summary())
        
        # Get detailed trade-level attribution
        trade_attrs = attributor.get_trade_level_attribution(days=7)
        
        # Get strategy comparison
        strategy_perf = attributor.compare_strategies(days=30)
    """
    
    def __init__(
        self,
        audit_trail: Optional[AuditTrail] = None,
        tca_monitor: Optional[TCAMonitor] = None
    ):
        """Initialize with audit trail and TCA monitor."""
        self.audit_trail = audit_trail or AuditTrail()
        self.tca_monitor = tca_monitor or TCAMonitor()
        
    def analyze(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        days: int = 30
    ) -> AttributionResult:
        """
        Perform comprehensive performance attribution.
        
        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            days: Number of days to analyze if start_time not provided
            
        Returns:
            AttributionResult with decomposed performance
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=days)
            
        LOGGER.info(f"Computing attribution from {start_time} to {end_time}")
        
        trades = self._get_completed_trades(start_time, end_time)
        
        if not trades:
            return AttributionResult(
                period_start=start_time,
                period_end=end_time,
                total_return=0.0,
                timing_effect=0.0,
                selection_effect=0.0,
                sizing_effect=0.0,
                cost_drag=0.0,
                residual=0.0,
                details={"message": "No trades in period"}
            )
            
        total_return = self._compute_total_return(trades)
        timing_effect = self._compute_timing_effect(trades, start_time, end_time)
        selection_effect = self._compute_selection_effect(trades)
        sizing_effect = self._compute_sizing_effect(trades)
        cost_drag = self._compute_cost_drag(trades, start_time, end_time)
        
        residual = total_return - (timing_effect + selection_effect + sizing_effect + cost_drag)
        
        details = {
            "num_trades": len(trades),
            "symbols_traded": list(set(t.get('symbol', '') for t in trades)),
            "strategies_used": list(set(t.get('strategy', '') for t in trades)),
            "total_volume": sum(abs(t.get('quantity', 0) * t.get('exit_price', 0)) for t in trades)
        }
        
        result = AttributionResult(
            period_start=start_time,
            period_end=end_time,
            total_return=total_return,
            timing_effect=timing_effect,
            selection_effect=selection_effect,
            sizing_effect=sizing_effect,
            cost_drag=cost_drag,
            residual=residual,
            details=details
        )
        
        LOGGER.info(f"Attribution complete: Total return {total_return:.2%}")
        return result
        
    def _get_completed_trades(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> List[Dict[str, Any]]:
        """
        Extract completed trades from audit trail.
        
        A completed trade is a pair of entry (ORDER_EXECUTION buy) and exit (ORDER_EXECUTION sell).
        """
        executions = self.audit_trail.query(
            event_type=EventType.ORDER_EXECUTION,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        if not executions:
            return []
            
        df = pd.DataFrame(executions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        trades = []
        positions: Dict[Tuple[str, str], Dict] = {}
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            strategy = row['strategy'] or 'unknown'
            side = row['side']
            quantity = row['quantity']
            price = row['price']
            timestamp = row['timestamp']
            
            key = (symbol, strategy)
            
            if side in ['BUY', 'buy']:
                if key not in positions or positions[key]['quantity'] <= 0:
                    positions[key] = {
                        'entry_time': timestamp,
                        'entry_price': price,
                        'quantity': quantity,
                        'symbol': symbol,
                        'strategy': strategy,
                        'side': 'LONG'
                    }
                else:
                    existing = positions[key]
                    total_qty = existing['quantity'] + quantity
                    avg_price = (existing['entry_price'] * existing['quantity'] + 
                                price * quantity) / total_qty
                    existing['entry_price'] = avg_price
                    existing['quantity'] = total_qty
                    existing['entry_time'] = timestamp
                    
            elif side in ['SELL', 'sell']:
                if key in positions and positions[key]['quantity'] > 0:
                    pos = positions[key]
                    exit_qty = min(quantity, pos['quantity'])
                    
                    pnl = (price - pos['entry_price']) * exit_qty
                    pnl_pct = (price - pos['entry_price']) / pos['entry_price']
                    
                    trade = {
                        'symbol': symbol,
                        'strategy': strategy,
                        'entry_time': pos['entry_time'],
                        'exit_time': timestamp,
                        'entry_price': pos['entry_price'],
                        'exit_price': price,
                        'quantity': exit_qty,
                        'side': 'LONG',
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'holding_period': (timestamp - pos['entry_time']).total_seconds() / 3600
                    }
                    trades.append(trade)
                    
                    pos['quantity'] -= exit_qty
                    if pos['quantity'] <= 1e-9:
                        del positions[key]
                else:
                    if key not in positions:
                        positions[key] = {
                            'entry_time': timestamp,
                            'entry_price': price,
                            'quantity': -quantity,
                            'symbol': symbol,
                            'strategy': strategy,
                            'side': 'SHORT'
                        }
                    else:
                        existing = positions[key]
                        total_qty = existing['quantity'] - quantity
                        avg_price = (existing['entry_price'] * abs(existing['quantity']) + 
                                    price * quantity) / abs(total_qty)
                        existing['entry_price'] = avg_price
                        existing['quantity'] = total_qty
                        existing['entry_time'] = timestamp
                        
        return trades
        
    def _compute_total_return(self, trades: List[Dict[str, Any]]) -> float:
        """Compute total return from all trades."""
        if not trades:
            return 0.0
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        total_notional = sum(abs(t.get('entry_price', 0) * t.get('quantity', 0)) for t in trades)
        return total_pnl / total_notional if total_notional > 0 else 0.0
        
    def _compute_timing_effect(
        self,
        trades: List[Dict[str, Any]],
        start_time: datetime,
        end_time: datetime
    ) -> float:
        """
        Compute timing effect: how well entries and exits were timed.
        
        Measures how close entries were to period lows and exits to period highs.
        Positive timing effect means buying near lows and selling near highs.
        """
        if not trades:
            return 0.0
            
        timing_scores = []
        
        for trade in trades:
            symbol = trade.get('symbol')
            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            
            signals = self.audit_trail.query(
                event_type=EventType.SIGNAL,
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            if not signals:
                continue
                
            prices = [s.get('price', 0) for s in signals if s.get('price')]
            if not prices:
                continue
                
            price_min = min(prices)
            price_max = max(prices)
            price_range = price_max - price_min
            
            if price_range > 0:
                entry_timing = 1.0 - (entry_price - price_min) / price_range
                exit_timing = (exit_price - price_min) / price_range
                
                timing_score = (entry_timing + exit_timing) / 2.0
                timing_scores.append(timing_score)
                
        if not timing_scores:
            return 0.0
            
        avg_timing = np.mean(timing_scores)
        return (avg_timing - 0.5) * 0.1
        
    def _compute_selection_effect(self, trades: List[Dict[str, Any]]) -> float:
        """
        Compute selection effect: contribution from choosing instruments/strategies.
        
        Measures how strategy and instrument choices contributed to overall returns
        by comparing individual performance to equal-weighted average.
        """
        if not trades:
            return 0.0
            
        df = pd.DataFrame(trades)
        
        by_strategy = df.groupby('strategy').agg({
            'pnl': 'sum',
            'quantity': 'count'
        }).reset_index()
        
        by_strategy['pnl_per_trade'] = by_strategy['pnl'] / by_strategy['quantity']
        
        avg_pnl_per_trade = df['pnl'].sum() / len(df)
        
        weighted_sum = 0.0
        for _, row in by_strategy.iterrows():
            deviation = row['pnl_per_trade'] - avg_pnl_per_trade
            weight = row['quantity'] / len(df)
            weighted_sum += deviation * weight
            
        total_notional = sum(abs(t.get('entry_price', 0) * t.get('quantity', 0)) for t in trades)
        return weighted_sum / total_notional if total_notional > 0 else 0.0
        
    def _compute_sizing_effect(self, trades: List[Dict[str, Any]]) -> float:
        """
        Compute sizing effect: contribution from position sizing decisions.
        
        Measures how position sizing contributed to returns by comparing
        actual returns to equal-sized returns.
        """
        if not trades:
            return 0.0
            
        df = pd.DataFrame(trades)
        df['notional'] = df['entry_price'] * df['quantity'].abs()
        
        total_notional = df['notional'].sum()
        if total_notional == 0:
            return 0.0
            
        df['weight'] = df['notional'] / total_notional
        
        actual_return = (df['pnl'] * df['weight']).sum() / total_notional
        
        equal_weight = 1.0 / len(df)
        equal_weighted_return = (df['pnl'] * equal_weight).sum() / total_notional
        
        sizing_effect = actual_return - equal_weighted_return
        
        return sizing_effect
        
    def _compute_cost_drag(
        self,
        trades: List[Dict[str, Any]],
        start_time: datetime,
        end_time: datetime
    ) -> float:
        """
        Compute cost drag from fees, slippage, and funding.
        
        Aggregates all transaction costs and expresses as percentage of total notional.
        """
        try:
            tca_stats = self.tca_monitor.get_stats()
            
            total_fees = tca_stats.get('total_fees', 0.0)
            avg_slippage_bps = tca_stats.get('avg_slippage_bps', 0.0)
            
            total_notional = sum(abs(t.get('entry_price', 0) * t.get('quantity', 0)) for t in trades)
            
            if total_notional == 0:
                return 0.0
                
            fee_impact = -total_fees / total_notional
            
            slippage_impact = -avg_slippage_bps / 10000.0
            
            return fee_impact + slippage_impact
            
        except Exception as e:
            LOGGER.warning(f"Failed to compute cost drag: {e}")
            return -0.001
            
    def get_trade_level_attribution(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        days: int = 7
    ) -> List[TradeAttribution]:
        """
        Get detailed attribution for individual trades.
        
        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            days: Number of days to analyze if start_time not provided
            
        Returns:
            List of TradeAttribution objects
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=days)
            
        trades = self._get_completed_trades(start_time, end_time)
        
        attributions = []
        
        for trade in trades:
            symbol = trade.get('symbol', '')
            strategy = trade.get('strategy', 'unknown')
            
            signals = self.audit_trail.query(
                event_type=EventType.SIGNAL,
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                limit=1000
            )
            
            timing_quality = 0.0
            if signals:
                prices = [s.get('price', 0) for s in signals if s.get('price')]
                if prices:
                    price_min = min(prices)
                    price_max = max(prices)
                    price_range = price_max - price_min
                    
                    if price_range > 0:
                        entry_price = trade.get('entry_price', 0)
                        exit_price = trade.get('exit_price', 0)
                        entry_timing = 1.0 - (entry_price - price_min) / price_range
                        exit_timing = (exit_price - price_min) / price_range
                        timing_quality = (entry_timing + exit_timing) / 2.0
                        
            execution_quality = 1.0
            
            cost_impact = -0.001
            
            attr = TradeAttribution(
                symbol=symbol,
                strategy=strategy,
                entry_time=trade.get('entry_time'),
                exit_time=trade.get('exit_time'),
                entry_price=trade.get('entry_price', 0),
                exit_price=trade.get('exit_price', 0),
                quantity=trade.get('quantity', 0),
                side=trade.get('side', 'LONG'),
                pnl=trade.get('pnl', 0),
                pnl_pct=trade.get('pnl_pct', 0),
                timing_quality=timing_quality,
                execution_quality=execution_quality,
                cost_impact=cost_impact,
                holding_period_hours=trade.get('holding_period', 0)
            )
            attributions.append(attr)
            
        return attributions
        
    def compare_strategies(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        days: int = 30
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance attribution across strategies.
        
        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            days: Number of days to analyze if start_time not provided
            
        Returns:
            Dictionary mapping strategy names to attribution metrics
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=days)
            
        trades = self._get_completed_trades(start_time, end_time)
        
        if not trades:
            return {}
            
        df = pd.DataFrame(trades)
        
        strategy_metrics = {}
        
        for strategy in df['strategy'].unique():
            strategy_trades = [t for t in trades if t.get('strategy') == strategy]
            
            if not strategy_trades:
                continue
                
            total_return = self._compute_total_return(strategy_trades)
            timing_effect = self._compute_timing_effect(strategy_trades, start_time, end_time)
            sizing_effect = self._compute_sizing_effect(strategy_trades)
            cost_drag = self._compute_cost_drag(strategy_trades, start_time, end_time)
            
            num_trades = len(strategy_trades)
            win_rate = sum(1 for t in strategy_trades if t.get('pnl', 0) > 0) / num_trades
            avg_pnl = sum(t.get('pnl', 0) for t in strategy_trades) / num_trades
            
            strategy_metrics[strategy] = {
                'total_return': total_return,
                'timing_effect': timing_effect,
                'sizing_effect': sizing_effect,
                'cost_drag': cost_drag,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': sum(t.get('pnl', 0) for t in strategy_trades)
            }
            
        return strategy_metrics
        
    def compare_instruments(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        days: int = 30
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare performance attribution across instruments.
        
        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            days: Number of days to analyze if start_time not provided
            
        Returns:
            Dictionary mapping symbols to attribution metrics
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=days)
            
        trades = self._get_completed_trades(start_time, end_time)
        
        if not trades:
            return {}
            
        df = pd.DataFrame(trades)
        
        instrument_metrics = {}
        
        for symbol in df['symbol'].unique():
            symbol_trades = [t for t in trades if t.get('symbol') == symbol]
            
            if not symbol_trades:
                continue
                
            total_return = self._compute_total_return(symbol_trades)
            timing_effect = self._compute_timing_effect(symbol_trades, start_time, end_time)
            
            num_trades = len(symbol_trades)
            win_rate = sum(1 for t in symbol_trades if t.get('pnl', 0) > 0) / num_trades
            avg_holding = sum(t.get('holding_period', 0) for t in symbol_trades) / num_trades
            
            instrument_metrics[symbol] = {
                'total_return': total_return,
                'timing_effect': timing_effect,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'avg_holding_hours': avg_holding,
                'total_pnl': sum(t.get('pnl', 0) for t in symbol_trades)
            }
            
        return instrument_metrics
        
    def generate_report(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        days: int = 30,
        output_path: str = "data/attribution_report.json"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive attribution report and save to file.
        
        Args:
            start_time: Start of analysis period
            end_time: End of analysis period
            days: Number of days to analyze if start_time not provided
            output_path: Path to save JSON report
            
        Returns:
            Complete report dictionary
        """
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=days)
            
        LOGGER.info("Generating comprehensive attribution report...")
        
        overall = self.analyze(start_time, end_time)
        
        strategy_comparison = self.compare_strategies(start_time, end_time)
        
        instrument_comparison = self.compare_instruments(start_time, end_time)
        
        trade_level = self.get_trade_level_attribution(start_time, end_time)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'days': days
            },
            'overall_attribution': overall.to_dict(),
            'strategy_comparison': strategy_comparison,
            'instrument_comparison': instrument_comparison,
            'trade_level_sample': [
                {
                    'symbol': t.symbol,
                    'strategy': t.strategy,
                    'pnl_pct': t.pnl_pct,
                    'timing_quality': t.timing_quality,
                    'holding_period_hours': t.holding_period_hours
                }
                for t in trade_level[:10]
            ],
            'summary': {
                'total_trades': len(trade_level),
                'profitable_trades': sum(1 for t in trade_level if t.pnl > 0),
                'average_timing_quality': np.mean([t.timing_quality for t in trade_level]) if trade_level else 0,
                'average_holding_hours': np.mean([t.holding_period_hours for t in trade_level]) if trade_level else 0
            }
        }
        
        try:
            from pathlib import Path
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
                
            LOGGER.info(f"Attribution report saved to {output_path}")
            
        except Exception as e:
            LOGGER.error(f"Failed to save attribution report: {e}")
            
        return report


def quick_attribution(days: int = 30) -> AttributionResult:
    """
    Convenience function for quick attribution analysis.
    
    Args:
        days: Number of days to analyze
        
    Returns:
        AttributionResult
    """
    attributor = PerformanceAttributor()
    result = attributor.analyze(days=days)
    print(result.summary())
    return result
