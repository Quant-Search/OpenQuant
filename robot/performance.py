"""
Performance Analytics Module
============================
Calculates key trading statistics and generates visualizations.

Key Metrics:
- Sharpe Ratio: > 1.0 good, > 2.0 excellent, > 3.0 exceptional
- Max Drawdown: < 10% conservative, < 20% moderate, < 30% aggressive
- Win Rate: Depends on R:R ratio (50% at 1:1, 33% at 2:1 is breakeven)
- Profit Factor: > 1.5 good, > 2.0 very good
- Expectancy: Should be positive (expected $ per trade)

Statistical Significance:
- Minimum 30 trades for basic statistics
- 100+ trades for robust backtesting
- p-value < 0.05 for significance
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TradeRecord:
    """Single trade record."""
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    volume: float
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class PerformanceMetrics:
    """Complete performance statistics."""
    # Returns
    total_return: float
    annualized_return: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    calmar_ratio: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Profit metrics
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float

    # Risk-adjusted
    recovery_factor: float
    risk_reward_ratio: float

    # Statistical
    is_statistically_significant: bool
    t_statistic: float
    p_value: float


class PerformanceAnalyzer:
    """Analyzes trading performance and generates statistics."""

    RISK_FREE_RATE = 0.04  # 4% annual risk-free rate
    TRADING_DAYS_PER_YEAR = 252

    def __init__(self, trades: List[TradeRecord] = None):
        self.trades = trades or []
        self.equity_curve: pd.Series = None

    def add_trade(self, trade: TradeRecord):
        """Add a completed trade."""
        self.trades.append(trade)

    def calculate_metrics(self, initial_capital: float = 10000) -> PerformanceMetrics:
        """Calculate all performance metrics."""
        if not self.trades:
            return self._empty_metrics()

        returns = self._calculate_returns()
        equity = self._build_equity_curve(initial_capital)

        # Basic stats
        total_trades = len(self.trades)
        pnls = [t.pnl for t in self.trades]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p < 0]

        win_rate = len(winning) / total_trades if total_trades > 0 else 0

        # Profit factor
        gross_profit = sum(winning) if winning else 0
        gross_loss = abs(sum(losing)) if losing else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Average win/loss
        avg_win = np.mean(winning) if winning else 0
        avg_loss = abs(np.mean(losing)) if losing else 0

        # Risk-reward ratio
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # Expectancy (expected value per trade)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        # Drawdown analysis
        max_dd, dd_duration = self._calculate_drawdown(equity)

        # Returns
        total_return = (equity.iloc[-1] - initial_capital) / initial_capital
        trading_days = len(equity)
        annualized_return = ((1 + total_return) ** (252 / max(trading_days, 1))) - 1

        # Risk metrics
        daily_returns = equity.pct_change().dropna()
        sharpe = self._calculate_sharpe(daily_returns)
        sortino = self._calculate_sortino(daily_returns)
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

        # Recovery factor
        recovery = total_return / abs(max_dd) if max_dd != 0 else 0

        # Statistical significance
        t_stat, p_val = self._calculate_significance(pnls)
        is_significant = p_val < 0.05 and total_trades >= 30

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            max_drawdown_duration=dd_duration,
            calmar_ratio=calmar,
            total_trades=total_trades,
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=max(winning) if winning else 0,
            largest_loss=abs(min(losing)) if losing else 0,
            recovery_factor=recovery,
            risk_reward_ratio=rr_ratio,
            is_statistically_significant=is_significant,
            t_statistic=t_stat,
            p_value=p_val
        )

    def _calculate_returns(self) -> pd.Series:
        """Calculate trade returns."""
        return pd.Series([t.pnl_pct for t in self.trades])

    def _build_equity_curve(self, initial: float) -> pd.Series:
        """Build equity curve from trades."""
        equity = [initial]
        for trade in self.trades:
            equity.append(equity[-1] + trade.pnl)
        self.equity_curve = pd.Series(equity)
        return self.equity_curve

    def _calculate_drawdown(self, equity: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min()

        # Duration calculation
        in_drawdown = drawdown < 0
        if not in_drawdown.any():
            return 0.0, 0

        # Find longest drawdown period
        dd_groups = (~in_drawdown).cumsum()
        dd_lengths = in_drawdown.groupby(dd_groups).sum()
        max_duration = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0

        return max_dd, max_duration

    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess_returns = returns - (self.RISK_FREE_RATE / 252)
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)

    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        if len(returns) < 2:
            return 0.0
        downside = returns[returns < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        excess_return = returns.mean() - (self.RISK_FREE_RATE / 252)
        return (excess_return / downside.std()) * np.sqrt(252)

    def _calculate_significance(self, pnls: List[float]) -> Tuple[float, float]:
        """Calculate t-statistic and p-value for trade returns."""
        if len(pnls) < 2:
            return 0.0, 1.0
        from scipy import stats
        t_stat, p_val = stats.ttest_1samp(pnls, 0)
        return float(t_stat), float(p_val)

    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty metrics when no trades."""
        return PerformanceMetrics(
            total_return=0, annualized_return=0, sharpe_ratio=0,
            sortino_ratio=0, max_drawdown=0, max_drawdown_duration=0,
            calmar_ratio=0, total_trades=0, winning_trades=0,
            losing_trades=0, win_rate=0, profit_factor=0, expectancy=0,
            avg_win=0, avg_loss=0, largest_win=0, largest_loss=0,
            recovery_factor=0, risk_reward_ratio=0,
            is_statistically_significant=False, t_statistic=0, p_value=1.0
        )


def evaluate_strategy_quality(metrics: PerformanceMetrics) -> Dict[str, str]:
    """
    Evaluate if a strategy is profitable based on metrics.

    Returns assessment for each metric with color coding:
    - "excellent" (green)
    - "good" (light green)
    - "acceptable" (yellow)
    - "poor" (red)
    """
    assessment = {}

    # Sharpe Ratio: Risk-adjusted returns
    # < 0: Losing money, 0-1: Subpar, 1-2: Good, 2-3: Very good, >3: Excellent
    if metrics.sharpe_ratio >= 3:
        assessment['sharpe'] = ('excellent', f'{metrics.sharpe_ratio:.2f}')
    elif metrics.sharpe_ratio >= 2:
        assessment['sharpe'] = ('good', f'{metrics.sharpe_ratio:.2f}')
    elif metrics.sharpe_ratio >= 1:
        assessment['sharpe'] = ('acceptable', f'{metrics.sharpe_ratio:.2f}')
    else:
        assessment['sharpe'] = ('poor', f'{metrics.sharpe_ratio:.2f}')

    # Max Drawdown: Capital preservation
    # < 10%: Conservative, 10-20%: Moderate, 20-30%: Aggressive, >30%: Dangerous
    dd_pct = abs(metrics.max_drawdown) * 100
    if dd_pct <= 10:
        assessment['drawdown'] = ('excellent', f'{dd_pct:.1f}%')
    elif dd_pct <= 20:
        assessment['drawdown'] = ('good', f'{dd_pct:.1f}%')
    elif dd_pct <= 30:
        assessment['drawdown'] = ('acceptable', f'{dd_pct:.1f}%')
    else:
        assessment['drawdown'] = ('poor', f'{dd_pct:.1f}%')

    # Profit Factor: Gross profit / Gross loss
    # < 1: Losing, 1-1.5: Marginal, 1.5-2: Good, >2: Excellent
    if metrics.profit_factor >= 2:
        assessment['profit_factor'] = ('excellent', f'{metrics.profit_factor:.2f}')
    elif metrics.profit_factor >= 1.5:
        assessment['profit_factor'] = ('good', f'{metrics.profit_factor:.2f}')
    elif metrics.profit_factor >= 1:
        assessment['profit_factor'] = ('acceptable', f'{metrics.profit_factor:.2f}')
    else:
        assessment['profit_factor'] = ('poor', f'{metrics.profit_factor:.2f}')

    # Win Rate (context-dependent based on R:R)
    # At 1:1 R:R need >50%, at 2:1 R:R need >33%
    breakeven_wr = 1 / (1 + metrics.risk_reward_ratio) if metrics.risk_reward_ratio > 0 else 0.5
    wr_margin = metrics.win_rate - breakeven_wr
    if wr_margin >= 0.15:
        assessment['win_rate'] = ('excellent', f'{metrics.win_rate:.1%}')
    elif wr_margin >= 0.05:
        assessment['win_rate'] = ('good', f'{metrics.win_rate:.1%}')
    elif wr_margin >= 0:
        assessment['win_rate'] = ('acceptable', f'{metrics.win_rate:.1%}')
    else:
        assessment['win_rate'] = ('poor', f'{metrics.win_rate:.1%}')

    # Statistical Significance
    if metrics.is_statistically_significant:
        assessment['significance'] = ('excellent', f'p={metrics.p_value:.4f}')
    elif metrics.total_trades >= 30:
        assessment['significance'] = ('acceptable', f'p={metrics.p_value:.4f}')
    else:
        assessment['significance'] = ('poor', f'Need {30 - metrics.total_trades} more trades')

    return assessment

