"""
Backtester Module - Evaluates strategy profitability on historical data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from robot.strategy import KalmanStrategy, BaseStrategy
from robot.performance import PerformanceAnalyzer, TradeRecord, PerformanceMetrics


@dataclass
class BacktestTrade:
    """Single backtest trade."""
    entry_time: datetime
    entry_price: float
    direction: str
    size: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    trades: List[BacktestTrade]
    equity_curve: pd.Series
    metrics: PerformanceMetrics
    signals: pd.Series
    prices: pd.Series


class Backtester:
    """Backtests a strategy on historical data."""

    def __init__(
        self,
        strategy: BaseStrategy = None,
        initial_capital: float = 10000,
        position_size_pct: float = 0.02,
        slippage_pct: float = 0.0001,
        commission_pct: float = 0.0001,
        stop_loss_atr: float = 2.0,
        take_profit_atr: float = 3.0
    ):
        self.strategy = strategy or KalmanStrategy()
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
        self._reset()

    def _reset(self):
        self.trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.current_position: Optional[BacktestTrade] = None
        self.cash = self.initial_capital

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df['High'], df['Low'], df['Close']
        tr = pd.concat([high - low, abs(high - close.shift(1)),
                        abs(low - close.shift(1))], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def run(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> BacktestResult:
        """Run backtest on historical OHLCV data."""
        self._reset()
        if len(df) < 100:
            raise ValueError("Need at least 100 bars")

        atr = self._calculate_atr(df)
        signals = self.strategy.generate_signals(df)
        prices = df['Close']
        self.equity_curve = [self.initial_capital]

        for i in range(50, len(df)):
            price = float(prices.iloc[i])
            signal = int(signals.iloc[i])
            curr_atr = float(atr.iloc[i]) if i < len(atr) else float(atr.iloc[-1])
            time = df.index[i]
            high, low = float(df['High'].iloc[i]), float(df['Low'].iloc[i])

            if self.current_position:
                exit_reason = self._check_exits(self.current_position, high, low)
                if exit_reason:
                    self._close_position(price, time, exit_reason)

            if not self.current_position:
                if signal == 1:
                    self._open_position("LONG", price, curr_atr, time)
                elif signal == -1:
                    self._open_position("SHORT", price, curr_atr, time)
            elif (signal == 1 and self.current_position.direction == "SHORT") or \
                 (signal == -1 and self.current_position.direction == "LONG"):
                self._close_position(price, time, "signal_reversal")
                new_dir = "LONG" if signal == 1 else "SHORT"
                self._open_position(new_dir, price, curr_atr, time)

            self.equity_curve.append(self._calculate_equity(price))

        if self.current_position:
            self._close_position(float(prices.iloc[-1]), df.index[-1], "end_of_data")

        analyzer = PerformanceAnalyzer()
        for t in self.trades:
            analyzer.add_trade(TradeRecord(
                symbol=symbol, direction=t.direction, entry_time=t.entry_time,
                exit_time=t.exit_time, entry_price=t.entry_price,
                exit_price=t.exit_price, volume=t.size, pnl=t.pnl, pnl_pct=t.pnl_pct
            ))

        return BacktestResult(
            trades=self.trades,
            equity_curve=pd.Series(self.equity_curve,
                                   index=df.index[:len(self.equity_curve)]),
            metrics=analyzer.calculate_metrics(self.initial_capital),
            signals=signals, prices=prices
        )

    def _open_position(self, direction: str, price: float, atr: float, time):
        """Open a new position with SL/TP."""
        slip = price * self.slippage_pct
        entry_price = price + slip if direction == "LONG" else price - slip

        risk_amount = self.cash * self.position_size_pct
        stop_distance = atr * self.stop_loss_atr
        size = risk_amount / stop_distance if stop_distance > 0 else 0.01

        if direction == "LONG":
            sl = entry_price - (atr * self.stop_loss_atr)
            tp = entry_price + (atr * self.take_profit_atr)
        else:
            sl = entry_price + (atr * self.stop_loss_atr)
            tp = entry_price - (atr * self.take_profit_atr)

        self.current_position = BacktestTrade(
            entry_time=time, entry_price=entry_price,
            direction=direction, size=size, stop_loss=sl, take_profit=tp
        )
        self.cash -= entry_price * size * self.commission_pct

    def _close_position(self, price: float, time, reason: str):
        """Close current position and record trade."""
        if not self.current_position:
            return
        pos = self.current_position
        slip = price * self.slippage_pct
        exit_price = price - slip if pos.direction == "LONG" else price + slip

        if pos.direction == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.size
        else:
            pnl = (pos.entry_price - exit_price) * pos.size

        pnl -= exit_price * pos.size * self.commission_pct
        pnl_pct = pnl / (pos.entry_price * pos.size) if pos.entry_price > 0 else 0

        pos.exit_time = time
        pos.exit_price = exit_price
        pos.pnl = pnl
        pos.pnl_pct = pnl_pct
        pos.exit_reason = reason

        self.cash += pnl
        self.trades.append(pos)
        self.current_position = None

    def _check_exits(self, pos: BacktestTrade, high: float, low: float) -> Optional[str]:
        """Check if SL or TP was hit."""
        if pos.direction == "LONG":
            if pos.stop_loss and low <= pos.stop_loss:
                return "stop_loss"
            if pos.take_profit and high >= pos.take_profit:
                return "take_profit"
        else:
            if pos.stop_loss and high >= pos.stop_loss:
                return "stop_loss"
            if pos.take_profit and low <= pos.take_profit:
                return "take_profit"
        return None

    def _calculate_equity(self, current_price: float) -> float:
        """Calculate current equity including unrealized P&L."""
        equity = self.cash
        if self.current_position:
            pos = self.current_position
            if pos.direction == "LONG":
                equity += (current_price - pos.entry_price) * pos.size
            else:
                equity += (pos.entry_price - current_price) * pos.size
        return equity


def run_backtest(symbol: str, timeframe: str = "H1", bars: int = 1000,
                 initial_capital: float = 10000, threshold: float = 1.5) -> BacktestResult:
    """Convenience function to run a complete backtest."""
    from robot.data_fetcher import DataFetcher

    fetcher = DataFetcher()
    df = fetcher.fetch(symbol, timeframe, bars)

    if df.empty or len(df) < 100:
        raise ValueError(f"Insufficient data for {symbol}")

    strategy = KalmanStrategy(threshold=threshold)
    backtester = Backtester(strategy=strategy, initial_capital=initial_capital)
    return backtester.run(df, symbol)