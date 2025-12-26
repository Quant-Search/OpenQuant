"""
Robot Orchestrator

Single Responsibility: Only coordinates components (strategy, data, trader, risk).
Dependency Inversion: Depends on abstractions (BaseStrategy interface).
"""
import time
import signal
from datetime import datetime
from typing import Dict

import pandas as pd
import numpy as np

from .config import Config
from .strategy import BaseStrategy, KalmanStrategy
from .data_fetcher import DataFetcher
from .risk_manager import RiskManager
from .trader import Trader


class Robot:
    """Main trading robot - orchestrates all components."""
    
    def __init__(self, mode: str = "paper", strategy: BaseStrategy = None):
        """
        Initialize the robot.
        
        Args:
            mode: "paper", "live", or "backtest"
            strategy: Strategy instance (defaults to KalmanStrategy)
        """
        self.mode = mode
        self.strategy = strategy or KalmanStrategy(
            process_noise=Config.PROCESS_NOISE,
            measurement_noise=Config.MEASUREMENT_NOISE,
            threshold=Config.SIGNAL_THRESHOLD
        )
        self.fetcher = DataFetcher(use_mt5=(mode != "backtest"))
        self.trader = Trader(mode=mode)
        self.running = True
        
    def run_once(self):
        """Run one cycle of the robot."""
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running cycle...")
        print(f"{'='*60}")
        
        equity = self.trader.get_equity()
        current_positions = self.trader.get_positions()
        # Track position count locally to update after each trade
        position_count = len(current_positions)
        print(f"[INFO] Equity: ${equity:.2f}, Positions: {position_count}")
        
        for symbol in Config.SYMBOLS:
            print(f"\n--- {symbol} ---")
            
            # Skip if max positions reached (use local count for accuracy)
            if position_count >= Config.MAX_POSITIONS:
                if symbol not in current_positions:
                    print(f"[SKIP] Max positions ({Config.MAX_POSITIONS}) reached")
                    continue
            
            # Fetch data
            df = self.fetcher.fetch(symbol, Config.TIMEFRAME, bars=500)
            if df.empty or len(df) < 50:
                print(f"[WARN] Not enough data for {symbol}")
                continue
                
            # Get current price and ATR
            current_price = float(df['Close'].iloc[-1])
            atr = RiskManager.calculate_atr(df)
            
            # Validate ATR before proceeding
            if pd.isna(atr) or atr <= 0:
                print(f"[WARN] Invalid ATR for {symbol}, skipping")
                continue
                
            print(f"[DATA] Price: {current_price:.5f}, ATR: {atr:.5f}")
            
            # Update paper trading prices for accurate equity calculation
            if self.mode == "paper":
                self.trader.update_paper_prices({symbol: current_price})
            
            # Generate signal
            signals = self.strategy.generate_signals(df)
            latest_signal = int(signals.iloc[-1])
            
            signal_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}[latest_signal]
            print(f"[SIGNAL] {signal_str}")
            
            # Check current position
            current_pos = current_positions.get(symbol, 0)
            
            # Trading logic
            if latest_signal == 1 and current_pos <= 0:
                # Go LONG
                side = "BUY"
                sl, tp = RiskManager.calculate_stops(
                    current_price, atr, "LONG",
                    Config.STOP_LOSS_ATR_MULT,
                    Config.TAKE_PROFIT_ATR_MULT
                )
                position_size = RiskManager.calculate_position_size(
                    equity, current_price, sl, Config.RISK_PER_TRADE
                )
                # Convert to lots (forex: 1 lot = 100,000 units)
                volume = position_size / 100000.0
                volume = max(0.01, round(volume, 2))
                
                if self.trader.place_order(symbol, side, volume, sl, tp):
                    # Only increment count if this is a NEW position (not a flip)
                    is_new_position = symbol not in current_positions or current_pos == 0
                    current_positions[symbol] = volume
                    if is_new_position:
                        position_count += 1
                
            elif latest_signal == -1 and current_pos >= 0:
                # Go SHORT
                side = "SELL"
                sl, tp = RiskManager.calculate_stops(
                    current_price, atr, "SHORT",
                    Config.STOP_LOSS_ATR_MULT,
                    Config.TAKE_PROFIT_ATR_MULT
                )
                position_size = RiskManager.calculate_position_size(
                    equity, current_price, sl, Config.RISK_PER_TRADE
                )
                volume = position_size / 100000.0
                volume = max(0.01, round(volume, 2))
                
                if self.trader.place_order(symbol, side, volume, sl, tp):
                    # Only increment count if this is a NEW position (not a flip)
                    is_new_position = symbol not in current_positions or current_pos == 0
                    current_positions[symbol] = -volume
                    if is_new_position:
                        position_count += 1
                
            else:
                print(f"[HOLD] No action needed")
        
        print(f"\n[INFO] Cycle complete. Next run in {Config.LOOP_INTERVAL_SECONDS}s")
    
    def run_backtest(self):
        """Run backtest on historical data."""
        print("\n" + "="*60)
        print("BACKTEST MODE")
        print("="*60)
        
        for symbol in Config.SYMBOLS:
            print(f"\n--- Backtesting {symbol} ---")
            
            # Fetch historical data
            df = self.fetcher.fetch(symbol, Config.TIMEFRAME, bars=1000)
            if df.empty or len(df) < 100:
                print(f"[WARN] Not enough data for {symbol}")
                continue
            
            # Generate signals
            signals = self.strategy.generate_signals(df)
            
            # Calculate returns
            returns = df['Close'].pct_change().fillna(0)
            # Shift signals and fill NaN to avoid propagation in calculations
            strategy_returns = signals.shift(1).fillna(0) * returns
            
            # Metrics (ensure no NaN values)
            strategy_returns = strategy_returns.fillna(0)
            total_return = (1 + strategy_returns).prod() - 1
            sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() > 0 else 0
            max_dd = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()
            
            n_trades = (signals.diff().abs() > 0).sum()
            
            print(f"[RESULTS]")
            print(f"  Total Return: {total_return:.2%}")
            print(f"  Sharpe Ratio: {sharpe:.2f}")
            print(f"  Max Drawdown: {max_dd:.2%}")
            print(f"  Number of Trades: {n_trades}")
    
    def run(self):
        """Main loop."""
        if self.mode == "backtest":
            self.run_backtest()
            return
            
        print(f"\nStarting OpenQuant MVP Robot in {self.mode.upper()} mode")
        print(f"Symbols: {Config.SYMBOLS}")
        print(f"Timeframe: {Config.TIMEFRAME}")
        print(f"Risk per trade: {Config.RISK_PER_TRADE:.0%}")
        print(f"Press Ctrl+C to stop\n")
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            print("\n[SHUTDOWN] Stopping robot...")
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Main loop
        while self.running:
            try:
                self.run_once()
                
                # Sleep until next run
                for _ in range(Config.LOOP_INTERVAL_SECONDS):
                    if not self.running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                print(f"[ERROR] Cycle failed: {e}")
                time.sleep(60)  # Wait before retry
        
        print("[SHUTDOWN] Robot stopped.")


