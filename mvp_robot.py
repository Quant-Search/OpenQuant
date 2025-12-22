#!/usr/bin/env python3
"""
OpenQuant MVP Robot - Simple Trading Robot
==========================================

This is a minimal, working trading robot that:
1. Connects to MetaTrader 5 (or runs in paper mode)
2. Fetches price data for configured symbols
3. Generates trading signals using Kalman Filter Mean Reversion strategy
4. Executes trades with basic risk management (position sizing, stop loss)

Usage:
    python mvp_robot.py --mode paper       # Paper trading (no real money)
    python mvp_robot.py --mode live        # Live trading on MT5 (requires MT5 credentials)
    python mvp_robot.py --mode backtest    # Backtest the strategy

Configuration via environment variables or .env file:
    MT5_LOGIN=12345678
    MT5_PASSWORD=yourpassword
    MT5_SERVER=YourBroker-Server
    MT5_TERMINAL_PATH=C:/Program Files/MetaTrader 5/terminal64.exe

Author: OpenQuant
License: MIT
"""

import os
import sys
import time
import signal
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

import pandas as pd
import numpy as np


# =============================================================================
# CONFIGURATION - Modify these values for your setup
# =============================================================================

class Config:
    """Robot configuration - all settings in one place."""
    
    # Trading symbols (MT5 format)
    SYMBOLS: List[str] = ["EURUSD", "GBPUSD", "USDJPY"]
    
    # Timeframe for analysis (1h, 4h, 1d)
    TIMEFRAME: str = "1h"
    
    # Strategy parameters (Kalman Filter)
    PROCESS_NOISE: float = 1e-5       # How much true price varies
    MEASUREMENT_NOISE: float = 1e-3   # How noisy are observations
    SIGNAL_THRESHOLD: float = 1.5     # Z-score threshold for signals
    
    # Risk management
    RISK_PER_TRADE: float = 0.02      # Risk 2% of equity per trade
    MAX_POSITIONS: int = 3            # Maximum concurrent positions
    STOP_LOSS_ATR_MULT: float = 2.0   # Stop loss = 2x ATR
    TAKE_PROFIT_ATR_MULT: float = 3.0 # Take profit = 3x ATR
    
    # Loop settings
    LOOP_INTERVAL_SECONDS: int = 3600  # Run every hour
    
    # MT5 credentials (from environment)
    # Safe conversion: returns None if MT5_LOGIN is missing or not a valid integer
    @staticmethod
    def _safe_int(value: Optional[str]) -> Optional[int]:
        """Convert string to int safely, return None on failure."""
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None
    
    MT5_LOGIN: Optional[int] = _safe_int.__func__(os.getenv("MT5_LOGIN"))
    MT5_PASSWORD: Optional[str] = os.getenv("MT5_PASSWORD")
    MT5_SERVER: Optional[str] = os.getenv("MT5_SERVER")
    MT5_TERMINAL_PATH: Optional[str] = os.getenv("MT5_TERMINAL_PATH")


# =============================================================================
# KALMAN FILTER STRATEGY - The core signal generation
# =============================================================================

class KalmanStrategy:
    """
    Kalman Filter Mean Reversion Strategy.
    
    This strategy uses a Kalman Filter to estimate the "true" price from noisy
    market observations. Trading signals are generated when the observed price
    deviates significantly from the estimated true price.
    
    Mathematical Model:
    ------------------
    State equation:     x(t+1) = x(t) + w(t),  w(t) ~ N(0, Q)
    Observation:        z(t) = x(t) + v(t),    v(t) ~ N(0, R)
    
    Where:
    - x(t) = true (hidden) price
    - z(t) = observed market price
    - Q = process noise variance (how much true price varies)
    - R = measurement noise variance (observation noise)
    
    Kalman Filter Update:
    1. Predict: x_pred = x_prev, P_pred = P_prev + Q
    2. Update:  K = P_pred / (P_pred + R)
                x = x_pred + K * (z - x_pred)
                P = (1 - K) * P_pred
    
    Trading Logic:
    - deviation = observed_price - kalman_estimate
    - z_score = deviation / rolling_std(deviation, 50 periods)
    - LONG when z_score < -threshold (price below true value)
    - SHORT when z_score > threshold (price above true value)
    """
    
    def __init__(
        self,
        process_noise: float = 1e-5,
        measurement_noise: float = 1e-3,
        threshold: float = 1.5
    ):
        """
        Initialize the Kalman Strategy.
        
        Args:
            process_noise: Q - how much the true price is expected to vary
            measurement_noise: R - how noisy the observed prices are
            threshold: Z-score threshold for generating signals
        """
        self.Q = process_noise      # Process noise variance
        self.R = measurement_noise  # Measurement noise variance
        self.threshold = threshold  # Signal generation threshold
        
    def _kalman_filter(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply 1D Kalman Filter to price series.
        
        Args:
            prices: Array of observed prices
            
        Returns:
            Tuple of (estimated_prices, deviations)
        """
        n = len(prices)
        
        # Initialize state
        x = prices[0]  # Initial estimate = first observation
        P = 1.0        # Initial uncertainty
        
        # Output arrays
        estimates = np.zeros(n)
        deviations = np.zeros(n)
        
        for i in range(n):
            # === PREDICTION STEP ===
            x_pred = x           # State prediction (random walk model)
            P_pred = P + self.Q  # Uncertainty grows by process noise
            
            # === UPDATE STEP ===
            z = prices[i]                    # Current observation
            K = P_pred / (P_pred + self.R)   # Kalman gain
            x = x_pred + K * (z - x_pred)    # Updated estimate
            P = (1 - K) * P_pred             # Updated uncertainty
            
            # Store results
            estimates[i] = x
            deviations[i] = z - x  # Innovation (deviation from estimate)
            
        return estimates, deviations
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals from OHLCV data.
        
        Args:
            df: DataFrame with 'Close' column
            
        Returns:
            Series of signals: +1 (long), 0 (flat), -1 (short)
        """
        if df.empty or len(df) < 50:
            return pd.Series(0, index=df.index, dtype=int)
        
        # Get close prices
        prices = df['Close'].values
        
        # Apply Kalman filter
        estimates, deviations = self._kalman_filter(prices)
        
        # Calculate z-score of deviations
        dev_series = pd.Series(deviations, index=df.index)
        rolling_std = dev_series.rolling(window=50).std().fillna(1.0)
        z_score = dev_series / rolling_std
        
        # Generate signals
        signals = pd.Series(0, index=df.index, dtype=int)
        signals[z_score > self.threshold] = -1   # Price above estimate -> SHORT
        signals[z_score < -self.threshold] = 1   # Price below estimate -> LONG
        
        return signals


# =============================================================================
# DATA FETCHER - Get price data
# =============================================================================

class DataFetcher:
    """Fetch OHLCV data from MT5 or fallback to yfinance."""
    
    def __init__(self, use_mt5: bool = True):
        """
        Initialize data fetcher.
        
        Args:
            use_mt5: Whether to use MT5 for data (falls back to yfinance if fails)
        """
        self.use_mt5 = use_mt5
        self._mt5 = None
        self._mt5_initialized = False
        
    def _init_mt5(self) -> bool:
        """Initialize MT5 connection."""
        if self._mt5_initialized:
            return self._mt5 is not None
            
        try:
            import MetaTrader5 as mt5
            
            # Initialize with credentials if available
            kwargs = {}
            if Config.MT5_LOGIN:
                kwargs["login"] = Config.MT5_LOGIN
            if Config.MT5_PASSWORD:
                kwargs["password"] = Config.MT5_PASSWORD
            if Config.MT5_SERVER:
                kwargs["server"] = Config.MT5_SERVER
                
            if Config.MT5_TERMINAL_PATH:
                ok = mt5.initialize(path=Config.MT5_TERMINAL_PATH, **kwargs)
            else:
                ok = mt5.initialize(**kwargs)
                
            if not ok:
                print(f"[WARN] MT5 init failed: {mt5.last_error()}")
                return False
                
            self._mt5 = mt5
            self._mt5_initialized = True
            # Safe access: check if account_info() returns valid object
            account = mt5.account_info()
            if account:
                print(f"[INFO] MT5 connected: Account {account.login}")
            else:
                print("[INFO] MT5 connected (account info unavailable)")
            return True
            
        except ImportError:
            print("[WARN] MetaTrader5 module not installed")
            return False
        except Exception as e:
            print(f"[WARN] MT5 init error: {e}")
            return False
    
    def fetch(self, symbol: str, timeframe: str = "1h", bars: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV data.
        
        Args:
            symbol: Trading symbol (e.g., EURUSD)
            timeframe: Timeframe (1h, 4h, 1d)
            bars: Number of bars to fetch
            
        Returns:
            DataFrame with Open, High, Low, Close, Volume columns
        """
        # Try MT5 first
        if self.use_mt5 and self._init_mt5():
            df = self._fetch_mt5(symbol, timeframe, bars)
            if not df.empty:
                return df
                
        # Fallback to yfinance
        return self._fetch_yfinance(symbol, timeframe, bars)
    
    def _fetch_mt5(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Fetch from MT5."""
        if not self._mt5:
            return pd.DataFrame()
            
        # Map timeframe to MT5 constant
        tf_map = {
            "1h": self._mt5.TIMEFRAME_H1,
            "4h": self._mt5.TIMEFRAME_H4,
            "1d": self._mt5.TIMEFRAME_D1,
        }
        tf = tf_map.get(timeframe.lower(), self._mt5.TIMEFRAME_H1)
        
        try:
            rates = self._mt5.copy_rates_from_pos(symbol, tf, 0, bars)
            if rates is None or len(rates) == 0:
                return pd.DataFrame()
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df = df.set_index('time')
            df = df.rename(columns={
                'open': 'Open', 
                'high': 'High', 
                'low': 'Low', 
                'close': 'Close',
                'tick_volume': 'Volume'
            })
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            
        except Exception as e:
            print(f"[WARN] MT5 fetch error for {symbol}: {e}")
            return pd.DataFrame()
    
    def _fetch_yfinance(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Fallback: Fetch from yfinance."""
        try:
            import yfinance as yf
            
            # Map symbol to yfinance format
            yf_symbol = symbol
            if len(symbol) == 6 and symbol.isupper():
                yf_symbol = f"{symbol}=X"  # Forex format
            
            # Map timeframe
            interval = "1h" if timeframe in ["1h", "4h"] else "1d"
            period = "1mo" if timeframe in ["1h", "4h"] else "1y"
            
            df = yf.download(yf_symbol, period=period, interval=interval, progress=False)
            
            if df.empty:
                return pd.DataFrame()
                
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Ensure we have required columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
            
            # Resample if needed (e.g., 1h -> 4h)
            if timeframe == "4h" and interval == "1h":
                df = df.resample("4h").agg({
                    'Open': 'first',
                    'High': 'max', 
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
            return df.tail(bars)
            
        except Exception as e:
            print(f"[ERROR] yfinance fetch error for {symbol}: {e}")
            return pd.DataFrame()


# =============================================================================
# RISK MANAGER - Position sizing and stop loss
# =============================================================================

class RiskManager:
    """
    Risk management for position sizing and stop loss.
    
    Implements:
    - ATR-based stop loss / take profit
    - Fixed fractional position sizing (risk X% per trade)
    """
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """
        Calculate Average True Range (ATR).
        
        ATR = Average of True Range over N periods
        True Range = max(High-Low, abs(High-PrevClose), abs(Low-PrevClose))
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean().iloc[-1]
        
        return float(atr)
    
    @staticmethod
    def calculate_position_size(
        equity: float,
        entry_price: float,
        stop_loss_price: float,
        risk_percent: float = 0.02
    ) -> float:
        """
        Calculate position size based on risk.
        
        Position Size = (Equity * Risk%) / (Entry - StopLoss)
        
        Args:
            equity: Account equity
            entry_price: Entry price
            stop_loss_price: Stop loss price
            risk_percent: Fraction of equity to risk (e.g., 0.02 = 2%)
            
        Returns:
            Position size in units
        """
        risk_amount = equity * risk_percent
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            return 0.0
            
        position_size = risk_amount / risk_per_unit
        return position_size
    
    @staticmethod
    def calculate_stops(
        entry_price: float,
        atr: float,
        side: str,  # "LONG" or "SHORT"
        sl_mult: float = 2.0,
        tp_mult: float = 3.0
    ) -> Tuple[float, float]:
        """
        Calculate stop loss and take profit levels.
        
        Args:
            entry_price: Entry price
            atr: Average True Range
            side: "LONG" or "SHORT"
            sl_mult: Stop loss ATR multiplier
            tp_mult: Take profit ATR multiplier
            
        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        if side.upper() == "LONG":
            stop_loss = entry_price - (atr * sl_mult)
            take_profit = entry_price + (atr * tp_mult)
        else:
            stop_loss = entry_price + (atr * sl_mult)
            take_profit = entry_price - (atr * tp_mult)
            
        return stop_loss, take_profit


# =============================================================================
# TRADER - Execute trades
# =============================================================================

class Trader:
    """Execute trades on MT5 or paper trading."""
    
    def __init__(self, mode: str = "paper"):
        """
        Initialize trader.
        
        Args:
            mode: "paper" for simulated trading, "live" for real MT5 trading
        """
        self.mode = mode
        self._mt5 = None
        self._paper_positions: Dict[str, Dict] = {}
        self._paper_equity = 10000.0  # Starting paper equity
        
    def _get_mt5(self):
        """Get MT5 module (lazy init)."""
        if self._mt5:
            return self._mt5
            
        try:
            import MetaTrader5 as mt5
            if mt5.account_info():
                self._mt5 = mt5
                return mt5
        except:
            pass
        return None
    
    def get_equity(self) -> float:
        """Get account equity."""
        if self.mode == "paper":
            return self._paper_equity
            
        mt5 = self._get_mt5()
        if mt5:
            info = mt5.account_info()
            if info:
                return float(info.equity)
        return 0.0
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        if self.mode == "paper":
            return {s: p["volume"] for s, p in self._paper_positions.items()}
            
        mt5 = self._get_mt5()
        if not mt5:
            return {}
            
        positions = mt5.positions_get()
        if not positions:
            return {}
            
        result = {}
        for p in positions:
            vol = float(p.volume)
            result[p.symbol] = vol if p.type == 0 else -vol  # 0=BUY, 1=SELL
        return result
    
    def place_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        volume: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """
        Place a market order.
        
        Args:
            symbol: Trading symbol
            side: "BUY" or "SELL"
            volume: Position size in lots
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            
        Returns:
            True if order successful
        """
        print(f"[ORDER] {side} {volume:.2f} {symbol} SL={stop_loss} TP={take_profit}")
        
        if self.mode == "paper":
            # Simulate paper trade
            self._paper_positions[symbol] = {
                "volume": volume if side == "BUY" else -volume,
                "side": side,
                "sl": stop_loss,
                "tp": take_profit,
                "entry_time": datetime.now(timezone.utc)
            }
            return True
            
        # Live MT5 trade
        mt5 = self._get_mt5()
        if not mt5:
            print("[ERROR] MT5 not available for live trading")
            return False
            
        # Get symbol info
        info = mt5.symbol_info(symbol)
        if not info:
            mt5.symbol_select(symbol, True)
            info = mt5.symbol_info(symbol)
            
        if not info:
            print(f"[ERROR] Symbol {symbol} not available")
            return False
            
        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            print(f"[ERROR] Cannot get price for {symbol}")
            return False
            
        price = tick.ask if side == "BUY" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        
        # Round volume to symbol specs
        vol_min = float(info.volume_min)
        vol_step = float(info.volume_step)
        volume = max(vol_min, round(volume / vol_step) * vol_step)
        
        # Build order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": 987654321,
            "comment": "OpenQuant MVP",
        }
        
        if stop_loss:
            request["sl"] = stop_loss
        if take_profit:
            request["tp"] = take_profit
            
        # Send order
        result = mt5.order_send(request)
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[SUCCESS] Order filled at {result.price}")
            return True
        else:
            code = result.retcode if result else "None"
            comment = result.comment if result else "No result"
            print(f"[ERROR] Order failed: {code} - {comment}")
            return False


# =============================================================================
# MAIN ROBOT LOOP
# =============================================================================

class Robot:
    """Main trading robot."""
    
    def __init__(self, mode: str = "paper"):
        """
        Initialize the robot.
        
        Args:
            mode: "paper", "live", or "backtest"
        """
        self.mode = mode
        self.strategy = KalmanStrategy(
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
        print(f"[INFO] Equity: ${equity:.2f}, Positions: {len(current_positions)}")
        
        for symbol in Config.SYMBOLS:
            print(f"\n--- {symbol} ---")
            
            # Skip if max positions reached
            if len(current_positions) >= Config.MAX_POSITIONS:
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
            print(f"[DATA] Price: {current_price:.5f}, ATR: {atr:.5f}")
            
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
                
                self.trader.place_order(symbol, side, volume, sl, tp)
                
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
                
                self.trader.place_order(symbol, side, volume, sl, tp)
                
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
            strategy_returns = signals.shift(1) * returns  # Signal from previous bar
            
            # Metrics
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


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="OpenQuant MVP Trading Robot")
    parser.add_argument(
        "--mode", 
        choices=["paper", "live", "backtest"],
        default="paper",
        help="Trading mode: paper (simulated), live (real MT5), backtest (historical)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbols (e.g., EURUSD,GBPUSD)"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe: 1h, 4h, 1d"
    )
    
    args = parser.parse_args()
    
    # Override config from args
    if args.symbols:
        Config.SYMBOLS = [s.strip() for s in args.symbols.split(",")]
    if args.timeframe:
        Config.TIMEFRAME = args.timeframe
    
    # Create and run robot
    robot = Robot(mode=args.mode)
    robot.run()


if __name__ == "__main__":
    main()

