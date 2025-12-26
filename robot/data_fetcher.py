"""
Data Fetcher Module

Single Responsibility: Only handles data fetching from various sources.
Includes retry logic and proper error handling.
"""
from typing import Dict, Optional
import pandas as pd

from .logger import get_logger, retry_with_backoff
from .validation import validate_symbol, validate_timeframe, ValidationError


class DataFetchError(Exception):
    """Exception raised when data fetching fails."""
    pass


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
        self._logger = get_logger("data_fetcher")

    def _init_mt5(self) -> bool:
        """Initialize MT5 connection with retry logic."""
        if self._mt5_initialized:
            return self._mt5 is not None

        try:
            import MetaTrader5 as mt5
            from .config import Config

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
                error = mt5.last_error()
                self._logger.warning(f"MT5 init failed: {error}")
                return False

            self._mt5 = mt5
            self._mt5_initialized = True
            account = mt5.account_info()
            if account:
                self._logger.info(f"MT5 connected: Account {account.login}")
            else:
                self._logger.info("MT5 connected (account info unavailable)")
            return True

        except ImportError:
            self._logger.warning("MetaTrader5 module not installed")
            return False
        except Exception as e:
            self._logger.error(f"MT5 init error: {e}")
            return False
    
    def fetch(self, symbol: str, timeframe: str = "1h", bars: int = 500) -> pd.DataFrame:
        """
        Fetch OHLCV data with validation and error handling.

        Args:
            symbol: Trading symbol (e.g., EURUSD)
            timeframe: Timeframe (1h, 4h, 1d)
            bars: Number of bars to fetch

        Returns:
            DataFrame with Open, High, Low, Close, Volume columns

        Raises:
            ValidationError: If symbol or timeframe is invalid
        """
        # Validate inputs
        symbol_result = validate_symbol(symbol)
        if not symbol_result.is_valid:
            self._logger.error(f"Invalid symbol: {symbol_result.message}")
            raise ValidationError(symbol_result.message)
        symbol = symbol_result.value

        tf_result = validate_timeframe(timeframe)
        if not tf_result.is_valid:
            self._logger.error(f"Invalid timeframe: {tf_result.message}")
            raise ValidationError(tf_result.message)

        if bars < 1 or bars > 10000:
            self._logger.error(f"Invalid bars count: {bars}")
            raise ValidationError(f"Bars must be between 1 and 10000, got {bars}")

        self._logger.debug(f"Fetching {bars} bars of {symbol} @ {timeframe}")

        # Try MT5 first
        if self.use_mt5 and self._init_mt5():
            df = self._fetch_mt5(symbol, timeframe, bars)
            if not df.empty:
                self._logger.info(f"Fetched {len(df)} bars from MT5 for {symbol}")
                return df

        # Fallback to yfinance
        df = self._fetch_yfinance(symbol, timeframe, bars)
        if not df.empty:
            self._logger.info(f"Fetched {len(df)} bars from yfinance for {symbol}")
        else:
            self._logger.warning(f"No data fetched for {symbol}")
        return df
    
    @retry_with_backoff(max_retries=2, initial_delay=0.5, exceptions=(Exception,))
    def _fetch_mt5(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Fetch from MT5 with retry logic."""
        if not self._mt5:
            return pd.DataFrame()

        # Map timeframe to MT5 constant
        tf_map = {
            "1h": self._mt5.TIMEFRAME_H1,
            "h1": self._mt5.TIMEFRAME_H1,
            "4h": self._mt5.TIMEFRAME_H4,
            "h4": self._mt5.TIMEFRAME_H4,
            "1d": self._mt5.TIMEFRAME_D1,
            "d1": self._mt5.TIMEFRAME_D1,
        }
        tf = tf_map.get(timeframe.lower(), self._mt5.TIMEFRAME_H1)

        rates = self._mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            self._logger.debug(f"No MT5 data for {symbol}")
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
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0, exceptions=(Exception,))
    def _fetch_yfinance(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Fallback: Fetch from yfinance with retry logic."""
        import yfinance as yf

        # Map symbol to yfinance format
        yf_symbol = symbol
        if len(symbol) == 6 and symbol.isupper():
            yf_symbol = f"{symbol}=X"  # Forex format

        # Map timeframe
        interval = "1h" if timeframe in ["1h", "4h", "h1", "h4"] else "1d"
        period = "1mo" if timeframe in ["1h", "4h", "h1", "h4"] else "1y"

        self._logger.debug(f"Fetching {yf_symbol} from yfinance")
        df = yf.download(yf_symbol, period=period, interval=interval, progress=False)

        if df.empty:
            self._logger.debug(f"No yfinance data for {yf_symbol}")
            return pd.DataFrame()

        # Handle MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Ensure we have required columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

        # Resample if needed (e.g., 1h -> 4h)
        if timeframe in ["4h", "h4"] and interval == "1h":
            df = df.resample("4h").agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()

        return df.tail(bars)

