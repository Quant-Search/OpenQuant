"""
Data Fetcher Module

Single Responsibility: Only handles data fetching from various sources.
"""
from typing import Dict
import pandas as pd


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
                print(f"[WARN] MT5 init failed: {mt5.last_error()}")
                return False
                
            self._mt5 = mt5
            self._mt5_initialized = True
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

