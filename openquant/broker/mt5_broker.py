"""
MetaTrader 5 Broker Implementation.
"""
import os
import logging
from typing import Dict, List, Any, Optional
from openquant.broker.abstract import Broker
from openquant.risk.trade_validator import TRADE_VALIDATOR
from openquant.utils.retry import ConnectionRetry, RetryConfig

logger = logging.getLogger(__name__)

try:
    from openquant.paper import mt5_bridge
    MT5_AVAILABLE = mt5_bridge.is_available()
except ImportError:
    MT5_AVAILABLE = False


class MT5ConnectionPool:
    """
    Connection pool manager for MT5.
    
    Maintains connection state and handles reconnection logic.
    Provides health check functionality.
    """
    
    def __init__(self, 
                 login: int,
                 password: str,
                 server: str,
                 terminal_path: Optional[str] = None,
                 retry_config: Optional[RetryConfig] = None):
        """Initialize connection pool.
        
        Args:
            login: MT5 account login
            password: MT5 account password
            server: MT5 server name
            terminal_path: Path to MT5 terminal executable
            retry_config: Retry configuration for reconnections
        """
        self.login = login
        self.password = password
        self.server = server
        self.terminal_path = terminal_path
        
        # Initialize retry handler
        config = retry_config or RetryConfig(
            max_retries=5,
            base_delay=2.0,
            max_delay=30.0,
            backoff_factor=2.0,
            jitter=0.1
        )
        self.retry_handler = ConnectionRetry(name="MT5", config=config)
        
        # MT5 module reference
        self.mt5 = None
        
        # Connection state
        self._connected = False
        self._last_health_check = 0.0
        self._health_check_interval = 30.0  # seconds
        
    def connect(self) -> bool:
        """Establish connection to MT5 with retry logic.
        
        Returns:
            True if connection successful, False otherwise
        """
        @self.retry_handler.with_retry
        def _connect():
            # Initialize MT5
            if not mt5_bridge.init(
                login=self.login,
                password=self.password,
                server=self.server,
                terminal_path=self.terminal_path
            ):
                raise RuntimeError("Failed to initialize/login to MT5")
            
            # Get MT5 module reference
            self.mt5 = mt5_bridge._lazy_import()
            if not self.mt5:
                raise RuntimeError("MT5 module not available after initialization")
            
            # Verify connection by checking account info
            account = self.mt5.account_info()
            if not account:
                raise RuntimeError("Cannot retrieve account info after connection")
            
            self._connected = True
            return True
        
        try:
            return _connect()
        except Exception:
            self._connected = False
            return False
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy.
        
        Performs a lightweight check by attempting to retrieve account info.
        
        Returns:
            True if connection is healthy, False otherwise
        """
        if not self._connected or not self.mt5:
            return False
        
        try:
            account = self.mt5.account_info()
            return account is not None
        except Exception:
            return False
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to MT5.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        # Shutdown existing connection
        try:
            if self.mt5:
                mt5_bridge.shutdown()
        except Exception:
            pass
        
        # Reset state
        self._connected = False
        self.mt5 = None
        self.retry_handler.reset()
        
        # Reconnect
        return self.connect()
    
    def ensure_connection(self) -> bool:
        """Ensure connection is active and healthy.
        
        Performs health check and reconnects if necessary.
        
        Returns:
            True if connection is healthy or reconnection successful
        """
        import time
        
        # Check if we need to perform health check
        current_time = time.time()
        time_since_last_check = current_time - self._last_health_check
        
        # Perform health check if interval elapsed or not connected
        if not self._connected or time_since_last_check >= self._health_check_interval:
            self._last_health_check = current_time
            
            if not self.is_healthy():
                # Connection unhealthy, attempt reconnection
                return self.reconnect()
        
        return self._connected
    
    def shutdown(self):
        """Shutdown connection pool."""
        try:
            if self.mt5:
                mt5_bridge.shutdown()
        except Exception:
            pass
        finally:
            self._connected = False
            self.mt5 = None


class MT5Broker(Broker):
    """
    MetaTrader 5 Broker implementation using mt5_bridge.
    
    Features:
    - Connection pooling with automatic reconnection
    - Health checks before order execution
    - Retry logic for connection failures
    """
    def __init__(self, 
                 login: Optional[int] = None, 
                 password: Optional[str] = None, 
                 server: Optional[str] = None,
                 terminal_path: Optional[str] = None,
                 retry_config: Optional[RetryConfig] = None):
        if not MT5_AVAILABLE:
            raise ImportError("MetaTrader5 not available. Install it or run on Windows.")
            
        self.login = login or (int(os.getenv("MT5_LOGIN")) if os.getenv("MT5_LOGIN") else None)
        self.password = password or os.getenv("MT5_PASSWORD")
        self.server = server or os.getenv("MT5_SERVER")
        self.terminal_path = terminal_path or os.getenv("MT5_TERMINAL_PATH")
        
        if not self.login or not self.password or not self.server:
            raise ValueError("MT5 credentials missing. Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env or pass as arguments.")
        
        # Initialize connection pool
        self.pool = MT5ConnectionPool(
            login=self.login,
            password=self.password,
            server=self.server,
            terminal_path=self.terminal_path,
            retry_config=retry_config
        )
        
        # Establish initial connection
        if not self.pool.connect():
            raise RuntimeError("Failed to initialize/login to MT5. Check credentials and ensure MT5 terminal is installed.")
        
        # Get the MT5 module reference
        self.mt5 = self.pool.mt5
        if not self.mt5:
            raise RuntimeError("MT5 module not available after initialization")
            
        try:
            from openquant.analysis.tca import TCAMonitor
            self.tca = TCAMonitor()
        except ImportError as e:
            logger.info(f"TCA monitoring not available: {e}")
            self.tca = None
        except Exception as e:
            logger.warning(f"Failed to initialize TCA monitor: {e}")
            self.tca = None
    
    def _ensure_connection_health(self):
        """Ensure connection is healthy before operations.
        
        Raises:
            RuntimeError: If connection cannot be established or is unhealthy
        """
        if not self.pool.ensure_connection():
            raise RuntimeError("MT5 connection unhealthy and reconnection failed")
        
        # Update mt5 reference in case of reconnection
        self.mt5 = self.pool.mt5

    def get_cash(self) -> float:
        """Return available cash balance."""
        try:
            self._ensure_connection_health()
            account = self.mt5.account_info()
            if account:
                return float(account.balance)
            logger.warning("MT5 account_info returned None")
            return 0.0
        except (AttributeError, ValueError) as e:
            logger.error(f"Error retrieving cash balance from MT5: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Unexpected error getting cash balance: {e}")
            return 0.0

    def get_equity(self) -> float:
        """Return total account equity (cash + positions)."""
        try:
            self._ensure_connection_health()
            return mt5_bridge.account_equity(self.mt5)
        except Exception:
            return 0.0

    def get_positions(self) -> Dict[str, float]:
        """
        Return current positions.
        Format: {"EURUSD": 0.5, "GBPUSD": -0.3}
        """
        try:
            self._ensure_connection_health()
            return mt5_bridge.positions_by_symbol(self.mt5)
        except Exception:
            return {}

    def place_order(self, 
                   symbol: str, 
                   quantity: float, 
                   side: str, 
                   order_type: str = "market", 
                   limit_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place an order.
        side: 'buy' or 'sell'
        order_type: 'market', 'limit'
        quantity: volume in lots
        
        SAFETY: Validates trade through comprehensive risk checks before executing.
        If any check fails, raises RuntimeError with detailed reason.
        """
        # Health check before order execution
        self._ensure_connection_health()
        
        # Map symbol if needed (e.g., BTC/USDT -> BTCUSD)
        mt5_symbol = mt5_bridge.map_symbol(symbol)
        
        if not mt5_bridge._ensure_symbol(self.mt5, mt5_symbol):
            raise ValueError(f"Symbol {mt5_symbol} not available in MT5")
        
        info = self.mt5.symbol_info(mt5_symbol)
        if not info:
            raise ValueError(f"Cannot get symbol info for {mt5_symbol}")
            
        vmin = float(getattr(info, "volume_min", 0.01) or 0.01)
        vstep = float(getattr(info, "volume_step", 0.01) or 0.01)
        vmax = float(getattr(info, "volume_max", 100.0) or 100.0)
        
        volume = min(max(mt5_bridge._round_step(abs(quantity), vstep), vmin), vmax)
        
        mt5_order_type = self.mt5.ORDER_TYPE_BUY if side.lower() == "buy" else self.mt5.ORDER_TYPE_SELL
        
        tick = self.mt5.symbol_info_tick(mt5_symbol)
        if not tick:
            raise RuntimeError(f"Cannot get tick data for {mt5_symbol}")
            
        arrival_price = float(tick.ask if side.lower() == "buy" else tick.bid)
        
        # COMPREHENSIVE RISK VALIDATION
        try:
            equity = self.get_equity()
            positions_dict = self.get_positions()
            
            # Convert positions to notional values
            current_positions = {}
            for pos_symbol, pos_qty in positions_dict.items():
                # Get approximate notional value
                # For forex, we need contract size and price
                pos_tick = self.mt5.symbol_info_tick(pos_symbol)
                if pos_tick:
                    pos_price = float(pos_tick.ask if pos_qty > 0 else pos_tick.bid)
                    pos_info = self.mt5.symbol_info(pos_symbol)
                    contract_size = float(getattr(pos_info, "trade_contract_size", 100000.0))
                    current_positions[pos_symbol] = abs(pos_qty) * contract_size * pos_price
                else:
                    current_positions[pos_symbol] = abs(pos_qty) * 100000.0  # Fallback
            
            # Calculate notional value for this trade
            contract_size = float(getattr(info, "trade_contract_size", 100000.0))
            validation_price = limit_price if limit_price else arrival_price
            
            result = TRADE_VALIDATOR.validate_trade(
                symbol=mt5_symbol,
                quantity=abs(volume),
                price=validation_price * contract_size,  # Notional per lot
                side=side,
                portfolio_value=equity,
                current_positions=current_positions,
                current_equity=equity,
                asset_class="forex" if "/" in symbol else None,
            )
            
            if not result.allowed:
                raise RuntimeError(f"Trade validation failed: {result.reason}")
            
            # Log warnings if any
            for warning in result.warnings:
                from openquant.utils.logging import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Trade warning for {mt5_symbol}: {warning}")
                
        except Exception as e:
            if "Trade validation failed" in str(e):
                raise
            raise RuntimeError(f"Pre-trade validation error: {str(e)}")
        
        # Build request
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_symbol,
            "volume": volume,
            "type": mt5_order_type,
            "deviation": 20,
            "magic": 987654321,
            "comment": "OpenQuant",
        }
        
        if order_type.lower() == "market":
            request["price"] = arrival_price
        elif order_type.lower() == "limit":
            if limit_price is None:
                raise ValueError("Limit price required for limit orders")
            request["price"] = float(limit_price)
            request["type"] = self.mt5.ORDER_TYPE_BUY_LIMIT if side.lower() == "buy" else self.mt5.ORDER_TYPE_SELL_LIMIT
            arrival_price = limit_price
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
        
        # Send order with retry logic for transient failures
        @self.pool.retry_handler.with_retry
        def _send_order():
            # Verify connection health immediately before sending
            if not self.pool.is_healthy():
                raise RuntimeError("Connection unhealthy before order send")
            
            result = self.mt5.order_send(request)
            
            if result is None:
                raise RuntimeError("MT5 order_send returned None")
            
            return result
        
        try:
            result = _send_order()
        except Exception as e:
            # Send alert on order failure
            try:
                from openquant.utils.alerts import send_alert
                send_alert(
                    subject=f"MT5 Order Failed: {mt5_symbol}",
                    body=f"Order failed after retries: {str(e)}",
                    severity="ERROR"
                )
            except Exception:
                pass
            raise
            
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            error_msg = f"Order failed. Code: {result.retcode}, Comment: {result.comment}"
            
            try:
                from openquant.utils.alerts import send_alert
                send_alert(
                    subject=f"MT5 Order Failed: {mt5_symbol}",
                    body=error_msg,
                    severity="ERROR"
                )
            except ImportError as e:
                logger.debug(f"Alerts module not available: {e}")
            except Exception as e:
                logger.warning(f"Failed to send order failure alert: {e}")
                
            raise RuntimeError(error_msg)
        
        if result.price > 0 and arrival_price > 0:
            slippage_pct = abs(result.price - arrival_price) / arrival_price
            if slippage_pct > 0.001:
                try:
                    from openquant.utils.alerts import send_alert
                    send_alert(
                        subject=f"MT5 High Slippage: {mt5_symbol}",
                        body=f"Slippage {slippage_pct:.4%}. Expected {arrival_price}, Got {result.price}",
                        severity="WARNING"
                    )
                except ImportError as e:
                    logger.debug(f"Alerts module not available: {e}")
                except Exception as e:
                    logger.warning(f"Failed to send slippage alert: {e}")
        
        if self.tca:
            try:
                self.tca.log_order(
                    order_id=str(result.order),
                    symbol=mt5_symbol,
                    side=side,
                    quantity=volume,
                    arrival_price=arrival_price
                )
                
                if result.retcode == self.mt5.TRADE_RETCODE_DONE and result.price > 0:
                    self.tca.update_fill(
                        order_id=str(result.order),
                        fill_price=float(result.price),
                        fill_qty=volume,
                        fee=0.0
                    )
            except AttributeError as e:
                logger.warning(f"TCA logging failed - invalid method call: {e}")
            except Exception as e:
                logger.warning(f"Failed to log order to TCA: {e}")
        
        return {
            "id": str(result.order),
            "deal": str(result.deal) if hasattr(result, 'deal') else None,
            "status": "filled" if result.retcode == self.mt5.TRADE_RETCODE_DONE else "unknown",
            "price": float(result.price) if result.price else None,
            "volume": volume
        }

    def close_all_positions(self):
        """Liquidate all positions immediately."""
        self._ensure_connection_health()
        count = mt5_bridge.close_all_positions()
        return {"closed_positions": count}
    
    def shutdown(self):
        """Shutdown MT5 connection."""
        self.pool.shutdown()
