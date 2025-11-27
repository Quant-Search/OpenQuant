"""
MetaTrader 5 Broker Implementation.
"""
import os
from typing import Dict, List, Any, Optional
from openquant.broker.abstract import Broker

try:
    from openquant.paper import mt5_bridge
    MT5_AVAILABLE = mt5_bridge.is_available()
except ImportError:
    MT5_AVAILABLE = False


class MT5Broker(Broker):
    """
    MetaTrader 5 Broker implementation using mt5_bridge.
    """
    def __init__(self, 
                 login: Optional[int] = None, 
                 password: Optional[str] = None, 
                 server: Optional[str] = None,
                 terminal_path: Optional[str] = None):
        if not MT5_AVAILABLE:
            raise ImportError("MetaTrader5 not available. Install it or run on Windows.")
            
        # Get credentials from environment if not provided
        self.login = login or (int(os.getenv("MT5_LOGIN")) if os.getenv("MT5_LOGIN") else None)
        self.password = password or os.getenv("MT5_PASSWORD")
        self.server = server or os.getenv("MT5_SERVER")
        self.terminal_path = terminal_path or os.getenv("MT5_TERMINAL_PATH")
        
        # Validate credentials
        if not self.login or not self.password or not self.server:
            raise ValueError("MT5 credentials missing. Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env or pass as arguments.")
            
        # Initialize MT5 connection
        if not mt5_bridge.init(
            login=self.login,
            password=self.password,
            server=self.server,
            terminal_path=self.terminal_path
        ):
            raise RuntimeError("Failed to initialize/login to MT5. Check credentials and ensure MT5 terminal is installed.")
        
        # Get the MT5 module reference
        self.mt5 = mt5_bridge._lazy_import()
        if not self.mt5:
            raise RuntimeError("MT5 module not available after initialization")
            
        # Initialize TCA
        try:
            from openquant.analysis.tca import TCAMonitor
            self.tca = TCAMonitor()
        except ImportError:
            self.tca = None

    def get_cash(self) -> float:
        """Return available cash balance."""
        try:
            account = self.mt5.account_info()
            if account:
                return float(account.balance)
            return 0.0
        except Exception:
            return 0.0

    def get_equity(self) -> float:
        """Return total account equity (cash + positions)."""
        return mt5_bridge.account_equity(self.mt5)

    def get_positions(self) -> Dict[str, float]:
        """
        Return current positions.
        Format: {"EURUSD": 0.5, "GBPUSD": -0.3}
        """
        return mt5_bridge.positions_by_symbol(self.mt5)

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
        """
        # Map symbol if needed (e.g., BTC/USDT -> BTCUSD)
        mt5_symbol = mt5_bridge.map_symbol(symbol)
        
        # Ensure symbol is available
        if not mt5_bridge._ensure_symbol(self.mt5, mt5_symbol):
            raise ValueError(f"Symbol {mt5_symbol} not available in MT5")
        
        # Get symbol info for validation
        info = self.mt5.symbol_info(mt5_symbol)
        if not info:
            raise ValueError(f"Cannot get symbol info for {mt5_symbol}")
            
        # Validate and round volume
        vmin = float(getattr(info, "volume_min", 0.01) or 0.01)
        vstep = float(getattr(info, "volume_step", 0.01) or 0.01)
        vmax = float(getattr(info, "volume_max", 100.0) or 100.0)
        
        volume = min(max(mt5_bridge._round_step(abs(quantity), vstep), vmin), vmax)
        
        # Determine order type
        mt5_order_type = self.mt5.ORDER_TYPE_BUY if side.lower() == "buy" else self.mt5.ORDER_TYPE_SELL
        
        # Get current price for arrival tracking
        tick = self.mt5.symbol_info_tick(mt5_symbol)
        if not tick:
            raise RuntimeError(f"Cannot get tick data for {mt5_symbol}")
            
        arrival_price = float(tick.ask if side.lower() == "buy" else tick.bid)
        
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
        
        # Send order
        result = self.mt5.order_send(request)
        
        if result is None:
            raise RuntimeError("MT5 order_send returned None")
            
        # Check result
        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            error_msg = f"Order failed. Code: {result.retcode}, Comment: {result.comment}"
            
            # Send alert if available
            try:
                from openquant.utils.alerts import send_alert
                send_alert(
                    subject=f"MT5 Order Failed: {mt5_symbol}",
                    body=error_msg,
                    severity="ERROR"
                )
            except Exception:
                pass
                
            raise RuntimeError(error_msg)
        
        # Check for high slippage
        if result.price > 0 and arrival_price > 0:
            slippage_pct = abs(result.price - arrival_price) / arrival_price
            if slippage_pct > 0.001:  # 0.1% threshold
                try:
                    from openquant.utils.alerts import send_alert
                    send_alert(
                        subject=f"MT5 High Slippage: {mt5_symbol}",
                        body=f"Slippage {slippage_pct:.4%}. Expected {arrival_price}, Got {result.price}",
                        severity="WARNING"
                    )
                except Exception:
                    pass
        
        # Log to TCA if available
        if self.tca:
            try:
                self.tca.log_order(
                    order_id=str(result.order),
                    symbol=mt5_symbol,
                    side=side,
                    quantity=volume,
                    arrival_price=arrival_price
                )
                
                # Immediately update with fill if order is filled
                if result.retcode == self.mt5.TRADE_RETCODE_DONE and result.price > 0:
                    self.tca.update_fill(
                        order_id=str(result.order),
                        fill_price=float(result.price),
                        fill_qty=volume,
                        fee=0.0  # MT5 doesn't expose commission in result, would need to query separately
                    )
            except Exception:
                pass
        
        return {
            "id": str(result.order),
            "deal": str(result.deal) if hasattr(result, 'deal') else None,
            "status": "filled" if result.retcode == self.mt5.TRADE_RETCODE_DONE else "unknown",
            "price": float(result.price) if result.price else None,
            "volume": volume
        }

    def close_all_positions(self):
        """Liquidate all positions immediately."""
        count = mt5_bridge.close_all_positions()
        return {"closed_positions": count}
    
    def shutdown(self):
        """Shutdown MT5 connection."""
        mt5_bridge.shutdown()
