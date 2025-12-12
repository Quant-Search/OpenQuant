"""
MetaTrader 5 Broker Implementation.
"""
import os
import logging
from typing import Dict, List, Any, Optional
from openquant.broker.abstract import Broker

logger = logging.getLogger(__name__)

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
            
        self.login = login or (int(os.getenv("MT5_LOGIN")) if os.getenv("MT5_LOGIN") else None)
        self.password = password or os.getenv("MT5_PASSWORD")
        self.server = server or os.getenv("MT5_SERVER")
        self.terminal_path = terminal_path or os.getenv("MT5_TERMINAL_PATH")
        
        if not self.login or not self.password or not self.server:
            raise ValueError("MT5 credentials missing. Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env or pass as arguments.")
            
        if not mt5_bridge.init(
            login=self.login,
            password=self.password,
            server=self.server,
            terminal_path=self.terminal_path
        ):
            raise RuntimeError("Failed to initialize/login to MT5. Check credentials and ensure MT5 terminal is installed.")
        
        self.mt5 = mt5_bridge._lazy_import()
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

    def get_cash(self) -> float:
        """Return available cash balance."""
        try:
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
        
        result = self.mt5.order_send(request)
        
        if result is None:
            raise RuntimeError("MT5 order_send returned None")
            
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
        count = mt5_bridge.close_all_positions()
        return {"closed_positions": count}
    
    def shutdown(self):
        """Shutdown MT5 connection."""
        mt5_bridge.shutdown()
