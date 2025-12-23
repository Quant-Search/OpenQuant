"""
Trader Module

Single Responsibility: Only handles trade execution (paper or live).
"""
from typing import Optional, Dict
from datetime import datetime, timezone


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
        self._paper_cash = 10000.0  # Starting paper cash
        self._paper_pnl = 0.0  # Realized P&L from closed trades
        
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
    
    def _get_paper_price(self, symbol: str) -> float:
        """Get simulated price for paper trading."""
        # If we have a stored current price, use it
        if symbol in self._paper_positions:
            return self._paper_positions[symbol].get("current_price", 1.0)
        # Default prices for common forex pairs (rough estimates)
        defaults = {
            "EURUSD": 1.10, "GBPUSD": 1.27, "USDJPY": 150.0,
            "USDCHF": 0.88, "AUDUSD": 0.67, "USDCAD": 1.35
        }
        return defaults.get(symbol, 1.0)
    
    def update_paper_prices(self, prices: Dict[str, float]):
        """Update current prices for paper positions (call after fetching data)."""
        for symbol, price in prices.items():
            if symbol in self._paper_positions:
                self._paper_positions[symbol]["current_price"] = price
    
    def get_equity(self) -> float:
        """Get account equity (cash + unrealized P&L)."""
        if self.mode == "paper":
            # Calculate unrealized P&L from open positions
            unrealized_pnl = 0.0
            for symbol, pos in self._paper_positions.items():
                entry_price = pos.get("entry_price", 0)
                current_price = pos.get("current_price", entry_price)
                volume = pos.get("volume", 0)
                # P&L = (current - entry) * volume * contract_size
                # For forex, 1 lot = 100,000 units
                contract_size = 100000.0
                if volume > 0:  # LONG
                    unrealized_pnl += (current_price - entry_price) * abs(volume) * contract_size
                else:  # SHORT
                    unrealized_pnl += (entry_price - current_price) * abs(volume) * contract_size
            return self._paper_cash + self._paper_pnl + unrealized_pnl
            
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
            # Get simulated entry price (use last known price or estimate)
            entry_price = self._get_paper_price(symbol)
            
            # If we have an existing position, close it first (realize P&L)
            if symbol in self._paper_positions:
                old_pos = self._paper_positions[symbol]
                old_volume = old_pos.get("volume", 0)
                old_entry = old_pos.get("entry_price", entry_price)
                contract_size = 100000.0
                
                # Calculate realized P&L from closing old position
                if old_volume > 0:  # Was LONG
                    realized = (entry_price - old_entry) * abs(old_volume) * contract_size
                else:  # Was SHORT
                    realized = (old_entry - entry_price) * abs(old_volume) * contract_size
                
                self._paper_pnl += realized
                print(f"[PAPER] Closed {symbol} position, realized P&L: ${realized:.2f}")
            
            # Open new position
            self._paper_positions[symbol] = {
                "volume": volume if side == "BUY" else -volume,
                "side": side,
                "entry_price": entry_price,
                "current_price": entry_price,  # Will be updated on next cycle
                "sl": stop_loss,
                "tp": take_profit,
                "entry_time": datetime.now(timezone.utc)
            }
            print(f"[PAPER] Opened {side} {volume:.2f} {symbol} @ {entry_price:.5f}")
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


