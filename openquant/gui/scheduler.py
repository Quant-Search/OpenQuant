"""Background scheduler for Robot Auto-Pilot."""
from __future__ import annotations
import threading
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Callable, Dict, Any

from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)

class RobotScheduler:
    """Manages a background thread for continuous robot execution."""
    
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.is_running = False
        self.last_run_time: Optional[datetime] = None
        self.next_run_time: Optional[datetime] = None
        self.interval_minutes = 60
        self.status_message = "Stopped"
        self.error_message: Optional[str] = None
        
        # Configuration for the run
        self.run_config: Dict[str, Any] = {
            "top_n": 10,
            "fee_bps": 2.0,
            "slippage_bps": 5.0,
            "use_mt5": False,
            "mt5_creds": {}
        }

    def start(self, interval_minutes: int = 60, config: Dict[str, Any] = None):
        """Start the auto-pilot loop."""
        if self.is_running:
            LOGGER.warning("Scheduler already running")
            return
            
        if config:
            self.run_config.update(config)
            
        self.interval_minutes = interval_minutes
        self._stop_event.clear()
        self.is_running = True
        self.error_message = None
        self.status_message = "Starting..."
        
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        LOGGER.info(f"Robot Auto-Pilot started. Interval: {interval_minutes}m")

    def stop(self):
        """Stop the auto-pilot loop."""
        if not self.is_running:
            return
            
        self.status_message = "Stopping..."
        self._stop_event.set()
        # Don't join here to avoid blocking GUI if loop is sleeping long
        # The loop will exit on its own
        self.is_running = False
        self.status_message = "Stopped"
        self.next_run_time = None
        LOGGER.info("Robot Auto-Pilot stopped.")

    def _loop(self):
        """Main background loop."""
        while not self._stop_event.is_set():
            self.status_message = "Running Research..."
            self.error_message = None
            
            try:
                self._run_cycle()
                self.last_run_time = datetime.now()
                # Calculate next run
                sleep_secs = self.interval_minutes * 60
                self.next_run_time = datetime.now() + timedelta(seconds=sleep_secs)
                wake_time = self.next_run_time.strftime('%H:%M')
                self.status_message = f"Active (Sleeping until {wake_time})"
                
                # Sleep in chunks
                last_regime_check = datetime.now()
                current_trend = getattr(self, "_last_trend", "neutral")
                
                for _ in range(sleep_secs):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)
                    
                    # Intra-cycle Regime Check (every 1 hour)
                    if (datetime.now() - last_regime_check).total_seconds() > 3600:
                        LOGGER.info("Performing Intra-Cycle Regime Check...")
                        new_trend = self._check_global_trend()
                        if new_trend != current_trend and current_trend != "neutral":
                            LOGGER.warning(f"⚠️ REGIME FLIP DETECTED: {current_trend} -> {new_trend}")
                            self.status_message = f"⚠️ REGIME FLIP: {new_trend.upper()}"
                            # Optional: Trigger emergency re-evaluation here
                        last_regime_check = datetime.now()
                    
            except Exception as e:
                LOGGER.error(f"Auto-Pilot Cycle Error: {e}", exc_info=True)
                self.status_message = "Error (Retrying in 1m)"
                self.error_message = str(e)
                time.sleep(60) # Retry delay

    def _run_cycle(self):
        """Execute one full robot cycle: Research -> Allocation -> Execution."""
        LOGGER.info("Auto-Pilot: Starting Cycle")
        
        # Lazy imports to avoid circular deps
        from openquant.research.universe_runner import run_universe
        from openquant.paper.io import load_state, save_state
        from openquant.paper.simulator import MarketSnapshot, compute_rebalance_orders, execute_orders
        from openquant.storage.portfolio_db import connect as _pdb_connect, record_rebalance, OrderFill
        from openquant.paper.mt5_bridge import apply_allocation_to_mt5
        from openquant.risk.portfolio_guard import GUARD
        from pathlib import Path
        import json
        import ccxt
        
        cfg = self.run_config
        
        # 0. Risk Check (Pre-Flight)
        # We need current equity to check risk. Load from state or DB.
        state_path = Path("data")/"paper_state.json"
        try:
            state = load_state(state_path)
            # We need to estimate positions value. Since we don't have prices yet, 
            # we might use the last known value or just cash if it's the first run.
            # Better approach: The PortfolioGuard needs an equity value.
            # Let's assume 0 positions value if we can't get it, or load from a separate tracking file.
            # For now, let's use cash as a conservative estimate if we can't price positions.
            current_equity = state.cash 
            # If we have holdings, we should try to value them, but we haven't fetched prices yet.
            # This is a chicken-and-egg problem. 
            # Solution: Skip pre-flight check if we can't value, or use the post-exec check which has prices.
            # Or, just use cash + 0 and accept it might be lower than real equity.
            
            is_safe, reason = GUARD.on_cycle_start(current_equity)
            if not is_safe:
                LOGGER.error(f"RISK STOP: {reason}")
                self.stop()
                self.status_message = f"RISK STOP: {reason}"
                self.error_message = reason
                
                # Emergency Close
                if cfg.get("use_mt5"):
                    try:
                        from openquant.paper.mt5_bridge import close_all_positions
                        n = close_all_positions()
                        LOGGER.warning(f"Emergency Close: Liquidated {n} positions.")
                    except Exception as e:
                        LOGGER.error(f"Emergency Close Failed: {e}")
                return
        except Exception as e:
            LOGGER.warning(f"Could not check risk pre-flight (first run?): {e}")

        # 0.5 Global Trend Check
        global_trend = self._check_global_trend()

        # 1. Research
        LOGGER.info(f"Auto-Pilot: Running Universe Research (top_n={cfg['top_n']})...")
        if cfg.get("use_mt5"):
             # Configure MT5 source if needed
             pass 
             
        exchange = "mt5" if cfg.get("use_mt5") else "binance"
        # Map symbols if using Alpaca (BTC/USD -> BTC/USDT for Binance data)
        symbols = cfg.get("symbols")
        if symbols and exchange == "binance":
            mapped_symbols = []
            for s in symbols:
                if s.endswith("/USD"):
                    mapped_symbols.append(s.replace("/USD", "/USDT"))
                else:
                    mapped_symbols.append(s)
            symbols = mapped_symbols

        run_universe(
            exchange=exchange, 
            top_n=int(cfg["top_n"]), 
            global_trend=global_trend,
            symbols=symbols
        )
        
        # 2. Load Allocation
        # Find latest allocation file
        r_dir = Path("reports")
        files = sorted(r_dir.glob("allocation_*.json"))
        if not files:
            LOGGER.warning("Auto-Pilot: No allocation file found after research.")
            return
        
        alloc_path = files[-1]
        with open(alloc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            alloc = data.get("allocations", []) if isinstance(data, dict) else data
            
        if not alloc:
            LOGGER.info("Auto-Pilot: Empty allocation, nothing to trade.")
            return

        # 2.5 Export Signals for Visualizer
        try:
            from openquant.paper.mt5_bridge import export_signals_to_csv
            # If MT5 path is configured, try to write to MQL5/Files if possible, 
            # otherwise write to local data dir and user has to symlink/copy.
            # For now, write to local data/signals.csv
            export_signals_to_csv(alloc)
        except Exception as e:
            LOGGER.warning(f"Failed to export signals: {e}")

        # 3. Paper Trading Execution
        LOGGER.info("Auto-Pilot: Executing Paper Trades...")
        state_path = Path("data")/"paper_state.json"
        state = load_state(state_path)
        
        # Build snapshot
        prices = {}
        # Initialize MT5 if needed for pricing
        mt5 = None
        if cfg.get("use_mt5"):
            try:
                import MetaTrader5 as _mt5
                mt5 = _mt5
                creds = cfg.get("mt5_creds", {})
                if not mt5.initialize(path=creds.get("path")):
                     LOGGER.warning(f"MT5 init failed: {mt5.last_error()}")
                elif creds.get("login"):
                     mt5.login(int(creds["login"]), password=creds.get("password"), server=creds.get("server"))
            except Exception as e:
                LOGGER.warning(f"MT5 init exception: {e}")

        clients = {}
        for e in alloc:
            sym = str(e.get("symbol",""))
            ex = str(e.get("exchange", "binance")).lower()
            tf = str(e.get("timeframe",""))
            strat = str(e.get("strategy",""))
            if not sym: continue
            
            px = 0.0
            if ex == "mt5" and mt5:
                try:
                    tick = mt5.symbol_info_tick(sym)
                    px = float(getattr(tick, "last", 0.0) or getattr(tick, "bid", 0.0) or getattr(tick, "ask", 0.0) or 0.0)
                except Exception:
                    px = 0.0
            else:
                # CCXT fallback
                if ex not in clients:
                    try:
                        clients[ex] = getattr(ccxt, ex)()
                    except:
                        continue
                try:
                    t = clients[ex].fetch_ticker(sym)
                    px = float(t.get("last") or t.get("close") or 0.0)
                except Exception:
                    px = 0.0
            
            prices[(ex.upper(), sym, tf, strat)] = px

        snap = MarketSnapshot(prices=prices)
        
        # Compute targets
        targets = []
        for e in alloc:
            key = (str(e.get("exchange","binance")).upper(), str(e.get("symbol","")), str(e.get("timeframe","")), str(e.get("strategy","")))
            w = float(e.get("weight",0.0))
            if w > 0:
                targets.append((key, w))
                
        orders = compute_rebalance_orders(state, targets, snap)
        summary, fills_raw = execute_orders(state, orders, fee_bps=float(cfg.get("fee_bps", 2.0)), slippage_bps=float(cfg.get("slippage_bps", 5.0)))
        
        # Record to DB
        db_path = Path("data/results.duckdb")
        con = _pdb_connect(db_path)
        try:
            fills = []
            for (k, du, ex_px, fee_paid) in fills_raw:
                fills.append(OrderFill(
                    key=k, 
                    side=("BUY" if du>0 else "SELL"), 
                    delta_units=float(du), 
                    exec_price=float(ex_px), 
                    notional=float(abs(du)*ex_px), 
                    fee_bps=float(cfg.get("fee_bps", 2.0)), 
                    slippage_bps=float(cfg.get("slippage_bps", 5.0)), 
                    fee_paid=float(fee_paid)
                ))
            record_rebalance(con, ts=datetime.now(timezone.utc), fills=fills, state=state, snap=snap)
        finally:
            con.close()
            
        save_state(state, state_path)
        LOGGER.info(f"Auto-Pilot: Paper execution done. Orders: {summary['orders']}, Turnover: {summary['turnover']:.2f}")

        # 3.5 Update Risk State
        try:
            # Calculate equity using the snapshot prices we just used
            pos_val = 0.0
            holdings_map = {}
            for k, units in state.holdings.items():
                # k is (Exchange, Symbol, Timeframe, Strategy)
                # snap.prices keys are (Exchange, Symbol, Timeframe, Strategy)
                price = snap.prices.get(k, 0.0)
                val = units * price
                pos_val += val
                
                # Aggregate value by symbol for exposure check
                sym = k[1]
                holdings_map[sym] = holdings_map.get(sym, 0.0) + val
            
            new_equity = state.cash + pos_val
            is_safe, reason = GUARD.on_cycle_start(new_equity, holdings=holdings_map) # Re-check to update HWM/Daily stats
            if not is_safe:
                LOGGER.error(f"RISK STOP (Post-Exec): {reason}")
                self.stop()
                self.status_message = f"RISK STOP: {reason}"
                self.error_message = reason
                
                # Emergency Close
                if cfg.get("use_mt5"):
                    try:
                        from openquant.paper.mt5_bridge import close_all_positions
                        n = close_all_positions()
                        LOGGER.warning(f"Emergency Close: Liquidated {n} positions.")
                    except Exception as e:
                        LOGGER.error(f"Emergency Close Failed: {e}")
        except Exception as e:
            LOGGER.error(f"Failed to update risk state: {e}")

        # 4. MT5 Execution (if enabled)
        # 4. MT5 Execution (if enabled)
        if cfg.get("use_mt5"):
            LOGGER.info("Auto-Pilot: Syncing with MT5...")
            creds = cfg.get("mt5_creds", {})
            try:
                apply_allocation_to_mt5(
                    alloc, 
                    terminal_path=creds.get("path"),
                    login=(int(creds["login"]) if creds.get("login") else None),
                    password=creds.get("password"),
                    server=creds.get("server")
                )
            except Exception as e:
                LOGGER.error(f"MT5 Sync Failed: {e}")

        # 5. Alpaca Execution (if enabled)
        if cfg.get("use_alpaca"):
            LOGGER.info("Auto-Pilot: Syncing with Alpaca...")
            try:
                from openquant.broker.alpaca_broker import AlpacaBroker
                broker = AlpacaBroker(
                    api_key=cfg.get("alpaca_key"), 
                    secret_key=cfg.get("alpaca_secret"), 
                    paper=cfg.get("alpaca_paper", True)
                )
                
                # Simple Sync: 
                # 1. Get current Alpaca positions
                current_pos = broker.get_positions()
                
                # 2. Calculate Target positions from 'alloc'
                # Alloc is list of dicts: {'symbol': 'BTC/USDT', 'weight': 0.5, ...}
                # We need to convert weights to quantities based on current equity
                equity = broker.get_equity()
                
                # Group targets by symbol (sum weights if multiple strategies trade same symbol)
                target_weights = {}
                for item in alloc:
                    sym = item.get("symbol")
                    # Map symbol if needed (e.g. BTC/USDT -> BTCUSD for Alpaca?)
                    # Alpaca uses "BTC/USD" or "BTCUSD" depending on asset class. 
                    # For crypto, it's usually "BTC/USD".
                    # Let's assume symbols are compatible or user maps them.
                    w = float(item.get("weight", 0.0))
                    target_weights[sym] = target_weights.get(sym, 0.0) + w
                
                # 3. Generate Orders (Diff)
                # Simple Rebalancing Logic:
                # Target Value = Equity * Target Weight
                # Current Value = Current Qty * Price
                # Delta Value = Target Value - Current Value
                # Delta Qty = Delta Value / Price
                
                for sym, target_weight in target_weights.items():
                    # Get Price (from snapshot if available, else fetch)
                    # We use the snapshot prices we collected earlier
                    # We need to find the price for this symbol (ignoring strategy/tf keys)
                    price = 0.0
                    for k, p in snap.prices.items():
                        if k[1] == sym:
                            price = p
                            break
                    
                    if price <= 0:
                        # Fallback: try to fetch from broker if possible or skip
                        # Alpaca broker doesn't expose get_price directly yet, but we can try
                        # For now, skip if no price
                        LOGGER.warning(f"Alpaca: No price for {sym}, skipping.")
                        continue
                        
                    current_qty = float(current_pos.get(sym, 0.0))
                    current_val = current_qty * price
                    target_val = equity * target_weight
                    
                    delta_val = target_val - current_val
                    
                    # Threshold to avoid dust trades (e.g. < $10)
                    if abs(delta_val) < 10.0:
                        continue
                        
                    delta_qty = delta_val / price
                    side = "buy" if delta_qty > 0 else "sell"
                    qty_abs = abs(delta_qty)
                    
                    # Round quantity to reasonable precision (e.g. 4 decimals for crypto)
                    # Alpaca handles fractional shares for many assets
                    qty_abs = round(qty_abs, 5)
                    
                    if qty_abs == 0:
                        continue
                        
                    LOGGER.info(f"Alpaca Order: {side.upper()} {qty_abs} {sym} (Delta: ${delta_val:.2f})")
                    
                    try:
                        broker.place_order(
                            symbol=sym,
                            quantity=qty_abs,
                            side=side,
                            order_type="market"
                        )
                    except Exception as oe:
                        LOGGER.error(f"Alpaca Order Failed {sym}: {oe}")

                # Sync TCA after orders
                broker.sync_tca()
                
            except Exception as e:
                LOGGER.error(f"Alpaca Sync Failed: {e}")
            
        LOGGER.info("Auto-Pilot: Cycle Complete")

    def _check_global_trend(self) -> str:
        """Check global market trend (BTC/USDT SMA200)."""
        global_trend = "neutral"
        try:
            import ccxt
            import pandas as pd
            
            # Use a fresh instance or cached one
            ex_trend = ccxt.binance()
            # Map BTC/USD to BTC/USDT if needed
            sym = "BTC/USDT"
            ohlcv = ex_trend.fetch_ohlcv(sym, timeframe="1d", limit=250)
            if ohlcv:
                df_trend = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                close = df_trend["close"]
                sma200 = close.rolling(200).mean().iloc[-1]
                current_price = close.iloc[-1]
                
                if current_price < sma200:
                    global_trend = "bear"
                elif current_price > sma200:
                    global_trend = "bull"
                    
                LOGGER.info(f"Global Trend (BTC/USDT): {global_trend.upper()} (Price: {current_price}, SMA200: {sma200})")
        except Exception as e:
            LOGGER.warning(f"Could not check global trend: {e}")
        return global_trend

# Global instance
SCHEDULER = RobotScheduler()
