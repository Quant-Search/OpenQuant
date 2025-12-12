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
        
        # Position monitor for continuous trade surveillance
        self._position_monitor = None
        self._mt5_broker_instance = None
        
        # Adaptive sizing for profit optimization
        self._adaptive_sizer = None
        
        # Trade filter for regime-aware filtering
        self._trade_filter = None
        
        # Performance tracker
        self._perf_tracker = None
        
        # Configuration for the run
        self.run_config: Dict[str, Any] = {
            "top_n": 10,
            "fee_bps": 2.0,
            "slippage_bps": 5.0,
            "use_mt5": True,
            "mt5_creds": {},
            "symbols": [
                "BTC/USDT",
                "ETH/USDT",
                "SOL/USDT",
                "BNB/USDT"
            ],
            "auto_apply_actions": True,
            "apply_actions_to_live": True,
            "position_monitoring": {
                "enabled": True,
                "check_interval_seconds": 60,
                "trailing_activation_bps": 50,
                "trailing_distance_bps": 30
            },
            # Trading config - using profitable MR strategy settings
            "aggressive_mode": False,  # Conservative for real trading
            "max_drawdown": 0.20,
            "min_probability": 0.62,
            "strict_session_filter": True,
            "min_session_quality": 0.6,
            "use_regime_filter": True,
            "strategy": "mean_reversion"  # Profitable strategy choice
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
        
        # Stop position monitor if running
        if self._position_monitor:
            try:
                self._position_monitor.stop()
            except Exception as e:
                LOGGER.error(f"Error stopping position monitor: {e}")
        
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
                            LOGGER.warning(f"REGIME FLIP DETECTED: {current_trend} -> {new_trend}")
                            self.status_message = f"REGIME FLIP: {new_trend.upper()}"
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
        from openquant.risk.circuit_breaker import CIRCUIT_BREAKER
        from openquant.risk.kill_switch import KILL_SWITCH
        from openquant.risk.market_hours import MarketHours, MarketType
        from openquant.risk.adaptive_sizing import AdaptiveSizer
        from openquant.trading.trade_filter import TradeFilter, FilterConfig, TradeSignal, FilterResult
        from openquant.reporting.performance_tracker import PERFORMANCE_TRACKER
        from pathlib import Path
        import json
        import ccxt

        cfg = self.run_config
        
        # Initialize profit optimization components (once)
        if self._adaptive_sizer is None:
            self._adaptive_sizer = AdaptiveSizer(
                method="kelly",
                max_drawdown=cfg.get("max_drawdown", 0.50),
                aggressive_mode=cfg.get("aggressive_mode", True)
            )
            LOGGER.info("AdaptiveSizer initialized (aggressive_mode=%s)", cfg.get("aggressive_mode", True))
            
        if self._trade_filter is None:
            filter_cfg = FilterConfig(
                min_probability=cfg.get("min_probability", 0.60),
                allow_counter_trend=False
            )
            self._trade_filter = TradeFilter(config=filter_cfg)
            LOGGER.info("TradeFilter initialized (min_prob=%s)", cfg.get("min_probability", 0.60))
            
        self._perf_tracker = PERFORMANCE_TRACKER

        # 0a. Kill Switch Check (Highest Priority)
        if KILL_SWITCH.is_active():
            LOGGER.error("KILL SWITCH ACTIVE - Trading halted")
            self.status_message = "KILL SWITCH ACTIVE"
            self.error_message = "Remove data/STOP file to resume"
            return

        # 0b. Circuit Breaker Check
        if CIRCUIT_BREAKER.is_tripped():
            status = CIRCUIT_BREAKER.get_status()
            LOGGER.error(f"CIRCUIT BREAKER TRIPPED - Trading halted: {status}")
            self.status_message = "CIRCUIT BREAKER TRIPPED"
            self.error_message = str(status)
            return

        preferred_market = MarketType.CRYPTO
        effective_market = preferred_market
        effective_exchange = "binance"
        if cfg.get("use_mt5"):
            preferred_market = MarketType.FOREX
            effective_exchange = "mt5"
        elif cfg.get("use_alpaca"):
            preferred_market = MarketType.US_STOCKS
            effective_exchange = "alpaca"

        market_hours = MarketHours(preferred_market)
        mt5_exec_enabled = bool(cfg.get("use_mt5")) and market_hours.is_open()
        alpaca_exec_enabled = bool(cfg.get("use_alpaca")) and market_hours.is_open()
        if preferred_market != MarketType.CRYPTO and not market_hours.is_open():
            next_open = market_hours.next_open()
            LOGGER.info(f"Market closed. Switching to crypto. Next {preferred_market.name} open: {next_open}")
            self.status_message = "Market closed. Trading crypto"
            effective_market = MarketType.CRYPTO
            effective_exchange = "binance"
        else:
            effective_market = preferred_market

        # Apply live adjustments from diagnostics
        try:
            if bool(cfg.get("apply_actions_to_live", True)):
                from pathlib import Path as _P
                import json as _json
                diag = {}
                cfgp = {"wfo_drop": 0.2, "profit_factor_min": 1.2, "mc_dd_p95_max": 0.25, "p_value_max": 0.10}
                pth = _P("data/diagnostic_report.json")
                if pth.exists():
                    with open(pth, "r") as f:
                        diag = _json.load(f)
                pth2 = _P("data/diagnostics_config.json")
                if pth2.exists():
                    try:
                        with open(pth2, "r") as f:
                            cfgp.update(_json.load(f))
                    except Exception:
                        pass
                agg = diag.get("aggregate", {})
                mcdd = float(agg.get("mean_mc_dd_p95", 0) or 0)
                pfm = float(agg.get("mean_profit_factor", 0) or 0)
                wfo = float(agg.get("mean_wfo_mts", 0) or 0)
                pval = float(agg.get("median_p_value", 1) or 1)
                if mcdd > float(cfgp.get("mc_dd_p95_max", 0.25)):
                    cfg["aggressive_mode"] = False
                    cfg["max_drawdown"] = min(0.15, float(cfg.get("max_drawdown", 0.20)))
                if pfm < float(cfgp.get("profit_factor_min", 1.2)):
                    cfg["min_probability"] = float(min(0.95, float(cfg.get("min_probability", 0.60)) + 0.02))
                if wfo < float(cfgp.get("wfo_drop", 0.2)) or pval > float(cfgp.get("p_value_max", 0.10)):
                    cfg["strict_session_filter"] = True
        except Exception:
            pass

        # 0d. Session Quality Check
        try:
            from openquant.trading.session_optimizer import SessionOptimizer
            session_opt = SessionOptimizer()
            session_name, session_quality = session_opt.get_current_session()
            
            min_session_quality = cfg.get("min_session_quality", 0.4)
            
            if session_quality < min_session_quality:
                LOGGER.info(f"Low session quality: {session_name} ({session_quality:.0%})")
                self.status_message = f"Waiting for better session (current: {session_name} {session_quality:.0%})"
                if cfg.get("strict_session_filter", False) and effective_exchange != "binance":
                    return
            else:
                LOGGER.info(f"Good session: {session_name} ({session_quality:.0%})")
        except Exception as e:
            LOGGER.warning(f"Session optimizer error: {e}")

        # 0d. Risk Check (Pre-Flight)
        state_path = Path("data")/"paper_state.json"
        try:
            state = load_state(state_path)
            current_equity = state.cash

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
        if cfg.get("use_mt5") and effective_exchange == "mt5":
            pass

        exchange = effective_exchange
        # Map symbols if using Alpaca (BTC/USD -> BTC/USDT for Binance data)
        symbols = cfg.get("symbols")
        if symbols is None and exchange == "binance":
            symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
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
            symbols=symbols,
            use_opt_actions=bool(cfg.get("auto_apply_actions", True))
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

        # 2.6 Live Execution on MT5 (optionnel)
        try:
            if mt5_exec_enabled and bool(cfg.get("use_mt5")):
                from openquant.paper.mt5_bridge import apply_allocation_to_mt5
                creds = cfg.get("mt5_creds", {})
                apply_allocation_to_mt5(
                    alloc,
                    terminal_path=creds.get("path"),
                    login=(int(creds["login"]) if creds.get("login") else None),
                    password=creds.get("password"),
                    server=creds.get("server")
                )
                LOGGER.info("MT5 execution applied successfully")
        except Exception as e:
            LOGGER.warning(f"MT5 execution failed: {e}")

        # 3. Paper Trading Execution
        LOGGER.info("Auto-Pilot: Executing Paper Trades...")
        state_path = Path("data")/"paper_state.json"
        state = load_state(state_path)
        
        # Build snapshot
        prices = {}
        # Initialize MT5 if needed for pricing
        # Initialize MT5 if needed for pricing
        mt5_broker = None
        if cfg.get("use_mt5"):
            try:
                from openquant.broker.mt5_broker import MT5Broker
                # Initialize broker (reads creds from env if not in config)
                creds = cfg.get("mt5_creds", {})
                mt5_broker = MT5Broker(
                    login=(int(creds["login"]) if creds.get("login") else None),
                    password=creds.get("password"),
                    server=creds.get("server"),
                    terminal_path=creds.get("path")
                )
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
            px = 0.0
            if ex == "mt5" and mt5_broker:
                try:
                    # Use broker's internal MT5 reference or add a get_price method
                    # For now, access .mt5 directly as we know it exists on the broker
                    tick = mt5_broker.mt5.symbol_info_tick(sym)
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

        for k in list(state.holdings.keys()):
            ex_k, sym_k, tf_k, strat_k = k
            key = (str(ex_k).upper(), str(sym_k), str(tf_k), str(strat_k))
            if key in prices and float(prices.get(key, 0.0)) > 0.0:
                continue
            ex_l = str(ex_k).lower()
            px2 = 0.0
            if ex_l == "mt5" and mt5_broker:
                try:
                    tick = mt5_broker.mt5.symbol_info_tick(sym_k)
                    px2 = float(getattr(tick, "last", 0.0) or getattr(tick, "bid", 0.0) or getattr(tick, "ask", 0.0) or 0.0)
                except Exception:
                    px2 = 0.0
            else:
                if ex_l not in clients:
                    try:
                        clients[ex_l] = getattr(ccxt, ex_l)()
                    except:
                        continue
                try:
                    t2 = clients[ex_l].fetch_ticker(sym_k)
                    px2 = float(t2.get("last") or t2.get("close") or 0.0)
                except Exception:
                    px2 = 0.0
            prices[key] = px2

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

            # 3.6 Update Circuit Breaker with current equity
            CIRCUIT_BREAKER.update(current_equity=new_equity)
            if CIRCUIT_BREAKER.is_tripped():
                LOGGER.error("CIRCUIT BREAKER TRIPPED after execution")
                self.stop()
                self.status_message = "CIRCUIT BREAKER TRIPPED"
                self.error_message = str(CIRCUIT_BREAKER.get_status())
                return

        except Exception as e:
            LOGGER.error(f"Failed to update risk state: {e}")

        # 4. MT5 Execution (if enabled)
        # 4. MT5 Execution (if enabled)
        # 4. MT5 Execution (if enabled)
        if mt5_exec_enabled and mt5_broker:
            LOGGER.info("Auto-Pilot: Syncing with MT5...")
            try:
                # Reuse the broker instance we created for pricing
                broker = mt5_broker
                
                # 1. Get current positions
                current_pos = broker.get_positions()
                
                # 2. Calculate Target positions
                equity = broker.get_equity()
                
                target_weights = {}
                for item in alloc:
                    sym = item.get("symbol")
                    w = float(item.get("weight", 0.0))
                    target_weights[sym] = target_weights.get(sym, 0.0) + w
                
                # 3. Generate Orders
                for sym, target_weight in target_weights.items():
                    # Map symbol to MT5 format if needed (e.g. BTC/USDT -> BTCUSD)
                    # The broker.place_order handles mapping, but we need price for delta calc.
                    # We can use the broker's internal mapping helper or just rely on the snapshot price 
                    # if the key matches.
                    
                    # Find price in snapshot
                    price = 0.0
                    for k, p in snap.prices.items():
                        if k[1] == sym:
                            price = p
                            break
                    
                    if price <= 0:
                        LOGGER.warning(f"MT5: No price for {sym}, skipping.")
                        continue
                        
                    # MT5Broker returns positions with mapped symbols (e.g. BTCUSD)
                    # We need to match 'sym' (e.g. BTC/USDT) to that.
                    # Let's try to map 'sym' to what MT5 would return.
                    from openquant.paper import mt5_bridge
                    mt5_sym = mt5_bridge.map_symbol(sym)
                    
                    # Get symbol info for contract size
                    info = broker.mt5.symbol_info(mt5_sym)
                    if not info:
                        LOGGER.warning(f"MT5: Cannot get symbol info for {mt5_sym}, skipping.")
                        continue
                    
                    # Contract size (e.g., 100,000 for Forex)
                    contract_size = float(getattr(info, "trade_contract_size", 100000.0) or 100000.0)
                    
                    current_qty = float(current_pos.get(mt5_sym, 0.0))
                    current_val = current_qty * price * contract_size
                    target_val = equity * target_weight
                    
                    delta_val = target_val - current_val
                    
                    # Threshold ($10)
                    if abs(delta_val) < 10.0:
                        continue
                        
                    # Calculate lots needed
                    # delta_val = lots * price * contract_size
                    # lots = delta_val / (price * contract_size)
                    delta_lots = delta_val / (price * contract_size)
                    side = "buy" if delta_lots > 0 else "sell"
                    lots_abs = abs(delta_lots)
                    
                    # Calculate TP/SL using dynamic exits (ATR-based)
                    tp_price = None
                    sl_price = None
                    
                    try:
                        from openquant.trading.dynamic_exits import DynamicExitCalculator
                        from openquant.data import mt5_source
                        
                        # Fetch recent price data for ATR calculation
                        df = mt5_source.fetch_ohlcv(mt5_sym, timeframe="1h", limit=100)
                        
                        if not df.empty and len(df) >= 20:
                            # Calculate TP/SL
                            exit_calc = DynamicExitCalculator(
                                method="atr",
                                tp_atr_multiplier=3.0,
                                sl_atr_multiplier=1.5,
                                atr_period=14
                            )
                            
                            tp_price, sl_price = exit_calc.calculate_exits(
                                df=df,
                                entry_price=price,
                                side="LONG" if side == "buy" else "SHORT"
                            )
                            
                            LOGGER.info(f"MT5 Order: {side.upper()} {lots_abs:.4f} lots {mt5_sym} (Delta: ${delta_val:.2f}) | TP: {tp_price:.5f}, SL: {sl_price:.5f}")
                        else:
                            LOGGER.warning(f"Insufficient data for TP/SL calculation on {mt5_sym}, using order without TP/SL")
                            LOGGER.info(f"MT5 Order: {side.upper()} {lots_abs:.4f} lots {mt5_sym} (Delta: ${delta_val:.2f})")
                            
                    except Exception as e:
                        LOGGER.warning(f"Failed to calculate TP/SL for {mt5_sym}: {e}, proceeding without TP/SL")
                        LOGGER.info(f"MT5 Order: {side.upper()} {lots_abs:.4f} lots {mt5_sym} (Delta: ${delta_val:.2f})")
                    
                    try:
                        # Place order with TP/SL
                        result = broker.place_order(
                            symbol=sym, # Broker handles mapping
                            quantity=lots_abs,
                            side=side,
                            order_type="market"
                        )
                        
                        # Set TP/SL on the position if calculated
                        if tp_price is not None and sl_price is not None:
                            try:
                                from openquant.paper.mt5_bridge import modify_position
                                modify_position(mt5_sym, sl=sl_price, tp=tp_price)
                                LOGGER.info(f"Set TP/SL for {mt5_sym}: TP={tp_price:.5f}, SL={sl_price:.5f}")
                            except Exception as e:
                                LOGGER.error(f"Failed to set TP/SL for {mt5_sym}: {e}")
                                
                    except Exception as oe:
                        LOGGER.error(f"MT5 Order Failed {sym}: {oe}")
                        
            except Exception as e:
                LOGGER.error(f"MT5 Sync Failed: {e}")
                
            # Store broker instance for position monitoring
            if mt5_broker:
                self._mt5_broker_instance = mt5_broker
                
                # Start position monitor if enabled
                if cfg.get("position_monitoring", {}).get("enabled", True):
                    try:
                        from openquant.trading.position_monitor import PositionMonitor
                        from openquant.risk.trailing_stop import TrailingStopManager
                        
                        # Stop existing monitor if running
                        if self._position_monitor:
                            self._position_monitor.stop()
                            
                        # Create new monitor with config
                        pm_cfg = cfg.get("position_monitoring", {})
                        trailing_mgr = TrailingStopManager(
                            trailing_bps=pm_cfg.get("trailing_distance_bps", 30),
                            activation_bps=pm_cfg.get("trailing_activation_bps", 50)
                        )
                        
                        self._position_monitor = PositionMonitor(
                            check_interval_seconds=pm_cfg.get("check_interval_seconds", 60),
                            trailing_stop_manager=trailing_mgr
                        )
                        
                        # Start monitoring
                        self._position_monitor.start(mt5_broker)
                        LOGGER.info("Position monitor started")
                        
                        # Immediately check positions (don't wait 60 seconds)
                        try:
                            metrics = self._position_monitor.get_all_metrics()
                            if metrics:
                                LOGGER.info(f"Currently monitoring {len(metrics)} position(s):")
                                for m in metrics:
                                    LOGGER.info(f"  {m.symbol} ({m.side}): P&L {m.unrealized_pnl_pct:+.2f}% (${m.unrealized_pnl_usd:+.2f}) | "
                                              f"SL: {m.current_sl:.5f}, TP: {m.current_tp:.5f}")
                            else:
                                LOGGER.info("No open positions to monitor")
                        except Exception as e:
                            LOGGER.debug(f"Could not get immediate position metrics: {e}")
                        
                    except Exception as e:
                        LOGGER.error(f"Failed to start position monitor: {e}")

        # 5. Alpaca Execution (if enabled)
        if alpaca_exec_enabled:
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
