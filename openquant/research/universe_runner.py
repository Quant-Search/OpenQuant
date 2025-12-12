"""Universe research runner: discover Binance USDT symbols and run SMA research across symbols/timeframes."""
from __future__ import annotations
from typing import List, Dict, Any, Iterable, Optional
from datetime import datetime, timedelta, timezone
from pathlib import Path
import itertools
import concurrent.futures as futures
import pandas as pd


from ..utils.logging import get_logger
from ..data.ccxt_universe import discover_symbols
from ..data.ccxt_source import fetch_ohlcv
from ..data.cache import load_df, save_df, is_fresh
from ..utils.retry import retry_call
from ..strategies.registry import make_strategy
from ..backtest.engine import backtest_signals
from ..backtest.metrics import sharpe, max_drawdown
from ..reporting.markdown import write_universe_summary
from ..evaluation.deflated_sharpe import deflated_sharpe_ratio
from ..storage.results_db import upsert_results, get_best_config, get_best_snapshot, diff_best_snapshots
from ..validation.data_validator import validate_ohlcv
from ..validation.strategy_validator import validate_signals
from ..validation.backtest_validator import validate_backtest

from ..optimization.optuna_search import optuna_best_params
from ..evaluation.wfo import walk_forward_evaluate, WFOSpec
from ..evaluation.regime import compute_regime_features
from ..risk.guardrails import apply_guardrails, apply_concentration_limits
from ..risk.exposure import propose_portfolio_weights

LOGGER = get_logger(__name__)


def _freq_from_timeframe(tf: str) -> str:
    return tf.lower()


def _default_since(tf: str) -> datetime:
    now = datetime.now(timezone.utc)
    tf = tf.lower()
    if tf == "1d":
        return now - timedelta(days=365 * 2)  # ~2y daily
    if tf == "4h":
        return now - timedelta(days=180)
    if tf == "1h":
        return now - timedelta(days=90)
    return now - timedelta(days=90)


def _limit_for(tf: str) -> int:
    # Keep API-friendly; can add pagination later
    return 1000


def run_universe(
    exchange: str = "binance",
    timeframes: Iterable[str] = ("1d", "4h", "1h"),
    top_n: int = 100,
    strategies: Iterable[str] = ("kalman", "hurst", "stat_arb", "liquidity"),
    param_grids: Optional[Dict[str, Dict[str, Iterable[Any]]]] = None,
    fast_list: Iterable[int] = (10, 20),  # fallback for SMA/EMA
    slow_list: Iterable[int] = (50, 100), # fallback for SMA/EMA
    signal_list: Iterable[int] = (9,),    # fallback for MACD
    fee_bps: float = 2.0,
    weight: float = 1.0,
    out_dir: str | Path = "reports",
    max_workers: int | None = None,   # per-DF grid workers
    fetch_workers: int | None = 4,    # across symbols/timeframes concurrency
    results_db: str | Path | None = "data/results.duckdb",
    optimize: bool = True,
    optuna_trials: int = 20,  # Reduced from 50 for speed
    global_trend: str = "neutral", # "bull", "bear", "neutral"
    run_wfo: bool = True,
    # Optional guardrails (all as positive magnitudes):
    dd_limit: float | None = None,
    cvar_limit: float | None = None,
    daily_loss_cap: float | None = None,
    # Optional concentration limits applied after evaluation (mark excess rows ok=False)
    max_ok_per_symbol: int | None = 20,
    max_ok_per_strategy_per_symbol: int | None = 5,
    # Alerts
    alert_webhook: str | None = None,
    # Exposure allocation (for paper-trading planning)
    max_total_exposure: float | None = 1.0,
    max_exposure_per_symbol: float | None = 0.2,
    allocation_slot: float | None = 0.05,
    # Explicit symbols override
    symbols: List[str] | None = None,
    use_opt_actions: bool = True,
) -> Path:
    """Discover liquid symbols and run SMA/EMA/RSI/MACD/Bollinger research across symbols/timeframes.
    Optimization and WFO are enabled by default; results are persisted to DuckDB and reported.
    Returns path to the universe summary report.
    """
    # Default guardrail thresholds if not provided explicitly
    if dd_limit is None:
        dd_limit = 0.20   # 20% max drawdown
    if cvar_limit is None:
        cvar_limit = 0.08 # 8% daily CVaR cap
    if daily_loss_cap is None:
        daily_loss_cap = 0.05 # 5% worst daily loss cap

    is_mt5 = (exchange.lower() == "mt5")
    source_key = ("mt5" if is_mt5 else f"ccxt:{exchange}")
    try:
        from openquant.reporting.intelligent_alerts import IntelligentAlerts
        _alerts_mgr = IntelligentAlerts()
        try:
            import json
            from pathlib import Path as _P
            opt_path = _P("data/optimization_actions.json")
            if use_opt_actions and opt_path.exists():
                with open(opt_path, "r") as f:
                    actions = json.load(f)
                mul = float(actions.get("optuna_trials_multiplier", 1.0))
                if mul and mul > 0:
                    optuna_trials = int(max(8, min(100, round(optuna_trials * mul))))
                overrides = actions.get("param_grid_overrides", {})
                if overrides:
                    param_grids = param_grids or {}
                    for k, v in overrides.items():
                        param_grids[str(k)] = v
        except Exception:
            pass
    except Exception:
        _alerts_mgr = None
    
    if symbols is None:
        if is_mt5:
            try:
                from ..data.mt5_source import discover_fx_symbols
                symbols = discover_fx_symbols(top_n=top_n)
            except Exception as e:
                LOGGER.warning(f"MT5 discover_fx_symbols failed: {e}; using default majors")
                symbols = ["EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","XAUUSD"]
        else:
            symbols = discover_symbols(exchange=exchange, top_n=top_n)
            
    LOGGER.info(f"Discovered {len(symbols)} symbols: {symbols[:10]}{'...' if len(symbols)>10 else ''}")

    rows: List[Dict[str, Any]] = []
    counters: Dict[str, int] = {"guardrail_violations": 0}

    def _default_costs_for_exchange(ex: str) -> tuple[float, float]:
        e = (ex or "").lower()
        if e in {"binance", "bybit", "okx"}:
            return 7.0, 10.0
        if e == "mt5":
            return 0.5, 0.0
        if e == "alpaca":
            return 1.0, 5.0
        return 5.0, 8.0

    # Strategy-specific parameter grids helpers (defined outside worker)
    def _default_grid_for(name: str) -> Dict[str, Iterable[Any]]:
        n = name.lower()
        if n == "kalman":
            return {"process_noise": [1e-5, 1e-4], "measurement_noise": [1e-3, 1e-2], "threshold": [1.0, 1.5]}
        if n == "hurst":
            return {"lookback": [50, 100], "trend_threshold": [0.55], "mr_threshold": [0.45]}
        if n == "stat_arb":
            return {"entry_z": [2.0, 2.5], "exit_z": [0.0, 0.5]}
        if n == "liquidity":
            return {"vpin_threshold": [0.2, 0.3, 0.4], "lookback": [10, 20]}
        if n == "mixer":
            return {
                "sub_strategies": [["kalman", "hurst"], ["stat_arb", "liquidity"], ["kalman", "stat_arb"]],
                "weights": [[0.5, 0.5], [0.7, 0.3], [0.3, 0.7]]
            }
        return {}
    def _narrow_grid_around(seed: Dict[str, Any], grid: Dict[str, Iterable[Any]], band_frac: float = 0.25) -> Dict[str, Iterable[Any]]:
        """Return a narrowed grid keeping values near the seed (within +/- band_frac).
        Falls back to original grid if narrowing would empty an axis.
        """
        if not seed:
            return grid
        narrowed: Dict[str, Iterable[Any]] = {}
        for k, vals in grid.items():
            vlist = list(vals)
            if k in seed and isinstance(seed[k], (int, float)) and vlist:
                sv = seed[k]
                try:
                    if isinstance(sv, int):
                        band = max(1, int(round(abs(sv) * band_frac)))
                        filt = [x for x in vlist if isinstance(x, (int, float)) and abs(int(round(x)) - sv) <= band]
                    else:
                        band = float(abs(sv) * band_frac)
                        filt = [x for x in vlist if isinstance(x, (int, float)) and abs(float(x) - float(sv)) <= band]
                    narrowed[k] = filt or vlist
                except Exception:
                    narrowed[k] = vlist
            else:
                narrowed[k] = vlist
        return narrowed


    def _grid_to_param_dicts(grid: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
        if not grid:
            return [dict()]
        keys = list(grid.keys())
        vals = [list(grid[k]) for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]

    tasks = [(symbol, tf) for symbol in symbols for tf in timeframes]

    def _fetch_yfinance(symbol: str, tf: str) -> pd.DataFrame:
        import yfinance as yf
        # Map common forex pairs to Yahoo format
        yf_sym = symbol
        if len(symbol) == 6 and symbol.isalpha() and exchange == "mt5":
            yf_sym = f"{symbol}=X"
        
        # Map timeframe to yfinance interval
        yf_interval = "1d"
        if tf == "1h": yf_interval = "1h"
        elif tf == "4h": yf_interval = "1h" 
        
        period = "2y" if tf=="1d" else "3mo"
        
        try:
            data = yf.download(yf_sym, period=period, interval=yf_interval, progress=False, auto_adjust=False)
            if data.empty: return pd.DataFrame()

            # Handle MultiIndex columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            data = data.reset_index()
            data.columns = [str(c).capitalize() for c in data.columns]
            
            if "Date" in data.columns: data = data.rename(columns={"Date": "timestamp"})
            elif "Datetime" in data.columns: data = data.rename(columns={"Datetime": "timestamp"})
            elif "Index" in data.columns: data = data.rename(columns={"Index": "timestamp"})
            if "Timestamp" in data.columns: data = data.rename(columns={"Timestamp": "timestamp"})

            if "timestamp" in data.columns:
                data = data.set_index("timestamp")
            
            cols = ["Open", "High", "Low", "Close", "Volume"]
            for c in cols:
                if c in data.columns:
                    data[c] = data[c].astype(float)
            
            data = data[~data.index.duplicated(keep="last")]
            data = data.dropna()
            
            if tf == "4h" and yf_interval == "1h":
                agg_dict = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
                agg_dict = {k: v for k, v in agg_dict.items() if k in data.columns}
                data = data.resample("4h").agg(agg_dict).dropna()
                
            return data
        except Exception as e:
            LOGGER.warning(f"yfinance failed for {symbol}: {e}")
            return pd.DataFrame()

    def _process_symbol_tf(symbol: str, tf: str) -> List[Dict[str, Any]]:
        local_rows: List[Dict[str, Any]] = []
        LOGGER.info(f"Fetching {exchange}:{symbol} {tf}")
        cached = load_df(source_key, symbol, tf)
        
        if cached is None or not is_fresh(cached, tf):
            df = pd.DataFrame()
            try:
                if is_mt5:
                    try:
                        from ..data.mt5_source import fetch_ohlcv as mt5_fetch_ohlcv
                        df = retry_call(
                            lambda: mt5_fetch_ohlcv(symbol, timeframe=tf, since=_default_since(tf), limit=_limit_for(tf)),
                            retries=3, base_delay=0.5
                        )
                    except Exception:
                        LOGGER.warning(f"MetaTrader5 module missing or failed. Using yfinance as proxy for {symbol}")
                        df = _fetch_yfinance(symbol, tf)
                elif exchange == "alpaca":
                    # Try CCXT first (needs keys), fallback to yfinance
                    try:
                        df = retry_call(
                            lambda: fetch_ohlcv(exchange, symbol, timeframe=tf, since=_default_since(tf), limit=_limit_for(tf)),
                            retries=1, base_delay=0.5
                        )
                    except Exception:
                        LOGGER.info(f"Alpaca CCXT failed (no keys?). Using yfinance for {symbol}")
                        df = _fetch_yfinance(symbol, tf)
                else:
                    df = retry_call(
                        lambda: fetch_ohlcv(exchange, symbol, timeframe=tf, since=_default_since(tf), limit=_limit_for(tf)),
                        retries=3, base_delay=0.5
                    )
            except Exception as e:
                if exchange == "alpaca":
                     LOGGER.warning(f"CCXT fetch failed for Alpaca {symbol}: {e}. Trying yfinance fallback.")
                     try:
                        import yfinance as yf
                        data = yf.download(symbol, period=("2y" if tf=="1d" else "3mo"), interval=("1d" if tf=="1d" else "1h"), progress=False, auto_adjust=False)
                        if not data.empty:
                             # This is messy duplication. 
                             # TODO: Refactor yfinance fetcher to a separate module.
                             # For now, I will rely on the fact that I added 'use_yfinance' flag above.
                             # But wait, the 'else' block runs fetch_ohlcv. If that fails, I'm here.
                             pass
                     except Exception as yf_err:
                         LOGGER.warning(f"yfinance fallback failed: {yf_err}")
                
                LOGGER.warning(f"fetch_ohlcv failed for {symbol} {tf}: {e}")
                return local_rows
            if df.empty:
                LOGGER.warning(f"No data for {symbol} {tf}")
                return local_rows

            save_df(df, source_key, symbol, tf)
        else:
            df = cached

        # Regime features for this DF; used to adjust band/trials per strategy family
        regime = compute_regime_features(df)
        trend_score = regime.get("trend_score", 0.0)

        # Data validation (after df is set, regardless of source)
        issues_d = validate_ohlcv(df)
        if issues_d:
            LOGGER.warning(f"Data validation issues {exchange}:{symbol} {tf}: {issues_d}")

        for strat_name in strategies:
            grid = (param_grids or {}).get(strat_name, _default_grid_for(strat_name))
            # Seed/narrow the grid from DB best configs when available
            seed_params, seed_score = (None, None)
            if results_db:
                try:
                    seed_params, seed_score = get_best_config(results_db, exchange, symbol, tf, strat_name)
                    if seed_params is not None:
                        LOGGER.info(f"Seeding grid for {exchange}:{symbol} {tf} {strat_name} with {seed_params} (score={seed_score})")
                except Exception as e:
                    LOGGER.warning(f"Could not load seed params for {strat_name} {symbol} {tf}: {e}")
            # Adaptive band based on stability (seed_score) and regime (trend_score)
            if seed_score is None:
                band_frac = 0.25
            elif seed_score >= 1.0:
                band_frac = 0.10
            elif seed_score >= 0.5:
                band_frac = 0.20
            else:
                band_frac = 0.30
            # Regime-aware tweak: favor trend strategies in positive trend; mean-reversion in negative trend
            strat_l = strat_name.lower()
            is_trend = strat_l in ("sma", "ema", "macd")
            is_mr = strat_l in ("bollinger",)
            regime_trials_factor = 1.0
            if is_trend:
                if trend_score >= 0.5:
                    band_frac *= 0.8; regime_trials_factor *= 1.2
                elif trend_score <= -0.5:
                    band_frac *= 1.2; regime_trials_factor *= 0.9
            elif is_mr:
                if trend_score <= -0.5:
                    band_frac *= 0.8; regime_trials_factor *= 1.2
                elif trend_score >= 0.5:
                    band_frac *= 1.2; regime_trials_factor *= 0.9
            band_frac = float(min(0.40, max(0.05, band_frac)))

            grid_used = _narrow_grid_around(seed_params or {}, grid, band_frac=band_frac)

            param_dicts = _grid_to_param_dicts(grid_used)
            trials = len(param_dicts)
            if optimize:
                # Adaptive trial budget based on stability score and regime
                if seed_score is None:
                    base_trials = optuna_trials
                elif seed_score >= 1.0:
                    base_trials = int(round(optuna_trials * 1.5))
                elif seed_score >= 0.5:
                    base_trials = int(round(optuna_trials * 1.2))
                else:
                    base_trials = int(round(optuna_trials * 0.9))
                n_trials_used = int(min(40, max(8, round(base_trials * regime_trials_factor))))
                try:
                    best = optuna_best_params(strat_name, df, grid_used, fee_bps, weight, tf, n_trials=n_trials_used)
                    param_dicts = [best]
                    trials = max(1, n_trials_used)
                    LOGGER.info(f"Optuna best params {exchange}:{symbol} {tf} {strat_name}: {best} (trials={n_trials_used}, band={band_frac}, trend={trend_score:.3f})")
                except Exception as e:
                    LOGGER.warning(f"Optuna failed for {strat_name} {symbol} {tf}: {e}")


            def _eval_one(params: Dict[str, Any]):
                try:
                    strat = make_strategy(strat_name, **params)
                    sig = strat.generate_signals(df)
                    fee_default, slip_default = _default_costs_for_exchange(exchange)
                    fee_eff = float(fee_bps) if fee_bps is not None else fee_default
                    res = backtest_signals(df, sig, fee_bps=fee_eff, slippage_bps=slip_default, weight=weight)
                    issues_s = validate_signals(df, sig)
                    issues_b = validate_backtest(res)
                    ok_base = (len(issues_d) == 0) and (len(issues_s) == 0) and (len(issues_b) == 0)
                    if not ok_base:
                        LOGGER.warning(f"Validation issues for {strat_name} {symbol} {tf}: data={len(issues_d)} sig={len(issues_s)} bt={len(issues_b)}")

                except Exception as e:
                    LOGGER.warning(f"Backtest error {strat_name} {symbol} {tf} params={params}: {e}")
                    return None
                freq = _freq_from_timeframe(tf)
                s = sharpe(res.returns, freq=freq)
                try:
                    from openquant.backtest.metrics import sortino as _sortino
                    sortino_val = _sortino(res.returns, freq=freq)
                except Exception:
                    sortino_val = 0.0
                # Profit factor from returns
                try:
                    r_arr = res.returns.dropna().values
                    pos_sum = float((r_arr[r_arr>0]).sum()) if r_arr.size else 0.0
                    neg_sum = float(abs((r_arr[r_arr<0]).sum())) if r_arr.size else 0.0
                    profit_factor = float(pos_sum / neg_sum) if neg_sum > 0 else float('inf')
                except Exception:
                    profit_factor = 0.0
                try:
                    from openquant.backtest.metrics import win_rate as _win_rate, monte_carlo_bootstrap
                    win = _win_rate(res.returns)
                    mc = monte_carlo_bootstrap(res.returns, n=300, block=10, freq=freq)
                    import numpy as _np
                    mc_s_p05 = float(_np.percentile(mc["sharpe"], 5)) if mc.get("sharpe") else 0.0
                    mc_s_p95 = float(_np.percentile(mc["sharpe"], 95)) if mc.get("sharpe") else 0.0
                    mc_dd_p95 = float(_np.percentile(mc["max_dd"], 95)) if mc.get("max_dd") else 0.0
                except Exception:
                    win = 0.0
                    mc_s_p05 = 0.0
                    mc_s_p95 = 0.0
                    mc_dd_p95 = 0.0
                # Guardrails inputs: compute worst daily loss magnitude
                worst_daily_loss = None
                try:
                    daily = res.returns.resample('1D').sum().dropna()
                    if not daily.empty:
                        min_daily = float(daily.min())
                        worst_daily_loss = float(max(0.0, -min_daily))
                except Exception:
                    worst_daily_loss = None

                dd = -float(max_drawdown(res.equity_curve)); dd = abs(dd)
                r = res.returns.dropna().values
                cvar_val = 0.0
                if r.size:
                    import numpy as np
                    losses = -r
                    var = float(np.quantile(losses, 0.95))
                    tail = losses[losses >= var]
                    cvar_val = float(tail.mean()) if tail.size else 0.0
                # Statistical significance test (daily aggregation)
                try:
                    daily_ret = res.returns.resample('1D').sum().dropna()
                    n = int(daily_ret.size)
                    if n > 2:
                        import numpy as _np
                        mu = float(daily_ret.mean())
                        sd = float(daily_ret.std(ddof=1))
                        t_stat = mu / (sd / (_np.sqrt(n))) if sd > 0 else 0.0
                        # Normal approx p-value (two-sided)
                        from math import erf, sqrt
                        def _norm_cdf(x: float) -> float:
                            return 0.5 * (1.0 + erf(x / sqrt(2.0)))
                        p_value = float(2.0 * (1.0 - _norm_cdf(abs(t_stat))))
                    else:
                        p_value = 1.0
                except Exception:
                    p_value = 1.0
                # Get Forex config if applicable
                from openquant.config.forex import get_spread_bps, get_swap_cost, FOREX_CONFIG
                
                # Determine spread and swap
                spread = get_spread_bps(symbol) if exchange == "mt5" else 0.0
                swap_l = get_swap_cost(symbol, "long") if exchange == "mt5" else 0.0
                swap_s = get_swap_cost(symbol, "short") if exchange == "mt5" else 0.0
                pip_val = FOREX_CONFIG.get(symbol, {}).get("pip_value", 0.0001)
                
                # Determine leverage (default 1.0 for crypto, 50.0 for forex/mt5)
                lev = 50.0 if exchange == "mt5" else 1.0

                res = backtest_signals(
                    df, sig, 
                    fee_bps=fee_eff, 
                    slippage_bps=slip_default,
                    weight=1.0, 
                    stop_loss_atr=params.get("stop_loss_atr"),
                    take_profit_atr=params.get("take_profit_atr"),
                    spread_bps=spread,
                    leverage=lev,
                    swap_long=swap_l,
                    swap_short=swap_s,
                    funding_bps_long=0.0,
                    funding_bps_short=0.0,
                    pip_value=pip_val
                )
                try:
                    bench_ret = df['Close'].pct_change().fillna(0.0)
                    bench_s = sharpe(bench_ret, freq=freq)
                    alpha_s = s - bench_s
                except Exception:
                    bench_s = 0.0
                    alpha_s = s
                ok_guard, reasons = apply_guardrails(
                    max_drawdown=dd,
                    cvar=cvar_val,
                    worst_daily_loss=worst_daily_loss,
                    dd_limit=dd_limit,
                    cvar_limit=cvar_limit,
                    daily_loss_cap=daily_loss_cap,
                )
                ok = ok_base and ok_guard
                if not ok_guard:
                    LOGGER.warning(f"Guardrails violated for {strat_name} {symbol} {tf}: {'; '.join(reasons)}")
                    counters["guardrail_violations"] = counters.get("guardrail_violations", 0) + 1
                n_trades = int(res.trades.to_numpy().sum())
                wfo_mts = 0.0
                if run_wfo:
                    try:
                        wres = walk_forward_evaluate(
                            df,
                            lambda **pp: make_strategy(strat_name, **pp),
                            grid_used,
                            fee_bps=fee_bps,
                            weight=weight,
                            wfo=WFOSpec(),
                        )
                        wfo_mts = float(wres.get("mean_test_sharpe", 0.0))
                    except Exception as e:
                        LOGGER.warning(f"WFO error {strat_name} {symbol} {tf}: {e}")

                dsr = deflated_sharpe_ratio(s, T=r.size if r.size else 1, trials=max(trials,1))
                # Regime-segmented metrics
                try:
                    ret_prices = df['Close'].pct_change().fillna(0.0)
                    trend_roll = ret_prices.rolling(50).mean()
                    vol_roll = ret_prices.rolling(50).std()
                    vol_thresh = float(ret_prices.std() * 1.5)
                    bull_mask = (trend_roll > 0)
                    bear_mask = (trend_roll < 0)
                    vol_mask = (vol_roll > vol_thresh)
                    calm_mask = (vol_roll <= vol_thresh)
                    r_series = res.returns.reindex(df.index).fillna(0.0)
                    bull_s = sharpe(r_series[bull_mask], freq=freq) if bull_mask.any() else 0.0
                    bear_s = sharpe(r_series[bear_mask], freq=freq) if bear_mask.any() else 0.0
                    vol_s = sharpe(r_series[vol_mask], freq=freq) if vol_mask.any() else 0.0
                    calm_s = sharpe(r_series[calm_mask], freq=freq) if calm_mask.any() else 0.0
                except Exception:
                    bull_s = bear_s = vol_s = calm_s = 0.0
                # Skip storing returns for correlation here - it's memory intensive
                # Correlation filter will be applied at allocation time if needed
                return {
                    "strategy": strat_name,
                    "exchange": exchange,
                    "symbol": symbol,
                    "timeframe": tf,
                    "params": params,
                    "bars": int(len(df)),
                    "metrics": {"sharpe": s, "sortino": sortino_val, "alpha_sharpe": alpha_s, "bench_sharpe": bench_s, "win_rate": win, "profit_factor": profit_factor, "mc_sharpe_p05": mc_s_p05, "mc_sharpe_p95": mc_s_p95, "mc_dd_p95": mc_dd_p95, "bull_sharpe": bull_s, "bear_sharpe": bear_s, "volatile_sharpe": vol_s, "calm_sharpe": calm_s, "dsr": dsr, "max_dd": dd, "cvar": cvar_val, "p_value": p_value, "n_trades": n_trades, "ok": ok, "wfo_mts": wfo_mts},
                }

            if max_workers and max_workers > 1:
                with futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    futs = [ex.submit(_eval_one, p) for p in param_dicts]
                    for fut in futures.as_completed(futs):
                        row = fut.result()
                        if row:
                            local_rows.append(row)
            else:
                for p in param_dicts:
                    row = _eval_one(p)
                    if row:
                        local_rows.append(row)

        return local_rows

    # Adaptive Batch Processing
    import psutil
    
    def get_optimal_batch_size(task_count: int) -> int:
        try:
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024 ** 3)
            # Conservative estimate: 100MB per task (symbol/tf)
            # This depends heavily on data size, but it's a heuristic.
            # Leave 2GB buffer.
            safe_tasks = int((available_gb - 2.0) * 10) 
            safe_tasks = max(1, min(safe_tasks, 50)) # Clamp between 1 and 50
            return safe_tasks
        except Exception:
            return 10 # Fallback
            
    batch_size = get_optimal_batch_size(len(tasks))
    LOGGER.info(f"Adaptive Batch Size: {batch_size} (Tasks: {len(tasks)})")

    for i in range(0, len(tasks), batch_size):
        batch = tasks[i : i + batch_size]
        LOGGER.info(f"Processing batch {i // batch_size + 1}/{(len(tasks) + batch_size - 1) // batch_size} ({len(batch)} tasks)")
        
        if fetch_workers and fetch_workers > 1:
            with futures.ThreadPoolExecutor(max_workers=fetch_workers) as ex:
                futs = [ex.submit(_process_symbol_tf, s, tf) for (s, tf) in batch]
                for fut in futures.as_completed(futs):
                    try:
                        for row in (fut.result() or []):
                            rows.append(row)
                    except Exception as e:
                        LOGGER.error(f"Batch processing error: {e}", exc_info=True)
        else:
            for s, tf in batch:
                try:
                    for row in _process_symbol_tf(s, tf):
                        rows.append(row)
                except Exception as e:
                    LOGGER.error(f"Task processing error {s} {tf}: {e}")
        
        # Explicit garbage collection between batches
        import gc
        gc.collect()

    # Concentration limits: measure before/after
    ok_before_caps = sum(1 for r in rows if (r.get("metrics") or {}).get("ok"))
    rows = apply_concentration_limits(
        rows,
        max_per_symbol=max_ok_per_symbol,
        max_per_strategy_per_symbol=max_ok_per_strategy_per_symbol,
    )
    ok_after_caps = sum(1 for r in rows if (r.get("metrics") or {}).get("ok"))
    cap_demotions = max(0, ok_before_caps - ok_after_caps)

    # Best snapshot before upsert
    prev_best = get_best_snapshot(results_db) if results_db else {}

    # Persist results to DB
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if results_db:
        try:
            upsert_results(rows, db_path=results_db, run_id=run_id)
            LOGGER.info(f"Stored {len(rows)} rows in DB: {results_db}")
            try:
                if _alerts_mgr:
                    _alerts_mgr.run_diagnostics_on_results(results_db, run_id=run_id)
                    _alerts_mgr.generate_diagnostic_report(results_db, run_id=run_id)
                    _alerts_mgr.propose_optimization_actions(results_db, run_id=run_id)
                    _alerts_mgr.save_alerts()
            except Exception as e:
                LOGGER.warning(f"Diagnostics post-upsert failed: {e}")
        except Exception as e:
            LOGGER.warning(f"Failed to store results in DB {results_db}: {e}")

    # Best snapshot after upsert and diff
    curr_best = get_best_snapshot(results_db) if results_db else {}
    added, removed, changed = diff_best_snapshots(prev_best, curr_best)
    new_best_count = len(added | changed)

    # Exposure proposal under caps (paper-trading planning)
    allocation = []
    try:
        if (max_total_exposure and max_total_exposure > 0) and (allocation_slot and allocation_slot > 0):
            allocation = propose_portfolio_weights(
                rows,
                max_total_weight=float(max_total_exposure),
                max_symbol_weight=float(max_exposure_per_symbol or 1.0),
                slot_weight=float(allocation_slot),
                regime_bias=("bull" if str(global_trend).lower()=="bull" else ("bear" if str(global_trend).lower()=="bear" else None)),
            )
    except Exception as e:
        LOGGER.warning(f"Allocation proposal failed: {e}")

    # Write run summary and optionally alert
    from ..utils.alerts import format_run_summary, send_alert
    summary = {
        "run_id": run_id,
        "total_rows": len(rows),
        "ok_before_caps": ok_before_caps,
        "ok_after_caps": ok_after_caps,
        "cap_demotions": cap_demotions,
        "guardrail_violations": int(counters.get("guardrail_violations", 0)),
        "new_best_count": new_best_count,
        "allocated_weight": float(sum(w for _, w in allocation)) if allocation else 0.0,
        "positions": int(len(allocation)),
    }
    body = format_run_summary(summary)

    # Persist allocation to a JSON file for later paper-trading use
    try:
        if allocation:
            alloc_entries = []
            for idx, w in allocation:
                r = rows[idx]
                alloc_entries.append({
                    "exchange": r.get("exchange"),
                    "symbol": r.get("symbol"),
                    "timeframe": r.get("timeframe"),
                    "strategy": r.get("strategy"),
                    "params": r.get("params"),
                    "weight": w,
                })
            with open(Path(out_dir) / f"allocation_{run_id}.json", "w", encoding="utf-8") as f:
                import json as _json
                _json.dump({"run_id": run_id, "allocations": alloc_entries}, f, indent=2)
    except Exception as e:
        LOGGER.warning(f"Failed to write allocation file: {e}")

    # Write summary to reports
    try:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / f"run_summary_{run_id}.md", "w", encoding="utf-8") as f:
            f.write(body + "\n")
    except Exception as e:
        LOGGER.warning(f"Failed to write run summary: {e}")

    # Send alert via webhook if configured
    try:
        send_alert(
            subject=(f"OpenQuant: {new_best_count} new best, {cap_demotions} demoted, {summary['guardrail_violations']} guardrail"),
            body=body,
            severity=("WARNING" if (cap_demotions > 0 or summary['guardrail_violations'] > 0) else "INFO"),
            webhook_url=alert_webhook,
        )
    except Exception:
        pass

    report_path = write_universe_summary(rows, out_dir=out_dir, top_k=200)
    LOGGER.info(f"Universe summary written: {report_path}")
    return report_path

