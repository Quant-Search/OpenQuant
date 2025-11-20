"""
Runner for Quantitative Pairs Trading Strategy.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import logging
import pandas as pd
import argparse
from openquant.data.mt5_source import fetch_ohlcv
from openquant.strategies.quant.stat_arb import StatArbStrategy
from openquant.quant.cointegration import engle_granger_test
# from openquant.backtest.engine import BacktestEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
LOGGER = logging.getLogger(__name__)

def run_pairs_trading(symbol_y: str, symbol_x: str, timeframe: str = "1h"):
    LOGGER.info(f"--- Starting Pairs Trading Analysis: {symbol_y} vs {symbol_x} ({timeframe}) ---")
    
    # 1. Fetch Data
    LOGGER.info("Fetching data...")
    try:
        df_y = fetch_ohlcv(symbol_y, timeframe, limit=2000)
        df_x = fetch_ohlcv(symbol_x, timeframe, limit=2000)
    except Exception as e:
        LOGGER.error(f"Data fetch failed: {e}")
        return

    # Align data
    df = pd.concat([df_y['Close'], df_x['Close']], axis=1).dropna()
    df.columns = ['Y', 'X']
    
    series_y = df['Y']
    series_x = df['X']
    
    # 2. Cointegration Test
    LOGGER.info("Running Engle-Granger Cointegration Test...")
    coint_res = engle_granger_test(series_y, series_x)
    
    LOGGER.info(f"Cointegration p-value: {coint_res.get('p_value', 1.0):.4f}")
    LOGGER.info(f"Static Hedge Ratio: {coint_res.get('hedge_ratio', 0.0):.4f}")
    LOGGER.info(f"Half-Life: {coint_res.get('half_life', 0.0):.1f} bars")
    
    if not coint_res.get('is_cointegrated', False):
        LOGGER.warning("⚠️ Pairs are NOT cointegrated. Statistical Arbitrage is risky.")
        # We proceed anyway for demonstration, but in production we would stop.
    else:
        LOGGER.info("✅ Pairs are Cointegrated! Proceeding with Stat Arb.")

    # 3. Run Strategy (Kalman Filter)
    LOGGER.info("Running Kalman Filter Stat Arb Strategy...")
    strategy = StatArbStrategy(pair_symbol=symbol_x, entry_z=2.0, exit_z=0.0)
    
    # We need to pass the full dataframe for Y, and pass X dataframe as 'pair_df'
    # Reconstruct full DFs aligned
    df_y_aligned = df_y.loc[df.index]
    df_x_aligned = df_x.loc[df.index]
    
    signals = strategy.generate_signals(df_y_aligned, pair_df=df_x_aligned)
    
    # 4. Backtest
    # Note: BacktestEngine currently supports single asset. 
    # For pairs trading, PnL = PnL(Y) + PnL(X) * HedgeRatio?
    # Or we treat the "Spread" as the asset?
    # The strategy generates signals for Y (Long/Short). 
    # Implicitly we must trade X in opposite direction.
    
    # Let's simulate PnL manually for the pair
    LOGGER.info("Calculating Pair PnL...")
    
    # Signal 1 (Long Spread) -> Long Y, Short X
    # Signal -1 (Short Spread) -> Short Y, Long X
    
    # Dynamic Beta from Kalman Filter would be better, but let's use the static one for simple PnL approx
    # or retrieve it from strategy if we stored it.
    # The strategy uses dynamic beta internally for Z-score, but we need it for PnL.
    
    # Re-run KF to get betas for PnL calculation
    from openquant.quant.filtering import run_kalman_strategy
    kf_res = run_kalman_strategy(series_y, series_x)
    betas = kf_res['beta']
    
    # Returns
    ret_y = series_y.pct_change().fillna(0)
    ret_x = series_x.pct_change().fillna(0)
    
    # Strategy Returns
    # If Signal=1: Long Y, Short Beta*X
    # Net Return = Ret_Y - Beta * Ret_X
    
    strat_ret = signals.shift(1) * (ret_y - betas * ret_x)
    
    cum_ret = (1 + strat_ret).cumprod()
    total_ret = (cum_ret.iloc[-1] - 1) * 100
    sharpe = strat_ret.mean() / strat_ret.std() * (252**0.5) if strat_ret.std() > 0 else 0
    
    LOGGER.info(f"Total Return: {total_ret:.2f}%")
    LOGGER.info(f"Sharpe Ratio: {sharpe:.2f}")
    
    # Save results
    res_df = pd.DataFrame({
        'Y': series_y,
        'X': series_x,
        'Beta': betas,
        'Signal': signals,
        'Strategy_Ret': strat_ret,
        'Cum_Ret': cum_ret
    })
    output_file = f"reports/statarb_{symbol_y}_{symbol_x}.csv"
    res_df.to_csv(output_file)
    LOGGER.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--y", type=str, required=True, help="Symbol Y (Dependent)")
    parser.add_argument("--x", type=str, required=True, help="Symbol X (Independent)")
    parser.add_argument("--tf", type=str, default="1h", help="Timeframe")
    args = parser.parse_args()
    
    run_pairs_trading(args.y, args.x, args.tf)
