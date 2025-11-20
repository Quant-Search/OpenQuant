"""
Market Microstructure Metrics.
Includes VPIN (Volume-Synchronized Probability of Informed Trading), Kyle's Lambda, and Amihud Illiquidity.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

def estimate_buy_sell_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate Buy and Sell volume from OHLCV using Bulk Volume Classification (BVC).
    Assumes normally distributed price changes.
    
    Args:
        df: DataFrame with 'Open', 'Close', 'Volume'.
        
    Returns:
        DataFrame with 'VolumeBuy', 'VolumeSell'.
    """
    # Delta P
    delta_p = df['Close'] - df['Open']
    
    # Sigma of price changes (rolling or static? BVC usually uses standardized change)
    # We'll use the std dev of the close-open changes over the whole window or rolling
    # For simplicity/vectorization, let's use a rolling std dev if possible, or just the series std.
    # BVC paper often uses standardized standardized variable Z = (dP) / sigma
    
    sigma = delta_p.std()
    if sigma == 0:
        sigma = 1e-9
        
    z = delta_p / sigma
    
    # CDF of Z gives probability that trade was buy-initiated
    prob_buy = norm.cdf(z)
    
    vol_buy = df['Volume'] * prob_buy
    vol_sell = df['Volume'] * (1 - prob_buy)
    
    return pd.DataFrame({
        'VolumeBuy': vol_buy,
        'VolumeSell': vol_sell
    }, index=df.index)

def vpin(df: pd.DataFrame, bucket_volume: float = None, window_buckets: int = 50) -> pd.Series:
    """
    Calculate VPIN (Volume-Synchronized Probability of Informed Trading).
    
    Args:
        df: OHLCV DataFrame.
        bucket_volume: Volume per bucket. If None, estimated as avg daily volume / 50.
        window_buckets: Number of buckets for VPIN calculation (n).
        
    Returns:
        Series of VPIN values (indexed by time of bucket completion).
    """
    if df.empty:
        return pd.Series()
        
    # 1. Estimate Buy/Sell Volume
    bs_vol = estimate_buy_sell_volume(df)
    
    # 2. Create Volume Buckets
    # Since we have bars, we need to aggregate them into buckets of size V
    # This is tricky with vectorized operations. We'll iterate or use cumulative sum.
    
    if bucket_volume is None:
        # Heuristic: Average volume per bar * 10? Or Total Volume / N?
        # Let's aim for ~50 buckets in the dataset if small, or fixed size.
        # Better: bucket_volume = Average Daily Volume / 50 (Easley et al)
        # Here we just take mean volume * 10 as a default if not specified
        bucket_volume = df['Volume'].mean() * 10
        
    df_vol = df.copy()
    df_vol['Buy'] = bs_vol['VolumeBuy']
    df_vol['Sell'] = bs_vol['VolumeSell']
    df_vol['CumVol'] = df_vol['Volume'].cumsum()
    
    # Assign bucket IDs
    # Bucket ID = floor(CumVol / bucket_volume)
    df_vol['BucketID'] = (df_vol['CumVol'] // bucket_volume).astype(int)
    
    # Group by BucketID
    # We want Order Imbalance per bucket: |V_buy - V_sell|
    bucket_stats = df_vol.groupby('BucketID').agg({
        'Buy': 'sum',
        'Sell': 'sum',
        'Volume': 'sum', # Should be approx bucket_volume
        'Close': 'last'  # Time/Price of bucket end
    })
    
    # Order Imbalance OI = |V_buy - V_sell|
    bucket_stats['OI'] = (bucket_stats['Buy'] - bucket_stats['Sell']).abs()
    
    # VPIN = SMA(OI, n) / SMA(TotalVolume, n)  (approx, usually sum(OI)/sum(Vol))
    # VPIN = Sum(OI in window) / Sum(Volume in window)
    
    rolling_oi = bucket_stats['OI'].rolling(window=window_buckets).sum()
    rolling_vol = bucket_stats['Volume'].rolling(window=window_buckets).sum()
    
    vpin_series = rolling_oi / rolling_vol
    
    # Reindex back to original timestamp? 
    # VPIN is updated when a bucket completes.
    # We can map bucket VPIN back to the bars that are IN that bucket.
    # Or just return the bucket series. For a strategy, we need it aligned to bars.
    
    # Map VPIN of Bucket N to all bars in Bucket N (or N+1?)
    # Actually, VPIN is known at the END of the bucket.
    # So bars in Bucket N should probably see VPIN of Bucket N-1?
    # Or we just ffill.
    
    # Let's merge back
    bucket_vpin = vpin_series.to_frame(name='VPIN')
    merged = df_vol.merge(bucket_vpin, left_on='BucketID', right_index=True, how='left')
    
    # Fill forward (if a bar is in the middle of a bucket, it knows the previous bucket's VPIN)
    # But wait, if we are in Bucket N, we don't know Bucket N's VPIN yet.
    # We only know Bucket N-1.
    # So we should shift VPIN by 1 bucket.
    
    merged['VPIN'] = merged.groupby('BucketID')['VPIN'].transform('first') # This assigns current bucket VPIN
    # But current bucket VPIN is future knowledge until bucket closes.
    # Correct approach: Shift bucket stats.
    
    vpin_shifted = vpin_series.shift(1) # VPIN available at start of bucket
    bucket_vpin_shifted = vpin_shifted.to_frame(name='VPIN_Prev')
    
    merged = df_vol.merge(bucket_vpin_shifted, left_on='BucketID', right_index=True, how='left')
    
    return merged['VPIN_Prev'].fillna(0.5) # Default to 0.5 (high uncertainty) or 0?

def kyle_lambda(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Kyle's Lambda (Price Impact).
    Lambda = Slope of |Delta P| ~ Volume.
    Simplified: Rolling Cov(|dP|, Vol) / Var(Vol) ? 
    Or just mean(|dP| / Vol) for a simpler illiquidity proxy.
    
    Strictly Kyle's Lambda is from regression: dP = lambda * OrderFlow + noise.
    Since we don't have signed order flow, we use Amihud-like proxy or regression on signed volume.
    
    We'll use a rolling regression of PriceChange vs NetVolume (Buy-Sell).
    """
    bs = estimate_buy_sell_volume(df)
    net_vol = bs['VolumeBuy'] - bs['VolumeSell']
    dp = df['Close'] - df['Open']
    
    # Rolling regression slope
    # Slope = Cov(X, Y) / Var(X)
    # X = NetVol, Y = dP
    
    cov = dp.rolling(window).cov(net_vol)
    var = net_vol.rolling(window).var()
    
    lam = cov / var
    return lam.fillna(0.0)

def amihud_illiquidity(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Amihud Illiquidity Ratio.
    ILLIQ = Mean( |Return| / (Price * Volume) )
    """
    ret = df['Close'].pct_change().abs()
    dollar_vol = df['Close'] * df['Volume']
    
    # Avoid division by zero
    dollar_vol = dollar_vol.replace(0, np.nan)
    
    illiq = (ret / dollar_vol)
    return illiq.rolling(window).mean().fillna(0.0)
