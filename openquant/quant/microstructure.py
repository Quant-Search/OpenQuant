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
    delta_p = df['Close'] - df['Open']

    sigma = delta_p.std()
    if sigma == 0:
        sigma = 1e-9

    z = delta_p / sigma

    prob_buy = norm.cdf(z)

    vol_buy = df['Volume'] * prob_buy
    vol_sell = df['Volume'] * (1 - prob_buy)

    return pd.DataFrame({
        'VolumeBuy': vol_buy,
        'VolumeSell': vol_sell
    }, index=df.index)

def vpin(df: pd.DataFrame, bucket_volume: float | None = None, window_buckets: int = 50) -> pd.Series:
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
        return pd.Series(dtype=float)

    bs_vol = estimate_buy_sell_volume(df)

    if bucket_volume is None:
        bucket_volume = df['Volume'].mean() * 10

    df_vol = df.copy()
    df_vol['Buy'] = bs_vol['VolumeBuy']
    df_vol['Sell'] = bs_vol['VolumeSell']
    df_vol['CumVol'] = df_vol['Volume'].cumsum()

    df_vol['BucketID'] = (df_vol['CumVol'] // bucket_volume).astype(int)

    bucket_stats = df_vol.groupby('BucketID').agg({
        'Buy': 'sum',
        'Sell': 'sum',
        'Volume': 'sum',
        'Close': 'last'
    })

    bucket_stats['OI'] = (bucket_stats['Buy'] - bucket_stats['Sell']).abs()

    rolling_oi = bucket_stats['OI'].rolling(window=window_buckets).sum()
    rolling_vol = bucket_stats['Volume'].rolling(window=window_buckets).sum()

    vpin_series = rolling_oi / rolling_vol

    vpin_shifted = vpin_series.shift(1)
    bucket_vpin_shifted = vpin_shifted.to_frame(name='VPIN_Prev')

    merged = df_vol.merge(bucket_vpin_shifted, left_on='BucketID', right_index=True, how='left')

    return merged['VPIN_Prev'].fillna(0.5)

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

    dollar_vol = dollar_vol.replace(0, np.nan)

    illiq = (ret / dollar_vol)
    return illiq.rolling(window).mean().fillna(0.0)
