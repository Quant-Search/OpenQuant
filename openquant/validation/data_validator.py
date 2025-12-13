"""OHLCV data validation utilities.

Checks implemented:
- Non-negative prices and volume
- High >= max(Open, Close), Low <= min(Open, Close)
- Monotonic, unique datetime index
- Sufficient rows (>= 10) for basic indicators
- Statistical outlier detection (Grubbs test, IQR method)
- Gap detection in time series
- Volume profile validation
- Data quality scoring metric
"""
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats


def grubbs_test(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float, int]:
    """
    Grubbs test for detecting a single outlier in univariate data.
    
    Null Hypothesis (H0): No outliers in the data.
    Alternative Hypothesis: Exactly one outlier exists.
    
    Args:
        data: 1D array of numeric values (should be approximately normal).
        alpha: Significance level (default 0.05).
    
    Returns:
        (has_outlier, test_statistic, outlier_index)
        has_outlier: True if an outlier is detected at the given alpha level.
        test_statistic: Computed Grubbs test statistic.
        outlier_index: Index of the suspected outlier.
    """
    if len(data) < 3:
        return False, 0.0, -1
    
    # Remove NaNs
    clean_data = data[~np.isnan(data)]
    if len(clean_data) < 3:
        return False, 0.0, -1
    
    n = len(clean_data)
    mean = np.mean(clean_data)
    std = np.std(clean_data, ddof=1)
    
    if std == 0:
        return False, 0.0, -1
    
    # Find max absolute deviation from mean
    abs_dev = np.abs(clean_data - mean)
    max_idx = np.argmax(abs_dev)
    G = abs_dev[max_idx] / std
    
    # Critical value from t-distribution
    t_dist = stats.t.ppf(1 - alpha / (2 * n), n - 2)
    G_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_dist**2 / (n - 2 + t_dist**2))
    
    has_outlier = G > G_critical
    return has_outlier, G, max_idx


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        data: Pandas Series of numeric values.
        multiplier: IQR multiplier (default 1.5 for standard outliers, 3.0 for extreme).
    
    Returns:
        Boolean Series indicating outlier positions (True = outlier).
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers


def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using Z-score method.
    
    Args:
        data: Pandas Series of numeric values.
        threshold: Z-score threshold (default 3.0).
    
    Returns:
        Boolean Series indicating outlier positions (True = outlier).
    """
    mean = data.mean()
    std = data.std()
    
    if std == 0:
        return pd.Series(False, index=data.index)
    
    z_scores = np.abs((data - mean) / std)
    outliers = z_scores > threshold
    return outliers


def detect_gaps(df: pd.DataFrame, max_gap_minutes: int = 60) -> List[Dict[str, Any]]:
    """
    Detect gaps in time series data.
    
    Args:
        df: DataFrame with DatetimeIndex.
        max_gap_minutes: Maximum expected gap in minutes (default 60).
    
    Returns:
        List of dictionaries with gap information:
        [{"start": datetime, "end": datetime, "duration_minutes": float}]
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return []
    
    if len(df) < 2:
        return []
    
    gaps = []
    time_diffs = df.index.to_series().diff()
    
    # Convert to minutes
    time_diffs_minutes = time_diffs.dt.total_seconds() / 60.0
    
    # Find gaps exceeding threshold
    gap_mask = time_diffs_minutes > max_gap_minutes
    gap_indices = np.where(gap_mask)[0]
    
    for idx in gap_indices:
        gap_info = {
            "start": df.index[idx - 1],
            "end": df.index[idx],
            "duration_minutes": time_diffs_minutes.iloc[idx]
        }
        gaps.append(gap_info)
    
    return gaps


def validate_volume_profile(df: pd.DataFrame, 
                            min_avg_volume: float = 1.0,
                            low_volume_threshold: float = 0.1) -> Dict[str, Any]:
    """
    Validate volume profile and detect anomalies.
    
    Args:
        df: DataFrame with 'Volume' column.
        min_avg_volume: Minimum acceptable average volume.
        low_volume_threshold: Fraction of average volume to flag low volume bars.
    
    Returns:
        Dictionary with volume validation results:
        {
            "avg_volume": float,
            "median_volume": float,
            "zero_volume_count": int,
            "low_volume_count": int,
            "volume_outliers_count": int,
            "volume_spikes": List[int],  # indices
            "is_valid": bool
        }
    """
    if "Volume" not in df.columns:
        return {"is_valid": False, "error": "Volume column missing"}
    
    volume = df["Volume"]
    
    avg_volume = volume.mean()
    median_volume = volume.median()
    
    # Count zero volume bars
    zero_volume_count = (volume == 0).sum()
    
    # Count low volume bars (below threshold * average)
    low_volume_count = (volume < low_volume_threshold * avg_volume).sum()
    
    # Detect volume outliers using IQR
    volume_outliers = detect_outliers_iqr(volume, multiplier=2.5)
    volume_outliers_count = volume_outliers.sum()
    
    # Detect volume spikes (outliers on high side only)
    Q3 = volume.quantile(0.75)
    IQR = volume.quantile(0.75) - volume.quantile(0.25)
    upper_bound = Q3 + 2.5 * IQR
    volume_spikes = volume[volume > upper_bound].index.tolist()
    
    is_valid = (
        avg_volume >= min_avg_volume and
        zero_volume_count / len(df) < 0.05  # Less than 5% zero volume
    )
    
    return {
        "avg_volume": avg_volume,
        "median_volume": median_volume,
        "zero_volume_count": int(zero_volume_count),
        "low_volume_count": int(low_volume_count),
        "volume_outliers_count": int(volume_outliers_count),
        "volume_spikes": volume_spikes,
        "is_valid": is_valid
    }


def calculate_data_quality_score(df: pd.DataFrame, 
                                 outlier_weight: float = 0.3,
                                 completeness_weight: float = 0.3,
                                 consistency_weight: float = 0.2,
                                 volume_weight: float = 0.2) -> Dict[str, Any]:
    """
    Calculate overall data quality score (0-100).
    
    Args:
        df: OHLCV DataFrame.
        outlier_weight: Weight for outlier detection component.
        completeness_weight: Weight for data completeness component.
        consistency_weight: Weight for data consistency component.
        volume_weight: Weight for volume profile component.
    
    Returns:
        Dictionary with quality score and sub-scores:
        {
            "overall_score": float (0-100),
            "outlier_score": float (0-100),
            "completeness_score": float (0-100),
            "consistency_score": float (0-100),
            "volume_score": float (0-100),
            "grade": str (A, B, C, D, F)
        }
    """
    if len(df) < 10:
        return {
            "overall_score": 0.0,
            "outlier_score": 0.0,
            "completeness_score": 0.0,
            "consistency_score": 0.0,
            "volume_score": 0.0,
            "grade": "F",
            "error": "Insufficient data"
        }
    
    # 1. Outlier Score (100 = no outliers, 0 = many outliers)
    price_cols = ["Open", "High", "Low", "Close"]
    total_outliers = 0
    total_points = 0
    
    for col in price_cols:
        if col in df.columns:
            outliers = detect_outliers_iqr(df[col], multiplier=1.5)
            total_outliers += outliers.sum()
            total_points += len(df)
    
    outlier_ratio = total_outliers / total_points if total_points > 0 else 0
    outlier_score = max(0, 100 * (1 - outlier_ratio * 10))  # Penalize heavily
    
    # 2. Completeness Score (100 = no missing data, 0 = all missing)
    missing_ratio = df[price_cols + ["Volume"]].isna().mean().mean()
    completeness_score = max(0, 100 * (1 - missing_ratio * 5))  # Penalize missing data
    
    # 3. Consistency Score (OHLC relationships, monotonic index)
    consistency_issues = 0
    
    # Check OHLC relationships
    if ((df["High"] < df[["Open", "Close"]].max(axis=1))).sum() > 0:
        consistency_issues += 1
    if ((df["Low"] > df[["Open", "Close"]].min(axis=1))).sum() > 0:
        consistency_issues += 1
    
    # Check negative values
    if (df[price_cols + ["Volume"]] < 0).any().any():
        consistency_issues += 1
    
    # Check index
    if not isinstance(df.index, pd.DatetimeIndex):
        consistency_issues += 1
    elif df.index.has_duplicates:
        consistency_issues += 1
    elif not df.index.is_monotonic_increasing:
        consistency_issues += 1
    
    # Check for extreme price spikes (> 50% single-bar move)
    pct_change = df['Close'].pct_change().abs()
    if (pct_change > 0.5).any():
        consistency_issues += 1
    
    max_consistency_issues = 7
    consistency_score = max(0, 100 * (1 - consistency_issues / max_consistency_issues))
    
    # 4. Volume Score
    volume_profile = validate_volume_profile(df)
    if "error" in volume_profile:
        volume_score = 0.0
    else:
        # Score based on zero volume ratio and low volume ratio
        zero_volume_ratio = volume_profile["zero_volume_count"] / len(df)
        low_volume_ratio = volume_profile["low_volume_count"] / len(df)
        
        volume_score = max(0, 100 * (1 - zero_volume_ratio * 20 - low_volume_ratio * 2))
    
    # Calculate weighted overall score
    overall_score = (
        outlier_weight * outlier_score +
        completeness_weight * completeness_score +
        consistency_weight * consistency_score +
        volume_weight * volume_score
    )
    
    # Assign grade
    if overall_score >= 90:
        grade = "A"
    elif overall_score >= 80:
        grade = "B"
    elif overall_score >= 70:
        grade = "C"
    elif overall_score >= 60:
        grade = "D"
    else:
        grade = "F"
    
    return {
        "overall_score": round(overall_score, 2),
        "outlier_score": round(outlier_score, 2),
        "completeness_score": round(completeness_score, 2),
        "consistency_score": round(consistency_score, 2),
        "volume_score": round(volume_score, 2),
        "grade": grade
    }


def validate_ohlcv(df: pd.DataFrame, 
                   detect_outliers: bool = True,
                   detect_time_gaps: bool = True,
                   validate_volume: bool = True) -> List[str]:
    """
    Comprehensive OHLCV validation with enhanced checks.
    
    Args:
        df: OHLCV DataFrame.
        detect_outliers: Enable statistical outlier detection.
        detect_time_gaps: Enable gap detection in time series.
        validate_volume: Enable volume profile validation.
    
    Returns:
        List of issue identifiers (empty list = no issues).
    """
    issues: List[str] = []
    req = {"Open", "High", "Low", "Close", "Volume"}
    if not req.issubset(df.columns):
        issues.append("missing_columns")
        return issues
    if not isinstance(df.index, pd.DatetimeIndex):
        issues.append("index_not_datetime")
    if df.index.has_duplicates:
        issues.append("duplicate_index")
    if not df.index.is_monotonic_increasing:
        issues.append("non_monotonic_index")
    vals = df[["Open", "High", "Low", "Close", "Volume"]]
    if (vals < 0).any().any():
        issues.append("negative_values")
    if ((df["High"] < df[["Open", "Close"]].max(axis=1))).any():
        issues.append("high_below_open_close")
    if ((df["Low"] > df[["Open", "Close"]].min(axis=1))).any():
        issues.append("low_above_open_close")
    if vals.isna().mean().mean() > 0.1:
        issues.append("too_many_nans")
    if len(df) < 10:
        issues.append("too_few_rows")
    if df['Volume'].mean() < 1.0:
        issues.append("low_volume_warning")
    pct_change = df['Close'].pct_change().abs()
    if (pct_change > 0.5).any():
        issues.append("price_spike_detected")
    
    # Enhanced validation checks
    if detect_outliers and len(df) >= 10:
        # Check for outliers in price data using IQR method
        price_cols = ["Open", "High", "Low", "Close"]
        total_outliers = 0
        for col in price_cols:
            outliers = detect_outliers_iqr(df[col], multiplier=1.5)
            total_outliers += outliers.sum()
        
        # Flag if more than 5% of data points are outliers
        if total_outliers / (len(df) * len(price_cols)) > 0.05:
            issues.append("excessive_outliers")
        
        # Grubbs test on returns for severe outliers
        returns = df['Close'].pct_change().dropna()
        if len(returns) > 10:
            has_outlier, _, _ = grubbs_test(returns.values, alpha=0.01)
            if has_outlier:
                issues.append("grubbs_outlier_detected")
    
    if detect_time_gaps and isinstance(df.index, pd.DatetimeIndex) and len(df) >= 2:
        # Infer expected frequency
        time_diffs = df.index.to_series().diff().dropna()
        median_diff_minutes = time_diffs.median().total_seconds() / 60.0
        
        # Set gap threshold as 3x median interval
        gap_threshold = max(median_diff_minutes * 3, 60)
        
        gaps = detect_gaps(df, max_gap_minutes=gap_threshold)
        if len(gaps) > 0:
            issues.append("time_gaps_detected")
        
        # Flag if too many gaps (> 5% of intervals)
        if len(gaps) / len(df) > 0.05:
            issues.append("excessive_gaps")
    
    if validate_volume:
        volume_profile = validate_volume_profile(df)
        if not volume_profile.get("is_valid", False):
            issues.append("volume_profile_invalid")
        
        # Check for zero volume bars
        if volume_profile.get("zero_volume_count", 0) > 0:
            issues.append("zero_volume_bars")
        
        # Check for excessive volume spikes
        if volume_profile.get("volume_outliers_count", 0) / len(df) > 0.10:
            issues.append("excessive_volume_spikes")
        
    return issues


def is_valid_ohlcv(df: pd.DataFrame) -> bool:
    """Check if OHLCV data passes basic validation."""
    return len(validate_ohlcv(df)) == 0


def get_validation_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive validation report with quality metrics.
    
    Args:
        df: OHLCV DataFrame.
    
    Returns:
        Dictionary with validation results, quality score, and detailed metrics.
    """
    issues = validate_ohlcv(df, detect_outliers=True, detect_time_gaps=True, validate_volume=True)
    quality = calculate_data_quality_score(df)
    
    # Detect gaps if datetime index
    gaps = []
    if isinstance(df.index, pd.DatetimeIndex) and len(df) >= 2:
        time_diffs = df.index.to_series().diff().dropna()
        median_diff_minutes = time_diffs.median().total_seconds() / 60.0
        gap_threshold = max(median_diff_minutes * 3, 60)
        gaps = detect_gaps(df, max_gap_minutes=gap_threshold)
    
    # Volume profile
    volume_profile = validate_volume_profile(df)
    
    # Outlier details
    outlier_details = {}
    if len(df) >= 10:
        for col in ["Open", "High", "Low", "Close"]:
            if col in df.columns:
                outliers_iqr = detect_outliers_iqr(df[col], multiplier=1.5)
                outlier_details[col] = {
                    "count": int(outliers_iqr.sum()),
                    "percentage": round(100 * outliers_iqr.sum() / len(df), 2)
                }
    
    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "quality_score": quality,
        "row_count": len(df),
        "date_range": {
            "start": str(df.index[0]) if len(df) > 0 else None,
            "end": str(df.index[-1]) if len(df) > 0 else None
        },
        "gaps": {
            "count": len(gaps),
            "details": gaps[:10]  # Limit to first 10 gaps
        },
        "volume_profile": volume_profile,
        "outliers": outlier_details
    }


def detect_gaps(df: pd.DataFrame, threshold_multiplier: float = 3.0) -> np.ndarray:
    """Detect gaps in price data (>3x median bar-to-bar change).
    
    Args:
        df: DataFrame with OHLCV data
        threshold_multiplier: Multiplier for median change (default 3.0)
        
    Returns:
        Boolean array indicating gaps
    """
    if len(df) < 2:
        return np.zeros(len(df), dtype=bool)
    
    close_changes = df['Close'].diff().abs()
    median_change = close_changes.median()
    
    if median_change == 0 or np.isnan(median_change):
        pct_changes = df['Close'].pct_change().abs()
        median_pct_change = pct_changes.median()
        if median_pct_change == 0 or np.isnan(median_pct_change):
            return np.zeros(len(df), dtype=bool)
        threshold = median_pct_change * threshold_multiplier
        gaps = pct_changes > threshold
    else:
        threshold = median_change * threshold_multiplier
        gaps = close_changes > threshold
    
    return gaps.fillna(False).values


def detect_volume_anomalies(df: pd.DataFrame, window: int = 20, sigma_threshold: float = 5.0) -> np.ndarray:
    """Detect volume anomalies (>5σ from rolling mean).
    
    Args:
        df: DataFrame with OHLCV data
        window: Rolling window size (default 20)
        sigma_threshold: Sigma multiplier (default 5.0)
        
    Returns:
        Boolean array indicating volume anomalies
    """
    if len(df) < window:
        return np.zeros(len(df), dtype=bool)
    
    volume = df['Volume'].values
    rolling_mean = pd.Series(volume).rolling(window=window, min_periods=1).mean()
    rolling_std = pd.Series(volume).rolling(window=window, min_periods=1).std()
    
    rolling_std = rolling_std.fillna(0)
    anomalies = np.abs(volume - rolling_mean.values) > (sigma_threshold * rolling_std.values)
    
    return anomalies


def clean_ohlcv(df: pd.DataFrame, 
                fill_method: str = "ffill",
                outlier_method: str = "iqr",
                iqr_multiplier: float = 2.5,
                grubbs_alpha: float = 0.05,
                cap_outliers: bool = False,
                handle_zero_volume: bool = True,
                zero_volume_fill: float = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean OHLCV data by filling time-series gaps, removing/capping outliers, and handling zero-volume bars.
    
    This function performs comprehensive data cleaning:
    1. Fills time-series gaps using forward-fill or interpolation
    2. Detects and removes/caps outliers using Grubbs test and/or IQR methods
    3. Handles zero-volume bars
    4. Returns cleaned DataFrame with metadata about applied corrections
    
    Args:
        df: DataFrame with OHLCV data and DatetimeIndex.
        fill_method: Method to fill gaps ('ffill', 'bfill', 'interpolate').
        outlier_method: Outlier detection method ('iqr', 'grubbs', 'both').
        iqr_multiplier: IQR multiplier for outlier detection (default 2.5).
        grubbs_alpha: Significance level for Grubbs test (default 0.05).
        cap_outliers: If True, cap outliers instead of removing them (default False).
        handle_zero_volume: Whether to handle zero-volume bars (default True).
        zero_volume_fill: Value to fill zero-volume bars. If None, use median volume.
        
    Returns:
        Tuple of (cleaned_df, metadata_dict):
        - cleaned_df: Cleaned DataFrame
        - metadata_dict: Dictionary with information about applied corrections:
            {
                "original_rows": int,
                "cleaned_rows": int,
                "gaps_filled": int,
                "outliers_detected": Dict[str, int],  # per column
                "outliers_removed": Dict[str, int],   # per column
                "outliers_capped": Dict[str, int],    # per column
                "zero_volume_bars": int,
                "zero_volume_filled": int,
                "time_gaps_filled": List[Dict],
                "fill_method_used": str,
                "outlier_method_used": str
            }
    """
    if len(df) == 0:
        metadata = {
            "original_rows": 0,
            "cleaned_rows": 0,
            "gaps_filled": 0,
            "outliers_detected": {},
            "outliers_removed": {},
            "outliers_capped": {},
            "zero_volume_bars": 0,
            "zero_volume_filled": 0,
            "time_gaps_filled": [],
            "fill_method_used": fill_method,
            "outlier_method_used": outlier_method
        }
        return df.copy(), metadata
    
    # Initialize metadata
    metadata = {
        "original_rows": len(df),
        "cleaned_rows": 0,
        "gaps_filled": 0,
        "outliers_detected": {},
        "outliers_removed": {},
        "outliers_capped": {},
        "zero_volume_bars": 0,
        "zero_volume_filled": 0,
        "time_gaps_filled": [],
        "fill_method_used": fill_method,
        "outlier_method_used": outlier_method
    }
    
    # Validate required columns
    req = {"Open", "High", "Low", "Close", "Volume"}
    if not req.issubset(df.columns):
        metadata["cleaned_rows"] = len(df)
        return df.copy(), metadata
    
    cleaned = df.copy()
    price_cols = ["Open", "High", "Low", "Close"]
    
    # Step 1: Detect and fill time-series gaps
    if isinstance(cleaned.index, pd.DatetimeIndex) and len(cleaned) >= 2:
        # Detect gaps before filling
        time_diffs = cleaned.index.to_series().diff().dropna()
        median_diff_minutes = time_diffs.median().total_seconds() / 60.0
        gap_threshold = max(median_diff_minutes * 3, 60)
        gaps_before = detect_gaps(cleaned, max_gap_minutes=gap_threshold)
        metadata["time_gaps_filled"] = gaps_before
    
    # Step 2: Detect and handle outliers
    if len(cleaned) >= 4:
        for col in price_cols + ["Volume"]:
            valid_data = cleaned[col].dropna()
            if len(valid_data) < 4:
                continue
            
            # Initialize outlier mask
            outlier_mask = pd.Series(False, index=cleaned.index)
            
            # IQR method
            if outlier_method in ["iqr", "both"]:
                outliers_iqr = detect_outliers_iqr(cleaned[col], multiplier=iqr_multiplier)
                outlier_mask = outlier_mask | outliers_iqr
            
            # Grubbs test (iterative)
            if outlier_method in ["grubbs", "both"] and len(valid_data) >= 3:
                temp_data = cleaned[col].dropna().copy()
                grubbs_outlier_indices = []
                
                # Iterative Grubbs test (detect multiple outliers)
                max_iterations = min(10, len(temp_data) // 10)  # At most 10% of data
                for _ in range(max_iterations):
                    if len(temp_data) < 3:
                        break
                    
                    has_outlier, _, outlier_idx = grubbs_test(temp_data.values, alpha=grubbs_alpha)
                    if not has_outlier:
                        break
                    
                    # Get actual index in original dataframe
                    actual_idx = temp_data.index[outlier_idx]
                    grubbs_outlier_indices.append(actual_idx)
                    
                    # Remove outlier and continue
                    temp_data = temp_data.drop(actual_idx)
                
                # Mark Grubbs outliers in the mask
                if grubbs_outlier_indices:
                    outlier_mask.loc[grubbs_outlier_indices] = True
            
            # Record outlier counts
            outlier_count = outlier_mask.sum()
            metadata["outliers_detected"][col] = int(outlier_count)
            
            if outlier_count > 0:
                if cap_outliers:
                    # Cap outliers using IQR bounds
                    Q1 = cleaned[col].quantile(0.25)
                    Q3 = cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - iqr_multiplier * IQR
                    upper_bound = Q3 + iqr_multiplier * IQR
                    
                    # Cap values
                    cleaned.loc[outlier_mask, col] = cleaned.loc[outlier_mask, col].clip(lower_bound, upper_bound)
                    metadata["outliers_capped"][col] = int(outlier_count)
                    metadata["outliers_removed"][col] = 0
                else:
                    # Remove outliers by setting to NaN
                    cleaned.loc[outlier_mask, col] = np.nan
                    metadata["outliers_removed"][col] = int(outlier_count)
                    metadata["outliers_capped"][col] = 0
            else:
                metadata["outliers_removed"][col] = 0
                metadata["outliers_capped"][col] = 0
    
    # Step 3: Handle zero-volume bars
    if handle_zero_volume and "Volume" in cleaned.columns:
        zero_volume_mask = cleaned["Volume"] == 0
        zero_volume_count = zero_volume_mask.sum()
        metadata["zero_volume_bars"] = int(zero_volume_count)
        
        if zero_volume_count > 0:
            if zero_volume_fill is None:
                # Use median of non-zero volumes
                non_zero_volumes = cleaned.loc[~zero_volume_mask, "Volume"]
                if len(non_zero_volumes) > 0:
                    zero_volume_fill = non_zero_volumes.median()
                else:
                    zero_volume_fill = 1.0  # Default fallback
            
            cleaned.loc[zero_volume_mask, "Volume"] = zero_volume_fill
            metadata["zero_volume_filled"] = int(zero_volume_count)
        else:
            metadata["zero_volume_filled"] = 0
    
    # Step 4: Fill remaining gaps/NaNs
    gaps_before_fill = cleaned[price_cols + ["Volume"]].isna().sum().sum()
    
    if fill_method == "ffill":
        cleaned[price_cols] = cleaned[price_cols].fillna(method='ffill')
        cleaned[price_cols] = cleaned[price_cols].fillna(method='bfill')  # Backfill any leading NaNs
        cleaned["Volume"] = cleaned["Volume"].fillna(method='ffill')
        cleaned["Volume"] = cleaned["Volume"].fillna(0)  # Fill any remaining with 0
    elif fill_method == "bfill":
        cleaned[price_cols] = cleaned[price_cols].fillna(method='bfill')
        cleaned[price_cols] = cleaned[price_cols].fillna(method='ffill')  # Forward-fill any trailing NaNs
        cleaned["Volume"] = cleaned["Volume"].fillna(method='bfill')
        cleaned["Volume"] = cleaned["Volume"].fillna(0)  # Fill any remaining with 0
    elif fill_method == "interpolate":
        cleaned[price_cols] = cleaned[price_cols].interpolate(method='linear', limit_direction='both')
        cleaned["Volume"] = cleaned["Volume"].interpolate(method='linear', limit_direction='both')
        cleaned["Volume"] = cleaned["Volume"].fillna(0)  # Fill any remaining with 0
    else:
        # Default to forward fill
        cleaned[price_cols] = cleaned[price_cols].fillna(method='ffill')
        cleaned[price_cols] = cleaned[price_cols].fillna(method='bfill')
        cleaned["Volume"] = cleaned["Volume"].fillna(0)
    
    gaps_after_fill = cleaned[price_cols + ["Volume"]].isna().sum().sum()
    metadata["gaps_filled"] = int(gaps_before_fill - gaps_after_fill)
    
    # Step 5: Drop rows that still have NaN in critical columns (Close)
    rows_before_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=["Close"])
    rows_dropped = rows_before_drop - len(cleaned)
    
    metadata["cleaned_rows"] = len(cleaned)
    
    return cleaned, metadata
