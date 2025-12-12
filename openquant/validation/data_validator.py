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
    # Non-negative values
    vals = df[["Open", "High", "Low", "Close", "Volume"]]
    if (vals < 0).any().any():
        issues.append("negative_values")
    # OHLC relationships
    if ((df["High"] < df[["Open", "Close"]].max(axis=1))).any():
        issues.append("high_below_open_close")
    if ((df["Low"] > df[["Open", "Close"]].min(axis=1))).any():
        issues.append("low_above_open_close")
    # NaNs excessive
    if vals.isna().mean().mean() > 0.1:
        issues.append("too_many_nans")
    # Size
    if len(df) < 10:
        issues.append("too_few_rows")
        
    # Sanity Check: Volume
    if df['Volume'].mean() < 1.0:
        issues.append("low_volume_warning")
    
    # Spike Filter: Detect unrealistic price movements
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
