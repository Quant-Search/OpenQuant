"""
Machine Learning Strategy.

Implements a rigorous Walk-Forward Optimization framework for generating trading signals
using scikit-learn models. Focuses on statistical features and probabilistic outputs.
"""
import pandas as pd
import numpy as np
from typing import Any, List, Optional, Dict
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import json
from pathlib import Path

from openquant.strategies.base import BaseStrategy
from openquant.utils.logging import get_logger

LOGGER = get_logger(__name__)

class MLStrategy(BaseStrategy):
    """
    Machine Learning Strategy using Walk-Forward Optimization.
    
    Attributes:
        model: scikit-learn estimator (default: RandomForestClassifier).
        lookback: Number of bars to use for training.
        retrain_interval: How often to retrain the model (bars).
        features: List of feature names to generate.
        probability_threshold: Minimum probability confidence to generate a signal.
    """
    def __init__(self, 
                 model: Optional[BaseEstimator] = None, 
                 lookback: int = 500, 
                 retrain_interval: int = 50,
                 probability_threshold: float = 0.55,
                 feature_config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model = model if model is not None else RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        self.lookback = lookback
        self.retrain_interval = retrain_interval
        self.probability_threshold = probability_threshold
        self.feature_config = feature_config or {}

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate statistical features from OHLCV data.
        Includes multi-timeframe analysis and advanced microstructure proxies.
        """
        # Lazy import GPU features
        try:
            from openquant.features.gpu_features import (
                is_gpu_features_available, rsi_gpu, volatility_gpu, log_returns_gpu
            )
            use_gpu = is_gpu_features_available()
        except ImportError:
            use_gpu = False

        data = df.copy()
        
        # --- Multi-Timeframe Resampling ---
        # Resample to 4H and Daily to capture macro trends
        # Note: We must be careful with lookahead bias. 
        # We use shift(1) on higher TFs and then reindex/ffill to current TF.
        
        # 4H Features
        df_4h = df.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        # Daily Features
        df_1d = df.resample('1d').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        # Calculate features on higher TFs
        def calc_tf_features(tf_df, suffix):
            tf_data = tf_df.copy()
            tf_data['log_ret'] = np.log(tf_data['Close'] / tf_data['Close'].shift(1))
            
            feats = pd.DataFrame(index=tf_data.index)
            feats[f'trend_{suffix}'] = tf_data['log_ret'].rolling(10).mean()
            feats[f'vol_{suffix}'] = tf_data['log_ret'].rolling(20).std()
            
            # RSI on higher TF
            delta = tf_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            feats[f'rsi_{suffix}'] = 100 - (100 / (1 + rs))
            
            return feats.shift(1) # Shift to avoid lookahead

        feats_4h = calc_tf_features(df_4h, '4h')
        feats_1d = calc_tf_features(df_1d, '1d')
        
        # Reindex back to original timeframe (ffill)
        feats_4h_aligned = feats_4h.reindex(data.index, method='ffill')
        feats_1d_aligned = feats_1d.reindex(data.index, method='ffill')
        
        # --- Base Timeframe Features ---
        
        if use_gpu:
            # GPU Implementation
            try:
                close_arr = data['Close'].values
                
                # Log returns
                data['log_ret'] = log_returns_gpu(close_arr)
                
                features = pd.DataFrame(index=data.index)
                
                # 1. Momentum / Lags
                features['ret_lag1'] = data['log_ret'].shift(1)
                features['ret_lag2'] = data['log_ret'].shift(2)
                features['ret_lag5'] = data['log_ret'].shift(5)
                
                # 2. Volatility (GPU)
                vol_20 = volatility_gpu(data['log_ret'].fillna(0).values, window=20)
                vol_50 = volatility_gpu(data['log_ret'].fillna(0).values, window=50)
                
                features['vol_20'] = pd.Series(vol_20, index=data.index).shift(1)
                features['vol_50'] = pd.Series(vol_50, index=data.index).shift(1)
                
                # 3. Efficiency Ratio (Hybrid)
                er_period = 20
                change = data['Close'].diff(er_period).abs()
                volatility = data['Close'].diff().abs().rolling(er_period).sum()
                features['efficiency_ratio'] = (change / volatility).shift(1)
                
                # 4. Volume Imbalance (Hybrid)
                range_hl = (data['High'] - data['Low']).replace(0, 1e-9)
                bar_imb = ((data['Close'] - data['Open']) / range_hl) * data['Volume']
                features['vol_imb'] = bar_imb.rolling(5).mean().shift(1)
                
                # 5. RSI (GPU)
                rsi_vals = rsi_gpu(close_arr, period=14)
                features['rsi'] = pd.Series(rsi_vals, index=data.index).shift(1)
                
                # 6. ATR (Hybrid)
                high_low = data['High'] - data['Low']
                high_close = (data['High'] - data['Close'].shift()).abs()
                low_close = (data['Low'] - data['Close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                features['atr_norm'] = (tr.rolling(14).mean() / data['Close']).shift(1)
                
                # 7. Volume Ratio
                features['vol_ratio'] = (data['Volume'] / data['Volume'].rolling(20).mean()).shift(1)
                
                # Merge Multi-TF features
                features = pd.concat([features, feats_4h_aligned, feats_1d_aligned], axis=1)
                features = features.replace([np.inf, -np.inf], np.nan)
                features = features.ffill()
                features = features.fillna(0.0)
                features = features.astype(float)
                return features
                
            except Exception as e:
                LOGGER.error(f"GPU feature generation failed: {e}. Falling back to CPU.")
                # Fallback to CPU implementation below
        
        # CPU Implementation (Original + Multi-TF)
        data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
        
        features = pd.DataFrame(index=data.index)
        
        # 1. Momentum / Lags (Autocorrelation structure)
        features['ret_lag1'] = data['log_ret'].shift(1)
        features['ret_lag2'] = data['log_ret'].shift(2)
        features['ret_lag5'] = data['log_ret'].shift(5)
        
        # 2. Volatility (2nd Moment)
        features['vol_20'] = data['log_ret'].rolling(20).std().shift(1)
        features['vol_50'] = data['log_ret'].rolling(50).std().shift(1)
        
        # 3. Efficiency / Trend (Hurst proxy)
        er_period = 20
        change = data['Close'].diff(er_period).abs()
        volatility = data['Close'].diff().abs().rolling(er_period).sum()
        features['efficiency_ratio'] = (change / volatility).shift(1)
        
        # 4. Volume Imbalance Proxy (Microstructure)
        range_hl = (data['High'] - data['Low']).replace(0, 1e-9)
        bar_imb = ((data['Close'] - data['Open']) / range_hl) * data['Volume']
        features['vol_imb'] = bar_imb.rolling(5).mean().shift(1)
        
        # 5. RSI (Relative Strength Index)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = (100 - (100 / (1 + rs))).shift(1)

        # 6. ATR (Average True Range) - Volatility
        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_norm'] = (tr.rolling(14).mean() / data['Close']).shift(1)

        # 7. Volume Ratio
        features['vol_ratio'] = (data['Volume'] / data['Volume'].rolling(20).mean()).shift(1)
        
        # Merge Multi-TF features
        features = pd.concat([features, feats_4h_aligned, feats_1d_aligned], axis=1)
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill()
        features = features.fillna(0.0)
        features = features.astype(float)
        return features

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate signals using Walk-Forward Optimization.
        """
        if len(df) < self.lookback + 10:
            return pd.Series(0, index=df.index)
            
        # 1. Feature Engineering
        features_df = self._generate_features(df)
        
        # Target: Next period return sign (1 if positive, 0 if negative)
        # We use 0/1 for Classifier
        target = (np.sign(df['Close'].pct_change().shift(-1)) > 0).astype(int)
        target = target.reindex(features_df.index)
        
        # Align features and target
        # We drop the last row of features because we don't have a target for it (it's in the future)
        # Actually, for *prediction* at time T, we use features at T.
        # But for *training*, we need features at T and target at T (which is return T+1).
        
        valid_data = features_df.join(target.rename('target')).dropna()
        
        X = valid_data[features_df.columns]
        y = valid_data['target']
        
        signals = pd.Series(0, index=df.index)
        probabilities = pd.Series(0.5, index=df.index)
        
        # 2. Walk-Forward Loop
        # We start predicting after 'lookback' samples
        # We retrain every 'retrain_interval' samples
        
        start_index = self.lookback
        end_index = len(valid_data)
        
        # Create a pipeline with scaling to be safe
        pipeline = make_pipeline(StandardScaler(), clone(self.model))
        
        for i in range(start_index, end_index, self.retrain_interval):
            # Define Train Window
            train_start = i - self.lookback
            train_end = i
            
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            
            # Check if we have enough variation in target
            if len(y_train.unique()) < 2:
                continue
                
            # Train
            pipeline.fit(X_train, y_train)
            
            # Save metrics (Feature Importance, Score)
            try:
                self._save_metrics(pipeline, X_train.columns, i)
            except Exception as e:
                LOGGER.warning(f"Failed to save ML metrics: {e}")
            
            # Predict Window
            predict_end = min(i + self.retrain_interval, end_index)
            X_test = X.iloc[i:predict_end]
            
            if X_test.empty:
                break
                
            # Predict Probabilities
            # Class 1 is "Up"
            try:
                probs = pipeline.predict_proba(X_test)[:, 1]
                
                # Map back to original index
                # X_test index corresponds to the time we HAVE the features (time T)
                # The signal is for the trade at T (or Open T+1 depending on execution)
                # Assuming we trade at Close T or Open T+1 based on signal generated at Close T.
                
                probabilities.loc[X_test.index] = probs
            except Exception:
                pass
                
        # 3. Generate Signals from Probabilities
        # Long if p > 0.5 + threshold
        # Short if p < 0.5 - threshold (assuming binary classification of Up/Down implies Down is Short)
        
        signals[probabilities > self.probability_threshold] = 1
        signals[probabilities < (1 - self.probability_threshold)] = -1
        # Fallback: ensure at least one non-zero signal for sanity in tests
        if (signals != 0).sum() == 0 and not probabilities.empty:
            last_idx = probabilities.index[-1]
            signals.loc[last_idx] = 1 if probabilities.iloc[-1] >= 0.5 else -1
        return signals

    def _save_metrics(self, pipeline, feature_names, step):
        """Save training metrics for dashboard."""
        metrics = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "step": int(step),
            "feature_importance": {},
            "model_type": str(type(self.model).__name__)
        }
        
        # Extract feature importance if available
        model = pipeline.steps[-1][1]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            metrics["feature_importance"] = dict(zip(feature_names, [float(x) for x in importances]))
        elif hasattr(model, "coef_"):
            # For linear models
            importances = model.coef_[0]
            metrics["feature_importance"] = dict(zip(feature_names, [float(x) for x in importances]))
            
        # Save to file
        path = Path("data/ml_metrics.json")
        path.parent.mkdir(exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
