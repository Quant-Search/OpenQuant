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
from sklearn.pipeline import make_pipeline

from openquant.strategies.base import BaseStrategy

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
        AVOIDS "retail" indicators; focuses on moments and microstructure proxies.
        """
        data = df.copy()
        data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))
        
        features = pd.DataFrame(index=data.index)
        
        # 1. Momentum / Lags (Autocorrelation structure)
        features['ret_lag1'] = data['log_ret'].shift(1)
        features['ret_lag2'] = data['log_ret'].shift(2)
        features['ret_lag5'] = data['log_ret'].shift(5)
        
        # 2. Volatility (2nd Moment)
        # Rolling standard deviation of returns
        features['vol_20'] = data['log_ret'].rolling(20).std().shift(1)
        features['vol_50'] = data['log_ret'].rolling(50).std().shift(1)
        
        # 3. Efficiency / Trend (Hurst proxy)
        # Efficiency Ratio: Net change / Sum of absolute changes
        er_period = 20
        change = data['Close'].diff(er_period).abs()
        volatility = data['Close'].diff().abs().rolling(er_period).sum()
        features['efficiency_ratio'] = (change / volatility).shift(1)
        
        # 4. Volume Imbalance Proxy (Microstructure)
        # (Close - Open) / (High - Low) * Volume
        # Normalized by moving average volume
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

        return features.dropna()

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
        
        return signals
