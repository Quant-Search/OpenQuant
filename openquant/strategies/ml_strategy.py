"""
Machine Learning Strategy.

Implements a rigorous Walk-Forward Optimization framework for generating trading signals
using scikit-learn models. Focuses on statistical features and probabilistic outputs.
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
                 model: BaseEstimator | None = None,
                 lookback: int = 500,
                 retrain_interval: int = 50,
                 probability_threshold: float = 0.55,
                 feature_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.model: BaseEstimator = model if model is not None else RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
        self.lookback: int = lookback
        self.retrain_interval: int = retrain_interval
        self.probability_threshold: float = probability_threshold
        self.feature_config: dict[str, Any] = feature_config or {}

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate statistical features from OHLCV data.
        Includes multi-timeframe analysis and advanced microstructure proxies.
        """
        try:
            from openquant.features.gpu_features import (
                is_gpu_features_available,
                log_returns_gpu,
                rsi_gpu,
                volatility_gpu,
            )
            use_gpu: bool = is_gpu_features_available()
        except ImportError:
            use_gpu = False

        data = df.copy()

        df_4h = df.resample('4h').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

        df_1d = df.resample('1d').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()

        def calc_tf_features(tf_df: pd.DataFrame, suffix: str) -> pd.DataFrame:
            tf_data = tf_df.copy()
            tf_data['log_ret'] = np.log(tf_data['Close'] / tf_data['Close'].shift(1))

            feats = pd.DataFrame(index=tf_data.index)
            feats[f'trend_{suffix}'] = tf_data['log_ret'].rolling(10).mean()
            feats[f'vol_{suffix}'] = tf_data['log_ret'].rolling(20).std()

            delta = tf_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            feats[f'rsi_{suffix}'] = 100 - (100 / (1 + rs))

            return feats.shift(1)

        feats_4h = calc_tf_features(df_4h, '4h')
        feats_1d = calc_tf_features(df_1d, '1d')

        feats_4h_aligned = feats_4h.reindex(data.index, method='ffill')
        feats_1d_aligned = feats_1d.reindex(data.index, method='ffill')

        if use_gpu:
            try:
                close_arr = data['Close'].values

                data['log_ret'] = log_returns_gpu(close_arr)

                features = pd.DataFrame(index=data.index)

                features['ret_lag1'] = data['log_ret'].shift(1)
                features['ret_lag2'] = data['log_ret'].shift(2)
                features['ret_lag5'] = data['log_ret'].shift(5)

                vol_20 = volatility_gpu(data['log_ret'].fillna(0).values, window=20)
                vol_50 = volatility_gpu(data['log_ret'].fillna(0).values, window=50)

                features['vol_20'] = pd.Series(vol_20, index=data.index).shift(1)
                features['vol_50'] = pd.Series(vol_50, index=data.index).shift(1)

                er_period = 20
                change = data['Close'].diff(er_period).abs()
                volatility = data['Close'].diff().abs().rolling(er_period).sum()
                features['efficiency_ratio'] = (change / volatility).shift(1)

                range_hl = (data['High'] - data['Low']).replace(0, 1e-9)
                bar_imb = ((data['Close'] - data['Open']) / range_hl) * data['Volume']
                features['vol_imb'] = bar_imb.rolling(5).mean().shift(1)

                rsi_vals = rsi_gpu(close_arr, period=14)
                features['rsi'] = pd.Series(rsi_vals, index=data.index).shift(1)

                high_low = data['High'] - data['Low']
                high_close = (data['High'] - data['Close'].shift()).abs()
                low_close = (data['Low'] - data['Close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                features['atr_norm'] = (tr.rolling(14).mean() / data['Close']).shift(1)

                features['vol_ratio'] = (data['Volume'] / data['Volume'].rolling(20).mean()).shift(1)

                features = pd.concat([features, feats_4h_aligned, feats_1d_aligned], axis=1)
                features = features.replace([np.inf, -np.inf], np.nan)
                features = features.ffill()
                features = features.fillna(0.0)
                features = features.astype(float)
                return features

            except Exception as e:
                LOGGER.error(f"GPU feature generation failed: {e}. Falling back to CPU.")

        data['log_ret'] = np.log(data['Close'] / data['Close'].shift(1))

        features = pd.DataFrame(index=data.index)

        features['ret_lag1'] = data['log_ret'].shift(1)
        features['ret_lag2'] = data['log_ret'].shift(2)
        features['ret_lag5'] = data['log_ret'].shift(5)

        features['vol_20'] = data['log_ret'].rolling(20).std().shift(1)
        features['vol_50'] = data['log_ret'].rolling(50).std().shift(1)

        er_period = 20
        change = data['Close'].diff(er_period).abs()
        volatility = data['Close'].diff().abs().rolling(er_period).sum()
        features['efficiency_ratio'] = (change / volatility).shift(1)

        range_hl = (data['High'] - data['Low']).replace(0, 1e-9)
        bar_imb = ((data['Close'] - data['Open']) / range_hl) * data['Volume']
        features['vol_imb'] = bar_imb.rolling(5).mean().shift(1)

        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = (100 - (100 / (1 + rs))).shift(1)

        high_low = data['High'] - data['Low']
        high_close = (data['High'] - data['Close'].shift()).abs()
        low_close = (data['Low'] - data['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_norm'] = (tr.rolling(14).mean() / data['Close']).shift(1)

        features['vol_ratio'] = (data['Volume'] / data['Volume'].rolling(20).mean()).shift(1)

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
            return pd.Series(0, index=df.index, dtype=int)

        features_df = self._generate_features(df)

        target = (np.sign(df['Close'].pct_change().shift(-1)) > 0).astype(int)
        target = target.reindex(features_df.index)

        valid_data = features_df.join(target.rename('target')).dropna()

        X = valid_data[features_df.columns]
        y = valid_data['target']

        signals = pd.Series(0, index=df.index, dtype=int)
        probabilities = pd.Series(0.5, index=df.index, dtype=float)

        start_index: int = self.lookback
        end_index: int = len(valid_data)

        pipeline = make_pipeline(StandardScaler(), clone(self.model))

        for i in range(start_index, end_index, self.retrain_interval):
            train_start = i - self.lookback
            train_end = i

            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]

            if len(y_train.unique()) < 2:
                continue

            pipeline.fit(X_train, y_train)

            try:
                self._save_metrics(pipeline, X_train.columns, i)
            except Exception as e:
                LOGGER.warning(f"Failed to save ML metrics: {e}")

            predict_end = min(i + self.retrain_interval, end_index)
            X_test = X.iloc[i:predict_end]

            if X_test.empty:
                break

            try:
                probs = pipeline.predict_proba(X_test)[:, 1]

                probabilities.loc[X_test.index] = probs
            except Exception:
                pass

        signals[probabilities > self.probability_threshold] = 1
        signals[probabilities < (1 - self.probability_threshold)] = -1
        if (signals != 0).sum() == 0 and not probabilities.empty:
            last_idx = probabilities.index[-1]
            signals.loc[last_idx] = 1 if probabilities.iloc[-1] >= 0.5 else -1
        return signals

    def _save_metrics(self, pipeline: Any, feature_names: pd.Index, step: int) -> None:
        """Save training metrics for dashboard."""
        metrics: dict[str, Any] = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "step": int(step),
            "feature_importance": {},
            "model_type": str(type(self.model).__name__)
        }

        model = pipeline.steps[-1][1]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            metrics["feature_importance"] = dict(zip(feature_names, [float(x) for x in importances], strict=False))
        elif hasattr(model, "coef_"):
            importances = model.coef_[0]
            metrics["feature_importance"] = dict(zip(feature_names, [float(x) for x in importances], strict=False))

        path = Path("data/ml_metrics.json")
        path.parent.mkdir(exist_ok=True)
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
