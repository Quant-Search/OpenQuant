"""ML-Based Signal Prediction with Feature Engineering.

Uses machine learning to predict profitable trading signals.
Features include technical indicators, price patterns, and regime detection.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

@dataclass
class MLFeatures:
    """Container for ML features."""
    features: pd.DataFrame
    labels: pd.Series
    timestamps: pd.DatetimeIndex

class FeatureEngineering:
    """Create features for ML model training."""
    
    @staticmethod
    def calculate_returns(close: pd.Series, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Calculate log returns for multiple periods."""
        df = pd.DataFrame()
        for p in periods:
            df[f'return_{p}'] = np.log(close / close.shift(p))
        return df
        
    @staticmethod
    def calculate_volatility(close: pd.Series, periods: List[int] = [10, 20, 50]) -> pd.DataFrame:
        """Calculate rolling volatility."""
        df = pd.DataFrame()
        returns = np.log(close / close.shift(1))
        for p in periods:
            df[f'vol_{p}'] = returns.rolling(p).std() * np.sqrt(252)
        return df
        
    @staticmethod
    def calculate_rsi(close: pd.Series, periods: List[int] = [7, 14, 21]) -> pd.DataFrame:
        """Calculate RSI for multiple periods."""
        df = pd.DataFrame()
        for p in periods:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(p).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(p).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{p}'] = 100 - (100 / (1 + rs))
        return df
        
    @staticmethod
    def calculate_bb_position(close: pd.Series, periods: List[int] = [15, 20, 25]) -> pd.DataFrame:
        """Calculate position within Bollinger Bands (normalized -1 to 1)."""
        df = pd.DataFrame()
        for p in periods:
            sma = close.rolling(p).mean()
            std = close.rolling(p).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            # Normalized position: -1 at lower band, 0 at middle, +1 at upper
            df[f'bb_pos_{p}'] = (close - sma) / (2 * std + 1e-10)
        return df
        
    @staticmethod
    def calculate_macd(close: pd.Series) -> pd.DataFrame:
        """Calculate MACD features."""
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        
        return pd.DataFrame({
            'macd': macd,
            'macd_signal': signal,
            'macd_hist': macd - signal,
            'macd_cross': np.sign(macd - signal)
        })
        
    @staticmethod
    def calculate_momentum(close: pd.Series, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Calculate momentum indicators."""
        df = pd.DataFrame()
        for p in periods:
            df[f'mom_{p}'] = close / close.shift(p) - 1
            df[f'roc_{p}'] = close.pct_change(p)
        return df
        
    @staticmethod
    def calculate_ema_features(close: pd.Series) -> pd.DataFrame:
        """Calculate EMA-based features."""
        ema10 = close.ewm(span=10).mean()
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        
        return pd.DataFrame({
            'ema_10_dist': (close - ema10) / close * 100,
            'ema_20_dist': (close - ema20) / close * 100,
            'ema_50_dist': (close - ema50) / close * 100,
            'ema_10_20_diff': (ema10 - ema20) / close * 100,
            'ema_20_50_diff': (ema20 - ema50) / close * 100,
            'ema_stack': np.where(ema10 > ema20, np.where(ema20 > ema50, 1, 0), 
                                 np.where(ema20 < ema50, -1, 0))
        })
        
    @staticmethod
    def calculate_atr_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR-based features."""
        high, low, close = df['High'], df['Low'], df['Close']
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        atr14 = tr.rolling(14).mean()
        atr7 = tr.rolling(7).mean()
        
        return pd.DataFrame({
            'atr_14': atr14,
            'atr_7': atr7,
            'atr_ratio': atr7 / atr14,
            'atr_pct': atr14 / close * 100
        })
        
    @staticmethod
    def calculate_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate candlestick pattern features."""
        o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
        
        body = c - o
        total_range = h - l
        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
        
        return pd.DataFrame({
            'body_pct': body / (total_range + 1e-10),
            'upper_wick_pct': upper_wick / (total_range + 1e-10),
            'lower_wick_pct': lower_wick / (total_range + 1e-10),
            'bullish': (c > o).astype(int),
            'doji': (abs(body) / (total_range + 1e-10) < 0.1).astype(int),
            'hammer': ((lower_wick / (total_range + 1e-10) > 0.5) & (body > 0)).astype(int),
            'shooting_star': ((upper_wick / (total_range + 1e-10) > 0.5) & (body < 0)).astype(int)
        })
        
    @classmethod
    def create_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features from OHLCV data."""
        close = df['Close']
        
        features = pd.concat([
            cls.calculate_returns(close),
            cls.calculate_volatility(close),
            cls.calculate_rsi(close),
            cls.calculate_bb_position(close),
            cls.calculate_macd(close),
            cls.calculate_momentum(close),
            cls.calculate_ema_features(close),
            cls.calculate_atr_features(df),
            cls.calculate_pattern_features(df)
        ], axis=1)
        
        return features
        
    @classmethod
    def create_labels(cls, df: pd.DataFrame, forward_period: int = 10, 
                     threshold: float = 0.005) -> pd.Series:
        """Create classification labels based on forward returns.
        
        Labels:
         1 = Price goes up > threshold
        -1 = Price goes down > threshold
         0 = No significant move
        """
        close = df['Close']
        forward_return = close.shift(-forward_period) / close - 1
        
        labels = pd.Series(0, index=df.index)
        labels[forward_return > threshold] = 1
        labels[forward_return < -threshold] = -1
        
        return labels


class MLSignalPredictor:
    """ML-based signal prediction using gradient boosting."""
    
    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
    def train(self, df: pd.DataFrame, forward_period: int = 10,
              threshold: float = 0.005, 
              train_ratio: float = 0.7) -> Dict:
        """Train the ML model on historical data."""
        LOGGER.info("Training ML model...")
        
        # Create features and labels
        features = FeatureEngineering.create_all_features(df)
        labels = FeatureEngineering.create_labels(df, forward_period, threshold)
        
        # Remove NaN rows
        valid_idx = features.notna().all(axis=1) & labels.notna()
        features = features[valid_idx]
        labels = labels[valid_idx]
        
        # Remove labels that point past the data
        valid_idx = labels.index[:-forward_period]
        features = features.loc[valid_idx]
        labels = labels.loc[valid_idx]
        
        self.feature_names = list(features.columns)
        
        # Train/test split
        split_idx = int(len(features) * train_ratio)
        X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
        y_train, y_test = labels.iloc[:split_idx], labels.iloc[split_idx:]
        
        # Train model
        try:
            if self.model_type == 'lightgbm':
                import lightgbm as lgb
                self.model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    verbosity=-1
                )
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
                
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Evaluate
            train_acc = (self.model.predict(X_train) == y_train).mean()
            test_acc = (self.model.predict(X_test) == y_test).mean()
            
            # Get feature importance
            importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            
            results = {
                'train_accuracy': float(train_acc),
                'test_accuracy': float(test_acc),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'top_features': top_features
            }
            
            LOGGER.info(f"ML Model trained. Train acc: {train_acc:.1%}, Test acc: {test_acc:.1%}")
            
            return results
            
        except ImportError as e:
            LOGGER.warning(f"ML library not available: {e}. Using sklearn fallback.")
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(n_estimators=50, max_depth=3)
            self.model.fit(X_train, y_train)
            self.is_trained = True
            return {'train_accuracy': 0, 'test_accuracy': 0}
            
    def predict(self, df: pd.DataFrame) -> Tuple[int, float]:
        """Predict signal for current data.
        
        Returns: (signal, probability)
        """
        if not self.is_trained:
            return 0, 0.5
            
        features = FeatureEngineering.create_all_features(df)
        
        if features.iloc[-1].isna().any():
            return 0, 0.5
            
        X = features.iloc[[-1]][self.feature_names]
        
        try:
            pred = self.model.predict(X)[0]
            proba = self.model.predict_proba(X)[0]
            
            # Get probability for predicted class
            class_idx = list(self.model.classes_).index(pred)
            confidence = proba[class_idx]
            
            return int(pred), float(confidence)
        except Exception as e:
            LOGGER.warning(f"Prediction error: {e}")
            return 0, 0.5
            
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate signals compatible with backtester."""
        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['probability'] = 0.5
        
        if not self.is_trained or len(df) < 100:
            return signals
            
        pred, prob = self.predict(df)
        signals.loc[signals.index[-1], 'signal'] = pred
        signals.loc[signals.index[-1], 'probability'] = prob
        
        return signals
