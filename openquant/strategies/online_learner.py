"""Online Learner for Real-time Adaptation.

Implements incremental learning using SGDClassifier to adapt to changing market conditions
based on closed trades.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

class OnlineLearner:
    """
    Online learning module that updates a model incrementally after each trade.
    """
    def __init__(self, model_path: str = "data/online_model.joblib"):
        self.model_path = Path(model_path)
        self.model = SGDClassifier(loss="log_loss", penalty="l2", alpha=0.0001, random_state=42)
        self.scaler = StandardScaler()
        self._buffer_X = []
        self._buffer_y = []
        self.is_fitted = False
        self._load_model()
        
    def _load_model(self):
        """Load existing model if available."""
        if self.model_path.exists():
            try:
                data = joblib.load(self.model_path)
                self.model = data["model"]
                self.scaler = data["scaler"]
                self.is_fitted = True
                LOGGER.info("Loaded online learning model.")
            except Exception as e:
                LOGGER.error(f"Failed to load online model: {e}")
                
    def _save_model(self):
        """Save model state."""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": self.model, "scaler": self.scaler}, self.model_path)
        except Exception as e:
            LOGGER.error(f"Failed to save online model: {e}")
            
    def update(self, features: np.ndarray, result: int):
        """
        Update the model with a single trade result.
        
        Args:
            features: Feature vector associated with the trade entry.
            result: 1 for Win, 0 for Loss (or -1 depending on convention).
                    Here we assume binary classification: 1=Good Trade, 0=Bad Trade.
        """
        # Reshape for sklearn
        X = np.array(features).reshape(1, -1)
        y = np.array([result])

        # Accumulate small buffer to stabilize scaler stats
        self._buffer_X.append(X)
        self._buffer_y.append(y)
        if len(self._buffer_X) >= 5:
            Xb = np.vstack(self._buffer_X)
            yb = np.vstack(self._buffer_y).ravel()
            try:
                self.scaler.partial_fit(Xb)
                Xs = self.scaler.transform(Xb)
            except Exception:
                Xs = Xb
            classes = np.array([0, 1])
            self.model.partial_fit(Xs, yb, classes=classes)
            self._buffer_X.clear()
            self._buffer_y.clear()
            self.is_fitted = True
            self._save_model()
        LOGGER.info(f"Online model updated. Prediction for this trade was: {self.predict_proba(features):.2f}")
        
    def predict_proba(self, features: np.ndarray) -> float:
        """Predict probability of a trade being successful."""
        if not self.is_fitted:
            return 0.5
            
        X = np.array(features).reshape(1, -1)
        try:
            Xs = self.scaler.transform(X)
        except Exception:
            Xs = X
        try:
            return self.model.predict_proba(Xs)[0, 1]
        except Exception:
            return 0.5
