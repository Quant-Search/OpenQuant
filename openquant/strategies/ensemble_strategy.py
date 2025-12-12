"""Ensemble Strategy - Combines Multiple Strategies for Diversification.

Combines signals from multiple strategies using voting or weighted average
to reduce variance and improve consistency.
"""
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from ..quant.regime_detector import RegimeDetector
from ..utils.logging import get_logger
from .base import BaseStrategy
from .ml_strategy import MLStrategy

LOGGER = get_logger(__name__)

class CombineMethod(Enum):
    """How to combine signals from multiple strategies."""
    VOTING = "voting"
    AVERAGE = "average"
    WEIGHTED = "weighted"
    CONFIDENT = "confident"

@dataclass
class StrategySummary:
    """Summary of a strategy's recent performance."""
    name: str
    win_rate: float
    avg_return: float
    sharpe: float
    weight: float

class EnsembleStrategy(BaseStrategy):
    """
    Ensemble Strategy combining multiple signal sources.

    Features:
    - Combines ML, momentum, and mean-reversion strategies
    - Uses voting or weighted averaging
    - Regime-aware: activates different sub-strategies per regime
    - Tracks individual strategy performance for adaptive weighting
    """

    def __init__(
        self,
        combine_method: CombineMethod = CombineMethod.VOTING,
        min_agreement: float = 0.4,
        use_regime_filter: bool = True,
        probability_threshold: float = 0.52
    ) -> None:
        super().__init__()
        self.combine_method: CombineMethod = combine_method
        self.min_agreement: float = min_agreement
        self.use_regime_filter: bool = use_regime_filter
        self.probability_threshold: float = probability_threshold

        self.ml_strategy: MLStrategy = MLStrategy(
            lookback=500,
            probability_threshold=probability_threshold
        )

        self.regime_detector: RegimeDetector = RegimeDetector(lookback=100)

        self.strategy_stats: dict[str, list[float]] = {
            "ml": [],
            "momentum": [],
            "mean_reversion": []
        }

    def _momentum_signal(self, df: pd.DataFrame) -> tuple[int, float]:
        """
        Simple momentum strategy: trend following.
        Returns (signal, confidence) where signal is 1 (long), -1 (short), 0 (neutral)
        """
        if len(df) < 50:
            return 0, 0.5

        close = df['Close']

        sma_fast = close.rolling(10).mean().iloc[-1]
        sma_slow = close.rolling(50).mean().iloc[-1]
        current_price = close.iloc[-1]

        signal: int = 0
        confidence: float = 0.5

        if current_price > sma_fast > sma_slow:
            signal = 1
            confidence = 0.65
        elif current_price < sma_fast < sma_slow:
            signal = -1
            confidence = 0.65
        elif current_price > sma_slow:
            signal = 1
            confidence = 0.55
        elif current_price < sma_slow:
            signal = -1
            confidence = 0.55

        return signal, confidence

    def _mean_reversion_signal(self, df: pd.DataFrame) -> tuple[int, float]:
        """
        Mean reversion strategy: trade back to mean.
        Best in ranging markets (Hurst < 0.5).
        """
        if len(df) < 30:
            return 0, 0.5

        close = df['Close']

        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std

        current_price = close.iloc[-1]
        upper_val = upper.iloc[-1]
        lower_val = lower.iloc[-1]
        sma.iloc[-1]

        if upper_val == lower_val:
            return 0, 0.5

        band_position = (current_price - lower_val) / (upper_val - lower_val)

        signal: int = 0
        confidence: float = 0.5

        if band_position > 0.9:
            signal = -1
            confidence = 0.60 + (band_position - 0.9) * 2
        elif band_position < 0.1:
            signal = 1
            confidence = 0.60 + (0.1 - band_position) * 2

        return signal, min(confidence, 0.75)

    def _rsi_signal(self, df: pd.DataFrame, period: int = 14) -> tuple[int, float]:
        """RSI-based signal - oversold/overbought."""
        if len(df) < period + 5:
            return 0, 0.5

        close = df['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        if current_rsi < 30:
            return 1, 0.60 + (30 - current_rsi) / 100
        elif current_rsi > 70:
            return -1, 0.60 + (current_rsi - 70) / 100
        elif current_rsi < 40:
            return 1, 0.55
        elif current_rsi > 60:
            return -1, 0.55

        return 0, 0.5

    def _macd_signal(self, df: pd.DataFrame) -> tuple[int, float]:
        """MACD crossover signal."""
        if len(df) < 30:
            return 0, 0.5

        close = df['Close']

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()

        current_macd = macd.iloc[-1]
        prev_macd = macd.iloc[-2]
        current_signal = signal_line.iloc[-1]
        prev_signal = signal_line.iloc[-2]

        if prev_macd <= prev_signal and current_macd > current_signal:
            return 1, 0.65
        elif prev_macd >= prev_signal and current_macd < current_signal:
            return -1, 0.65
        elif current_macd > current_signal:
            return 1, 0.55
        elif current_macd < current_signal:
            return -1, 0.55

        return 0, 0.5

    def _stochastic_signal(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[int, float]:
        """Stochastic oscillator signal."""
        if len(df) < k_period + d_period:
            return 0, 0.5

        high = df['High']
        low = df['Low']
        close = df['Close']

        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = k.rolling(d_period).mean()

        current_k = k.iloc[-1]
        current_d = d.iloc[-1]
        prev_k = k.iloc[-2]
        prev_d = d.iloc[-2]

        if current_k < 20 and prev_k <= prev_d and current_k > current_d:
            return 1, 0.70
        elif current_k > 80 and prev_k >= prev_d and current_k < current_d:
            return -1, 0.70
        elif current_k < 30:
            return 1, 0.55
        elif current_k > 70:
            return -1, 0.55

        return 0, 0.5

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble signals by combining sub-strategies.
        """
        LOGGER.info("EnsembleStrategy: Generating signals...")

        signals = pd.DataFrame(index=df.index)
        signals['signal'] = 0
        signals['probability'] = 0.5

        if len(df) < 100:
            return signals

        regime_info = self.regime_detector.detect_regime(df)
        trend_regime = regime_info['trend_regime']
        hurst = regime_info['hurst_exponent']

        LOGGER.info(f"Market regime: {trend_regime}, Hurst: {hurst:.2f}")

        strategy_signals: list[tuple[str, int, float]] = []

        try:
            ml_signals = self.ml_strategy.generate_signals(df)
            if not ml_signals.empty and 'signal' in ml_signals.columns:
                last_ml_signal = int(ml_signals['signal'].iloc[-1])
                ml_prob = float(ml_signals.get('probability', pd.Series([0.5])).iloc[-1])
                strategy_signals.append(("ml", last_ml_signal, ml_prob))
        except Exception as e:
            LOGGER.debug(f"ML strategy skipped: {e}")

        mom_signal, mom_conf = self._momentum_signal(df)
        if self.use_regime_filter and hurst < 0.50:
            mom_conf *= 0.7
        strategy_signals.append(("momentum", mom_signal, mom_conf))

        mr_signal, mr_conf = self._mean_reversion_signal(df)
        if self.use_regime_filter and hurst > 0.55:
            mr_conf *= 0.7
        strategy_signals.append(("mean_reversion", mr_signal, mr_conf))

        rsi_signal, rsi_conf = self._rsi_signal(df)
        strategy_signals.append(("rsi", rsi_signal, rsi_conf))

        macd_signal, macd_conf = self._macd_signal(df)
        strategy_signals.append(("macd", macd_signal, macd_conf))

        stoch_signal, stoch_conf = self._stochastic_signal(df)
        strategy_signals.append(("stochastic", stoch_signal, stoch_conf))

        if self.combine_method == CombineMethod.VOTING:
            final_signal, final_prob = self._voting_combine(strategy_signals)
        elif self.combine_method == CombineMethod.AVERAGE:
            final_signal, final_prob = self._average_combine(strategy_signals)
        elif self.combine_method == CombineMethod.CONFIDENT:
            final_signal, final_prob = self._confident_combine(strategy_signals)
        else:
            final_signal, final_prob = self._weighted_combine(strategy_signals)

        if final_prob < self.probability_threshold:
            final_signal = 0

        signals.loc[signals.index[-1], 'signal'] = final_signal
        signals.loc[signals.index[-1], 'probability'] = final_prob

        LOGGER.info(f"Ensemble signal: {final_signal} (prob: {final_prob:.2%})")

        return signals

    def _voting_combine(
        self,
        strategies: list[tuple[str, int, float]]
    ) -> tuple[int, float]:
        """Majority voting combination - more active version."""
        longs = sum(1 for _, sig, conf in strategies if sig > 0 and conf > 0.50)
        shorts = sum(1 for _, sig, conf in strategies if sig < 0 and conf > 0.50)
        total = len(strategies)

        effective_agreement = max(0.4, self.min_agreement)

        if longs / total >= effective_agreement:
            long_confs = [conf for _, sig, conf in strategies if sig > 0 and conf > 0.5]
            bonus = min(0.1, (longs / total - effective_agreement) * 0.5)
            return 1, np.mean(long_confs) + bonus if long_confs else 0.55

        if shorts / total >= effective_agreement:
            short_confs = [conf for _, sig, conf in strategies if sig < 0 and conf > 0.5]
            bonus = min(0.1, (shorts / total - effective_agreement) * 0.5)
            return -1, np.mean(short_confs) + bonus if short_confs else 0.55

        if longs / total >= 0.3 and longs > shorts:
            long_confs = [conf for _, sig, conf in strategies if sig > 0]
            return 1, np.mean(long_confs) * 0.9 if long_confs else 0.52

        if shorts / total >= 0.3 and shorts > longs:
            short_confs = [conf for _, sig, conf in strategies if sig < 0]
            return -1, np.mean(short_confs) * 0.9 if short_confs else 0.52

        return 0, 0.5

    def _average_combine(
        self,
        strategies: list[tuple[str, int, float]]
    ) -> tuple[int, float]:
        """Simple average of signals."""
        if not strategies:
            return 0, 0.5

        avg_signal = np.mean([sig for _, sig, _ in strategies])
        avg_conf = np.mean([conf for _, _, conf in strategies])

        if avg_signal > 0.3:
            return 1, avg_conf
        elif avg_signal < -0.3:
            return -1, avg_conf
        return 0, 0.5

    def _confident_combine(
        self,
        strategies: list[tuple[str, int, float]]
    ) -> tuple[int, float]:
        """Take the signal with highest confidence."""
        if not strategies:
            return 0, 0.5

        best = max(strategies, key=lambda x: x[2])
        return best[1], best[2]

    def _weighted_combine(
        self,
        strategies: list[tuple[str, int, float]]
    ) -> tuple[int, float]:
        """Performance-weighted average (based on recent win rates)."""
        if not strategies:
            return 0, 0.5

        weighted_signal: float = 0.0
        total_weight: float = 0.0

        for name, sig, conf in strategies:
            history = self.strategy_stats.get(name, [])
            if history:
                win_rate = sum(1 for x in history if x > 0) / len(history)
            else:
                win_rate = 0.5

            weight = conf * (0.5 + win_rate)
            weighted_signal += sig * weight
            total_weight += weight

        if total_weight == 0:
            return 0, 0.5

        avg_signal = weighted_signal / total_weight
        avg_conf = np.mean([conf for _, _, conf in strategies])

        if avg_signal > 0.3:
            return 1, avg_conf
        elif avg_signal < -0.3:
            return -1, avg_conf
        return 0, 0.5

    def update_performance(self, strategy_name: str, pnl: float) -> None:
        """Update strategy performance for weighted combining."""
        if strategy_name in self.strategy_stats:
            self.strategy_stats[strategy_name].append(pnl)
            self.strategy_stats[strategy_name] = self.strategy_stats[strategy_name][-100:]
