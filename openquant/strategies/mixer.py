"""
Strategy Mixer.
Combines multiple strategies into a single portfolio strategy using weighted voting.
Enhanced with dynamic weight adjustment, correlation-aware portfolio construction,
and regime-dependent strategy selection.
"""
from __future__ import annotations

from typing import Any
from collections import deque
import logging

import numpy as np
import pandas as pd

from openquant.backtest.metrics import sharpe
from openquant.evaluation.regime import compute_regime_features
from openquant.utils.validation import normalize_weights, ValidationError

logger = logging.getLogger(__name__)


class StrategyMixer:
    """
    Combines signals from multiple strategies with advanced features:
    - Dynamic weight adjustment based on rolling Sharpe ratios
    - Correlation-aware portfolio construction
    - Regime-dependent strategy selection
    """
    def __init__(
        self, 
        strategies: list[Any], 
        weights: list[float] | None = None,
        enable_dynamic_weights: bool = True,
        enable_correlation_filter: bool = True,
        enable_regime_selection: bool = True,
        rolling_window: int = 100,
        correlation_threshold: float = 0.7,
        sharpe_lookback: int = 50,
        min_weight: float = 0.0,
        max_weight: float = 0.5,
    ) -> None:
        """
        Args:
            strategies: List of instantiated strategy objects (must have generate_signals method).
            weights: List of initial weights for each strategy. If None, equal weights are used.
            enable_dynamic_weights: Enable dynamic weight adjustment based on rolling Sharpe.
            enable_correlation_filter: Enable correlation-aware weighting.
            enable_regime_selection: Enable regime-dependent strategy selection.
            rolling_window: Window size for rolling performance metrics.
            correlation_threshold: Max correlation between strategies for diversification.
            sharpe_lookback: Lookback period for Sharpe ratio calculation.
            min_weight: Minimum weight for any strategy.
            max_weight: Maximum weight for any strategy.
        """
        if not strategies:
            raise ValidationError("strategies list cannot be empty")
        
        self.strategies: list[Any] = strategies
        self.enable_dynamic_weights = enable_dynamic_weights
        self.enable_correlation_filter = enable_correlation_filter
        self.enable_regime_selection = enable_regime_selection
        self.rolling_window = rolling_window
        self.correlation_threshold = correlation_threshold
        self.sharpe_lookback = sharpe_lookback
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        if weights is None:
            self.weights: list[float] = [1.0 / len(strategies)] * len(strategies)
        else:
            if len(weights) != len(strategies):
                raise ValidationError(
                    f"Number of weights ({len(weights)}) must match number of strategies ({len(strategies)})"
                )
            self.weights = normalize_weights(weights, "strategy weights")
        
        self.returns_history: list[deque] = [deque(maxlen=rolling_window) for _ in strategies]
        self.sharpe_history: list[float] = [0.0] * len(strategies)
        self.correlation_matrix: pd.DataFrame | None = None
        self.regime_scores: dict[str, list[float]] = {
            'trending': [1.0] * len(strategies),
            'ranging': [1.0] * len(strategies),
            'high_vol': [1.0] * len(strategies),
            'low_vol': [1.0] * len(strategies),
        }
                
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate combined signals with dynamic weighting, correlation filtering,
        and regime-dependent strategy selection.
        """
        if not self.strategies:
            return pd.Series(0, index=df.index)
        
        regime_info = None
        if self.enable_regime_selection:
            regime_info = compute_regime_features(df)
        
        combined_signal = pd.Series(0.0, index=df.index)
        strategy_signals: list[pd.Series] = []
        strategy_returns: list[pd.Series] = []
        
        market_ret = df['Close'].pct_change().fillna(0)
        
        for i, strat in enumerate(self.strategies):
            try:
                sig = strat.generate_signals(df)
                sig = sig.clip(-1, 1).fillna(0)
                strategy_signals.append(sig)
                
                strat_ret = sig.shift(1).fillna(0) * market_ret
                strategy_returns.append(strat_ret)
                
                self._update_returns_history(i, strat_ret)
                
            except AttributeError as e:
                logger.error(f"Strategy {i} missing generate_signals method: {e}")
                strategy_signals.append(pd.Series(0, index=df.index))
                strategy_returns.append(pd.Series(0, index=df.index))
            except (KeyError, ValueError) as e:
                logger.warning(f"Strategy {i} signal generation failed with data error: {e}")
                strategy_signals.append(pd.Series(0, index=df.index))
                strategy_returns.append(pd.Series(0, index=df.index))
            except Exception as e:
                logger.error(f"Strategy {i} failed with unexpected error: {e}", exc_info=True)
                strategy_signals.append(pd.Series(0, index=df.index))
                strategy_returns.append(pd.Series(0, index=df.index))
        
        if self.enable_dynamic_weights and len(strategy_returns) > 0:
            self._update_dynamic_weights(strategy_returns)
        
        if self.enable_correlation_filter and len(strategy_returns) > 1:
            self._update_correlation_matrix(strategy_returns)
            self._apply_correlation_adjustment()
        
        if self.enable_regime_selection and regime_info:
            self._apply_regime_adjustment(regime_info)
        
        for i, sig in enumerate(strategy_signals):
            combined_signal += sig * self.weights[i]
        
        threshold: float = 0.2
        final_signal = pd.Series(0, index=df.index)
        final_signal[combined_signal > threshold] = 1
        final_signal[combined_signal < -threshold] = -1

        return final_signal

    def _update_returns_history(self, strategy_idx: int, returns: pd.Series) -> None:
        """Update rolling returns history for a strategy."""
        recent_returns = returns.tail(self.rolling_window).values
        for ret in recent_returns:
            if not np.isnan(ret) and not np.isinf(ret):
                self.returns_history[strategy_idx].append(ret)

    def _calculate_rolling_sharpe(self, returns: pd.Series, lookback: int | None = None) -> float:
        """Calculate rolling Sharpe ratio for a strategy."""
        if lookback is None:
            lookback = self.sharpe_lookback
        
        recent_returns = returns.tail(lookback)
        if len(recent_returns) < 10:
            return 0.0
        
        try:
            return sharpe(recent_returns, freq='1d')
        except Exception:
            return 0.0

    def _update_dynamic_weights(self, strategy_returns: list[pd.Series]) -> None:
        """
        Update weights based on rolling Sharpe ratios.
        Strategies with higher recent Sharpe get higher weights.
        """
        sharpe_ratios: list[float] = []
        
        for i, strat_ret in enumerate(strategy_returns):
            if len(self.returns_history[i]) >= 10:
                returns_series = pd.Series(list(self.returns_history[i]))
                sharp = self._calculate_rolling_sharpe(returns_series)
                self.sharpe_history[i] = sharp
                sharpe_ratios.append(max(0.0, sharp))
            else:
                sharpe_ratios.append(0.0)
        
        if sum(sharpe_ratios) == 0:
            return
        
        raw_weights = np.array(sharpe_ratios)
        raw_weights = np.exp(raw_weights / 2.0)
        raw_weights = np.clip(raw_weights, self.min_weight, self.max_weight)
        
        total_weight = raw_weights.sum()
        if total_weight > 0:
            self.weights = (raw_weights / total_weight).tolist()

    def _update_correlation_matrix(self, strategy_returns: list[pd.Series]) -> None:
        """Calculate correlation matrix between strategy returns."""
        try:
            returns_df = pd.DataFrame(strategy_returns).T
            returns_df = returns_df.fillna(0)
            
            if len(returns_df) >= 20:
                self.correlation_matrix = returns_df.tail(self.rolling_window).corr()
            else:
                self.correlation_matrix = None
        except Exception as e:
            logger.warning(f"Correlation matrix calculation failed: {e}")
            self.correlation_matrix = None

    def _apply_correlation_adjustment(self) -> None:
        """
        Adjust weights to penalize highly correlated strategies.
        Promotes diversification.
        """
        if self.correlation_matrix is None or len(self.correlation_matrix) < 2:
            return
        
        n = len(self.weights)
        correlation_penalties = np.ones(n)
        
        for i in range(n):
            penalty = 0.0
            for j in range(n):
                if i != j:
                    corr_val = abs(self.correlation_matrix.iloc[i, j])
                    if corr_val > self.correlation_threshold:
                        penalty += (corr_val - self.correlation_threshold) * self.weights[j]
            
            correlation_penalties[i] = max(0.1, 1.0 - penalty)
        
        adjusted_weights = np.array(self.weights) * correlation_penalties
        adjusted_weights = np.clip(adjusted_weights, self.min_weight, self.max_weight)
        
        total_weight = adjusted_weights.sum()
        if total_weight > 0:
            self.weights = (adjusted_weights / total_weight).tolist()

    def _apply_regime_adjustment(self, regime_info: dict[str, float]) -> None:
        """
        Adjust weights based on market regime.
        Different strategies perform better in different regimes.
        """
        trend_score = regime_info.get('trend_score', 0.0)
        volatility = regime_info.get('volatility', 0.0)
        
        if np.isnan(trend_score) or np.isinf(trend_score):
            trend_score = 0.0
        if np.isnan(volatility) or np.isinf(volatility):
            volatility = 0.01
        
        is_trending = abs(trend_score) > 1.0
        is_high_vol = volatility > 0.02
        
        regime_multipliers = np.ones(len(self.strategies))
        
        for i in range(len(self.strategies)):
            if is_trending:
                regime_multipliers[i] *= self.regime_scores['trending'][i]
            else:
                regime_multipliers[i] *= self.regime_scores['ranging'][i]
            
            if is_high_vol:
                regime_multipliers[i] *= self.regime_scores['high_vol'][i]
            else:
                regime_multipliers[i] *= self.regime_scores['low_vol'][i]
        
        adjusted_weights = np.array(self.weights) * regime_multipliers
        adjusted_weights = np.clip(adjusted_weights, self.min_weight, self.max_weight)
        
        total_weight = adjusted_weights.sum()
        if total_weight > 0:
            self.weights = (adjusted_weights / total_weight).tolist()

    def set_regime_scores(
        self,
        strategy_idx: int,
        trending_score: float = 1.0,
        ranging_score: float = 1.0,
        high_vol_score: float = 1.0,
        low_vol_score: float = 1.0
    ) -> None:
        """
        Set regime-specific performance scores for a strategy.
        Scores > 1.0 indicate the strategy performs well in that regime.
        Scores < 1.0 indicate the strategy performs poorly in that regime.
        
        Args:
            strategy_idx: Index of the strategy.
            trending_score: Performance multiplier in trending markets.
            ranging_score: Performance multiplier in ranging/mean-reverting markets.
            high_vol_score: Performance multiplier in high volatility.
            low_vol_score: Performance multiplier in low volatility.
        """
        if 0 <= strategy_idx < len(self.strategies):
            self.regime_scores['trending'][strategy_idx] = trending_score
            self.regime_scores['ranging'][strategy_idx] = ranging_score
            self.regime_scores['high_vol'][strategy_idx] = high_vol_score
            self.regime_scores['low_vol'][strategy_idx] = low_vol_score

    def get_strategy_stats(self) -> pd.DataFrame:
        """
        Get current statistics for all strategies.
        
        Returns:
            DataFrame with columns: weight, sharpe, avg_return
        """
        stats: list[dict[str, Any]] = []
        for i in range(len(self.strategies)):
            if len(self.returns_history[i]) > 0:
                returns_series = pd.Series(list(self.returns_history[i]))
                avg_return = returns_series.mean()
            else:
                avg_return = 0.0
            
            stats.append({
                'strategy_idx': i,
                'weight': self.weights[i],
                'sharpe': self.sharpe_history[i],
                'avg_return': avg_return,
                'num_trades': len(self.returns_history[i])
            })
        
        return pd.DataFrame(stats)

    def get_correlation_matrix(self) -> pd.DataFrame | None:
        """Get the current correlation matrix between strategies."""
        return self.correlation_matrix

    def optimize_weights(self, df: pd.DataFrame) -> None:
        """
        Find optimal weights based on historical performance (Sharpe).
        Uses scipy.optimize to maximize Portfolio Sharpe Ratio.
        """
        try:
            from scipy.optimize import minimize
        except ImportError as e:
            logger.warning(f"scipy not installed, skipping weight optimization: {e}")
            return

        returns_list: list[pd.Series] = []
        valid_indices: list[int] = []
        
        try:
            market_ret = df['Close'].pct_change().fillna(0)
        except KeyError as e:
            logger.error(f"Missing 'Close' column in dataframe: {e}")
            return
        
        for i, strat in enumerate(self.strategies):
            try:
                sig = strat.generate_signals(df)
                sig = sig.clip(-1, 1).fillna(0)
                
                strat_ret = sig.shift(1).fillna(0) * market_ret

                returns_list.append(strat_ret)
                valid_indices.append(i)
            except AttributeError as e:
                logger.error(f"Strategy {i} missing generate_signals method: {e}")
            except (KeyError, ValueError) as e:
                logger.warning(f"Strategy {i} optimization failed with data error: {e}")
            except Exception as e:
                logger.error(f"Strategy {i} optimization failed with unexpected error: {e}", exc_info=True)
        
        if not returns_list:
            logger.warning("No valid strategy returns for optimization")
            return

        returns_df = pd.DataFrame(returns_list).T.fillna(0)

        def neg_sharpe(weights: np.ndarray) -> float:
            port_ret = (returns_df * weights).sum(axis=1)

            mean_ret = port_ret.mean()
            std_ret = port_ret.std()

            if std_ret == 0:
                return 0.0
            
            return - (mean_ret / std_ret)

        n = len(returns_list)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        initial_weights = [1./n] * n

        try:
            result = minimize(
                neg_sharpe, 
                initial_weights, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            if result.success:
                new_weights: list[float] = [0.0] * len(self.strategies)
                for idx, weight in zip(valid_indices, result.x, strict=False):
                    new_weights[idx] = max(0.0, float(weight))
                
                s = sum(new_weights)
                if s > 0:
                    self.weights = [w/s for w in new_weights]
                
                logger.info(f"Optimized Weights: {[f'{w:.2f}' for w in self.weights]}")
            else:
                logger.warning(f"Optimization failed: {result.message}")
        except ValueError as e:
            logger.error(f"Optimization value error: {e}")
        except RuntimeError as e:
            logger.error(f"Optimization runtime error: {e}")
        except Exception as e:
            logger.error(f"Unexpected optimization error: {e}", exc_info=True)

    def optimize_weights_with_correlation(self, df: pd.DataFrame, correlation_penalty: float = 0.5) -> None:
        """
        Find optimal weights considering both Sharpe ratio and correlation penalty.
        Promotes diversification by penalizing correlated strategies.
        
        Args:
            df: DataFrame with OHLCV data.
            correlation_penalty: Weight for correlation penalty (0 to 1).
        """
        try:
            from scipy.optimize import minimize
        except ImportError as e:
            logger.warning(f"scipy not installed, skipping correlation optimization: {e}")
            return

        returns_list: list[pd.Series] = []
        valid_indices: list[int] = []
        
        try:
            market_ret = df['Close'].pct_change().fillna(0)
        except KeyError as e:
            logger.error(f"Missing 'Close' column in dataframe: {e}")
            return
        
        for i, strat in enumerate(self.strategies):
            try:
                sig = strat.generate_signals(df)
                sig = sig.clip(-1, 1).fillna(0)
                
                strat_ret = sig.shift(1).fillna(0) * market_ret
                
                returns_list.append(strat_ret)
                valid_indices.append(i)
            except AttributeError as e:
                logger.error(f"Strategy {i} missing generate_signals method: {e}")
            except (KeyError, ValueError) as e:
                logger.warning(f"Strategy {i} optimization failed with data error: {e}")
            except Exception as e:
                logger.error(f"Strategy {i} optimization failed with unexpected error: {e}", exc_info=True)
        
        if not returns_list or len(returns_list) < 2:
            logger.warning("Not enough valid strategy returns for correlation optimization")
            return

        returns_df = pd.DataFrame(returns_list).T.fillna(0)
        corr_matrix = returns_df.corr().values
        
        def objective(weights: np.ndarray) -> float:
            port_ret = (returns_df * weights).sum(axis=1)
            
            mean_ret = port_ret.mean()
            std_ret = port_ret.std()
            
            if std_ret == 0:
                sharpe_term = 0.0
            else:
                sharpe_term = mean_ret / std_ret
            
            corr_term = 0.0
            for i in range(len(weights)):
                for j in range(i + 1, len(weights)):
                    corr_term += abs(corr_matrix[i, j]) * weights[i] * weights[j]
            
            return -(sharpe_term - correlation_penalty * corr_term)

        n = len(returns_list)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n))
        initial_weights = [1./n] * n
        
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                new_weights: list[float] = [0.0] * len(self.strategies)
                for idx, weight in zip(valid_indices, result.x, strict=False):
                    new_weights[idx] = max(0.0, float(weight))
                
                s = sum(new_weights)
                if s > 0:
                    self.weights = [w/s for w in new_weights]
                
                logger.info(f"Optimized Weights (with correlation penalty): {[f'{w:.2f}' for w in self.weights]}")
            else:
                logger.warning(f"Optimization failed: {result.message}")
        except ValueError as e:
            logger.error(f"Optimization value error: {e}")
        except RuntimeError as e:
            logger.error(f"Optimization runtime error: {e}")
        except Exception as e:
            logger.error(f"Unexpected optimization error: {e}", exc_info=True)
