"""Portfolio Optimizer with Markowitz Mean-Variance Optimization.

Implements:
- Markowitz mean-variance optimization
- Kelly Criterion overlay for position sizing
- Correlation matrix constraints (max pairwise correlation 0.7)
- Sector/asset class exposure limits
- CVaR (Conditional Value at Risk) constraints
- Integration with forex_correlation.py for FX pairs
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..risk.forex_correlation import get_correlation, FOREX_CORRELATIONS
from ..risk.adaptive_sizing import kelly_criterion
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class OptimizationConstraints:
    """Constraints for portfolio optimization."""
    max_position_weight: float = 0.25
    max_total_weight: float = 1.0
    max_pairwise_correlation: float = 0.7
    max_sector_weight: float = 0.40
    max_asset_class_weight: float = 0.50
    max_cvar: Optional[float] = None
    cvar_confidence: float = 0.95
    min_position_weight: float = 0.01
    allow_short: bool = False
    risk_free_rate: float = 0.03


@dataclass
class AssetMetadata:
    """Metadata for each asset."""
    symbol: str
    sector: Optional[str] = None
    asset_class: Optional[str] = None
    
    
class PortfolioOptimizer:
    """Portfolio optimizer using Markowitz mean-variance with Kelly overlay."""
    
    def __init__(
        self,
        constraints: Optional[OptimizationConstraints] = None,
        kelly_fraction: float = 0.5
    ):
        """Initialize portfolio optimizer.
        
        Args:
            constraints: Optimization constraints
            kelly_fraction: Fraction of Kelly criterion to use (0.5 = Half Kelly)
        """
        self.constraints = constraints or OptimizationConstraints()
        self.kelly_fraction = kelly_fraction
        
    def optimize_allocation(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        symbols: List[str],
        asset_metadata: Optional[List[AssetMetadata]] = None,
        current_weights: Optional[np.ndarray] = None,
        risk_aversion: float = 1.0
    ) -> Dict[str, Any]:
        """Optimize portfolio allocation using mean-variance optimization.
        
        Args:
            expected_returns: Array of expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            symbols: List of asset symbols
            asset_metadata: Optional metadata for sector/asset class constraints
            current_weights: Optional current portfolio weights for comparison
            risk_aversion: Risk aversion parameter (higher = more conservative)
            
        Returns:
            Dictionary containing:
                - weights: Optimized portfolio weights
                - expected_return: Portfolio expected return
                - expected_volatility: Portfolio volatility
                - sharpe_ratio: Portfolio Sharpe ratio
                - cvar: Conditional Value at Risk
                - turnover: Portfolio turnover (if current_weights provided)
                - kelly_adjusted_weights: Kelly-adjusted weights
        """
        n_assets = len(symbols)
        
        if len(expected_returns) != n_assets:
            raise ValueError(f"Expected returns length {len(expected_returns)} != symbols length {n_assets}")
        if covariance_matrix.shape != (n_assets, n_assets):
            raise ValueError(f"Covariance matrix shape {covariance_matrix.shape} != ({n_assets}, {n_assets})")
        
        if asset_metadata is None:
            asset_metadata = [AssetMetadata(symbol=s) for s in symbols]
        
        # Step 1: Check correlation constraints and filter assets
        valid_indices = self._filter_by_correlation(symbols, covariance_matrix)
        
        if len(valid_indices) == 0:
            LOGGER.warning("No assets passed correlation filter, returning equal weights")
            return self._equal_weight_fallback(symbols)
        
        # Filter to valid assets
        filtered_symbols = [symbols[i] for i in valid_indices]
        filtered_returns = expected_returns[valid_indices]
        filtered_cov = covariance_matrix[np.ix_(valid_indices, valid_indices)]
        filtered_metadata = [asset_metadata[i] for i in valid_indices]
        
        # Step 2: Perform mean-variance optimization
        try:
            optimal_weights = self._mean_variance_optimize(
                filtered_returns,
                filtered_cov,
                risk_aversion=risk_aversion
            )
        except Exception as e:
            LOGGER.error(f"Mean-variance optimization failed: {e}")
            return self._equal_weight_fallback(symbols)
        
        # Step 3: Apply sector/asset class constraints
        optimal_weights = self._apply_sector_constraints(
            optimal_weights,
            filtered_metadata
        )
        
        # Step 4: Apply CVaR constraints
        if self.constraints.max_cvar is not None:
            optimal_weights = self._apply_cvar_constraint(
                optimal_weights,
                filtered_returns,
                filtered_cov
            )
        
        # Step 5: Normalize and apply min/max position constraints
        optimal_weights = self._normalize_weights(optimal_weights)
        
        # Step 6: Calculate Kelly-adjusted weights
        kelly_weights = self._apply_kelly_overlay(
            optimal_weights,
            filtered_returns,
            filtered_cov
        )
        
        # Map back to full symbol list with zeros for filtered assets
        full_weights = np.zeros(n_assets)
        full_kelly_weights = np.zeros(n_assets)
        for i, idx in enumerate(valid_indices):
            full_weights[idx] = optimal_weights[i]
            full_kelly_weights[idx] = kelly_weights[i]
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(full_weights, expected_returns)
        portfolio_variance = full_weights @ covariance_matrix @ full_weights
        portfolio_volatility = np.sqrt(max(portfolio_variance, 0))
        
        sharpe = (portfolio_return - self.constraints.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        # Calculate CVaR
        cvar = self._calculate_cvar(full_weights, expected_returns, covariance_matrix)
        
        # Calculate turnover if current weights provided
        turnover = None
        if current_weights is not None:
            if len(current_weights) == n_assets:
                turnover = np.sum(np.abs(full_weights - current_weights))
        
        # Create weight dictionary
        weights_dict = {symbols[i]: full_weights[i] for i in range(n_assets) if full_weights[i] > 1e-6}
        kelly_weights_dict = {symbols[i]: full_kelly_weights[i] for i in range(n_assets) if full_kelly_weights[i] > 1e-6}
        
        return {
            'weights': weights_dict,
            'weights_array': full_weights,
            'kelly_adjusted_weights': kelly_weights_dict,
            'kelly_weights_array': full_kelly_weights,
            'expected_return': float(portfolio_return),
            'expected_volatility': float(portfolio_volatility),
            'sharpe_ratio': float(sharpe),
            'cvar': float(cvar),
            'turnover': float(turnover) if turnover is not None else None,
            'n_positions': int(np.sum(full_weights > 1e-6)),
            'max_weight': float(np.max(full_weights))
        }
    
    def _filter_by_correlation(
        self,
        symbols: List[str],
        covariance_matrix: np.ndarray
    ) -> List[int]:
        """Filter assets based on maximum pairwise correlation constraint.
        
        Returns list of valid asset indices.
        """
        n = len(symbols)
        correlation_matrix = self._cov_to_corr(covariance_matrix)
        
        # Use greedy algorithm: keep assets in order, reject if correlated with kept assets
        valid_indices = []
        
        for i in range(n):
            is_valid = True
            
            for j in valid_indices:
                # Check computed correlation from returns
                corr_from_cov = abs(correlation_matrix[i, j])
                
                # For FX pairs, also check static correlation matrix
                corr_from_fx = abs(get_correlation(symbols[i], symbols[j]))
                
                # Use the more conservative (higher) correlation
                max_corr = max(corr_from_cov, corr_from_fx)
                
                if max_corr > self.constraints.max_pairwise_correlation:
                    LOGGER.debug(f"Filtering {symbols[i]} due to correlation {max_corr:.3f} with {symbols[j]}")
                    is_valid = False
                    break
            
            if is_valid:
                valid_indices.append(i)
        
        return valid_indices
    
    def _mean_variance_optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        risk_aversion: float = 1.0
    ) -> np.ndarray:
        """Perform mean-variance optimization using quadratic programming.
        
        Solves: min_w [λ * w'Σw - μ'w]
        subject to: sum(w) = 1, 0 <= w <= max_weight
        """
        from scipy.optimize import minimize
        
        n = len(expected_returns)
        
        # Objective function: minimize risk minus return
        def objective(w):
            portfolio_variance = w @ covariance_matrix @ w
            portfolio_return = np.dot(expected_returns, w)
            return risk_aversion * portfolio_variance - portfolio_return
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # weights sum to 1
        ]
        
        # Bounds for each weight
        if self.constraints.allow_short:
            bounds = [(-self.constraints.max_position_weight, self.constraints.max_position_weight) for _ in range(n)]
        else:
            bounds = [(0, self.constraints.max_position_weight) for _ in range(n)]
        
        # Initial guess: equal weights
        w0 = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            LOGGER.warning(f"Optimization did not converge: {result.message}")
        
        return result.x
    
    def _apply_sector_constraints(
        self,
        weights: np.ndarray,
        metadata: List[AssetMetadata]
    ) -> np.ndarray:
        """Apply sector and asset class exposure limits."""
        # Build sector and asset class mappings
        sector_weights = {}
        asset_class_weights = {}
        
        for i, meta in enumerate(metadata):
            if meta.sector:
                sector_weights.setdefault(meta.sector, []).append((i, weights[i]))
            if meta.asset_class:
                asset_class_weights.setdefault(meta.asset_class, []).append((i, weights[i]))
        
        # Check and scale down if limits exceeded
        modified = weights.copy()
        
        # Apply sector limits
        for sector, positions in sector_weights.items():
            total_weight = sum(w for _, w in positions)
            if total_weight > self.constraints.max_sector_weight:
                scale_factor = self.constraints.max_sector_weight / total_weight
                for idx, _ in positions:
                    modified[idx] *= scale_factor
                LOGGER.debug(f"Scaled sector {sector} by {scale_factor:.3f}")
        
        # Apply asset class limits
        for asset_class, positions in asset_class_weights.items():
            total_weight = sum(w for _, w in positions)
            if total_weight > self.constraints.max_asset_class_weight:
                scale_factor = self.constraints.max_asset_class_weight / total_weight
                for idx, _ in positions:
                    modified[idx] *= scale_factor
                LOGGER.debug(f"Scaled asset class {asset_class} by {scale_factor:.3f}")
        
        return modified
    
    def _apply_cvar_constraint(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """Apply CVaR constraint by scaling down portfolio if needed."""
        cvar = self._calculate_cvar(weights, expected_returns, covariance_matrix)
        
        if self.constraints.max_cvar is not None and abs(cvar) > abs(self.constraints.max_cvar):
            # Scale down to meet CVaR constraint
            scale_factor = abs(self.constraints.max_cvar) / abs(cvar)
            weights = weights * scale_factor
            LOGGER.debug(f"Scaled portfolio by {scale_factor:.3f} to meet CVaR constraint")
        
        return weights
    
    def _apply_kelly_overlay(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray
    ) -> np.ndarray:
        """Apply Kelly criterion overlay to adjust position sizes.
        
        Kelly fraction for each asset based on expected return and variance.
        """
        kelly_weights = np.zeros_like(weights)
        
        for i in range(len(weights)):
            if weights[i] < 1e-6:
                continue
            
            # Kelly formula: f* = μ / σ²
            # For portfolio context: scale by expected return / variance
            asset_return = expected_returns[i]
            asset_variance = covariance_matrix[i, i]
            
            if asset_variance > 0 and asset_return > 0:
                kelly_f = (asset_return - self.constraints.risk_free_rate) / asset_variance
                kelly_f = max(0, kelly_f * self.kelly_fraction)  # Apply fractional Kelly
                kelly_f = min(kelly_f, self.constraints.max_position_weight)  # Cap at max weight
                
                # Blend with optimized weight
                kelly_weights[i] = 0.7 * weights[i] + 0.3 * kelly_f
            else:
                kelly_weights[i] = weights[i]
        
        # Renormalize
        kelly_sum = np.sum(kelly_weights)
        if kelly_sum > 0:
            kelly_weights = kelly_weights / kelly_sum
        
        return kelly_weights
    
    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """Normalize weights to sum to 1 and apply min/max constraints."""
        # Apply minimum weight threshold
        weights[weights < self.constraints.min_position_weight] = 0
        
        # Renormalize
        weight_sum = np.sum(weights)
        if weight_sum > 0:
            weights = weights / weight_sum
        
        # Ensure total weight doesn't exceed max
        if np.sum(weights) > self.constraints.max_total_weight:
            weights = weights * (self.constraints.max_total_weight / np.sum(weights))
        
        return weights
    
    def _calculate_cvar(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        n_scenarios: int = 10000
    ) -> float:
        """Calculate parametric CVaR assuming normal distribution.
        
        CVaR (Conditional Value at Risk) = Expected loss given that loss exceeds VaR.
        """
        from scipy import stats
        
        # Portfolio statistics
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_variance = weights @ covariance_matrix @ weights
        portfolio_std = np.sqrt(max(portfolio_variance, 0))
        
        if portfolio_std == 0:
            return 0.0
        
        # Calculate VaR at confidence level
        alpha = self.constraints.cvar_confidence
        z_alpha = stats.norm.ppf(1 - alpha)
        var = portfolio_return + portfolio_std * z_alpha
        
        # Parametric CVaR for normal distribution
        # CVaR = μ + σ * φ(z_α) / (1 - α)
        # where φ is the standard normal PDF
        phi_z = stats.norm.pdf(z_alpha)
        cvar = portfolio_return + portfolio_std * (phi_z / (1 - alpha))
        
        return cvar
    
    def _cov_to_corr(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix."""
        std_dev = np.sqrt(np.diag(covariance_matrix))
        std_dev[std_dev == 0] = 1.0  # Avoid division by zero
        correlation_matrix = covariance_matrix / np.outer(std_dev, std_dev)
        return correlation_matrix
    
    def _equal_weight_fallback(self, symbols: List[str]) -> Dict[str, Any]:
        """Return equal-weighted portfolio as fallback."""
        n = len(symbols)
        weights = np.ones(n) / n
        
        return {
            'weights': {s: 1.0/n for s in symbols},
            'weights_array': weights,
            'kelly_adjusted_weights': {s: 1.0/n for s in symbols},
            'kelly_weights_array': weights,
            'expected_return': 0.0,
            'expected_volatility': 0.0,
            'sharpe_ratio': 0.0,
            'cvar': 0.0,
            'turnover': None,
            'n_positions': n,
            'max_weight': 1.0/n
        }
    
    def calculate_efficient_frontier(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        symbols: List[str],
        n_points: int = 20,
        asset_metadata: Optional[List[AssetMetadata]] = None
    ) -> List[Dict[str, Any]]:
        """Calculate efficient frontier by varying risk aversion.
        
        Args:
            expected_returns: Array of expected returns
            covariance_matrix: Covariance matrix
            symbols: List of symbols
            n_points: Number of points on frontier
            asset_metadata: Optional asset metadata
            
        Returns:
            List of portfolio allocations along efficient frontier
        """
        frontier = []
        
        # Range of risk aversion parameters
        risk_aversions = np.logspace(-1, 2, n_points)
        
        for risk_aversion in risk_aversions:
            try:
                result = self.optimize_allocation(
                    expected_returns=expected_returns,
                    covariance_matrix=covariance_matrix,
                    symbols=symbols,
                    asset_metadata=asset_metadata,
                    risk_aversion=risk_aversion
                )
                result['risk_aversion'] = float(risk_aversion)
                frontier.append(result)
            except Exception as e:
                LOGGER.warning(f"Failed to optimize for risk_aversion={risk_aversion}: {e}")
        
        return frontier
    
    def optimize_from_research_rows(
        self,
        rows: List[Dict[str, Any]],
        current_weights: Optional[Dict[str, float]] = None,
        risk_aversion: float = 1.0
    ) -> Dict[str, Any]:
        """Optimize portfolio from research output rows.
        
        Args:
            rows: List of research result rows with metrics
            current_weights: Optional current portfolio weights
            risk_aversion: Risk aversion parameter
            
        Returns:
            Optimization result dictionary
        """
        # Extract symbols and metrics
        symbols = []
        returns_list = []
        sharpe_list = []
        sectors = []
        asset_classes = []
        
        for row in rows:
            metrics = row.get('metrics', {})
            if not metrics.get('ok'):
                continue
            
            symbol = row.get('symbol')
            if not symbol:
                continue
            
            # Use Sharpe ratio as proxy for expected return
            sharpe = metrics.get('sharpe', 0.0) or 0.0
            
            # Use returns series if available for covariance
            returns = metrics.get('returns')
            
            symbols.append(symbol)
            sharpe_list.append(float(sharpe))
            returns_list.append(returns)
            
            # Extract sector/asset class if available
            sector = row.get('sector')
            asset_class = row.get('asset_class')
            sectors.append(sector)
            asset_classes.append(asset_class)
        
        if not symbols:
            LOGGER.warning("No valid assets to optimize")
            return self._equal_weight_fallback([])
        
        # Build expected returns (use Sharpe as proxy)
        expected_returns = np.array(sharpe_list)
        
        # Build covariance matrix
        if all(isinstance(r, pd.Series) and not r.empty for r in returns_list):
            # Use actual returns to compute covariance
            returns_df = pd.DataFrame({s: r for s, r in zip(symbols, returns_list)})
            returns_df = returns_df.fillna(0)
            covariance_matrix = returns_df.cov().values
        else:
            # Fallback: diagonal covariance based on max_dd
            variances = []
            for row in rows:
                metrics = row.get('metrics', {})
                max_dd = metrics.get('max_dd', 0.1) or 0.1
                variances.append(max(float(max_dd), 0.01))
            covariance_matrix = np.diag(variances)
        
        # Build asset metadata
        asset_metadata = [
            AssetMetadata(
                symbol=s,
                sector=sectors[i] if sectors[i] else None,
                asset_class=asset_classes[i] if asset_classes[i] else None
            )
            for i, s in enumerate(symbols)
        ]
        
        # Convert current weights to array
        current_weights_array = None
        if current_weights:
            current_weights_array = np.array([current_weights.get(s, 0.0) for s in symbols])
        
        # Optimize
        return self.optimize_allocation(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
            symbols=symbols,
            asset_metadata=asset_metadata,
            current_weights=current_weights_array,
            risk_aversion=risk_aversion
        )
