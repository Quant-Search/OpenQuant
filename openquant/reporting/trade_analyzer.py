"""Automated Trade Analysis System.

Analyzes losing trades to identify patterns and common features.
Uses clustering and attribution analysis to generate actionable insights.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json

from ..utils.logging import get_logger
from ..storage.trade_memory import TradeMemory

LOGGER = get_logger(__name__)

class TradeAnalyzer:
    """
    Analyzes closed trades to identify patterns in losing trades.
    """
    def __init__(self, trade_memory: Optional[TradeMemory] = None):
        self.trade_memory = trade_memory or TradeMemory()
        
    def analyze_losing_trades(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analyze losing trades to find common patterns.
        
        Args:
            lookback_days: Number of days to analyze.
            
        Returns:
            Analysis report with clusters and insights.
        """
        try:
            # Load losing trades
            df = self.trade_memory.get_losing_trades(limit=1000)
            
            if df.empty:
                return {
                    "status": "no_data",
                    "message": "No losing trades found"
                }
                
            LOGGER.info(f"Analyzing {len(df)} losing trades...")
            
            # Extract features from trades
            features_list = []
            valid_indices = []
            
            for idx, row in df.iterrows():
                try:
                    features_json = row.get('features_json')
                    if features_json and isinstance(features_json, str):
                        features = json.loads(features_json)
                        features_list.append(features)
                        valid_indices.append(idx)
                except Exception:
                    continue
                    
            if not features_list:
                return {
                    "status": "no_features",
                    "message": "No feature data available for analysis"
                }
                
            # Convert to DataFrame
            features_df = pd.DataFrame(features_list, index=valid_indices)
            
            # Basic statistics
            stats = {
                "total_losing_trades": len(df),
                "total_loss": float(df['pnl'].sum()),
                "avg_loss_pct": float(df['pnl_pct'].mean() * 100),
                "worst_loss_pct": float(df['pnl_pct'].min() * 100)
            }
            
            # Exit reason analysis
            if 'exit_reason' in df.columns:
                exit_reasons = df['exit_reason'].value_counts().to_dict()
                stats['exit_reasons'] = exit_reasons
                
            # Clustering analysis
            clusters = self._cluster_losing_trades(features_df, n_clusters=3)
            
            # Feature attribution
            attribution = self._analyze_feature_attribution(features_df, df.loc[valid_indices])
            
            report = {
                "status": "success",
                "statistics": stats,
                "clusters": clusters,
                "attribution": attribution,
                "recommendations": self._generate_recommendations(stats, clusters, attribution)
            }
            
            # Save report
            self._save_report(report)
            
            return report
            
        except Exception as e:
            LOGGER.error(f"Trade analysis failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def _cluster_losing_trades(self, features_df: pd.DataFrame, n_clusters: int = 3) -> Dict:
        """Cluster losing trades by features."""
        try:
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_df.fillna(0))
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(features_df)), random_state=42)
            labels = kmeans.fit_predict(features_scaled)
            
            # Analyze each cluster
            clusters_info = {}
            for i in range(n_clusters):
                mask = labels == i
                cluster_features = features_df[mask]
                
                if len(cluster_features) > 0:
                    clusters_info[f"cluster_{i}"] = {
                        "size": int(mask.sum()),
                        "mean_features": cluster_features.mean().to_dict()
                    }
                    
            return clusters_info
            
        except Exception as e:
            LOGGER.warning(f"Clustering failed: {e}")
            return {}
            
    def _analyze_feature_attribution(self, features_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
        """Analyze which features correlate with larger losses."""
        try:
            # Correlate features with loss magnitude
            losses = trades_df['pnl_pct'].abs()
            
            correlations = {}
            for col in features_df.columns:
                try:
                    corr = features_df[col].corr(losses)
                    if not np.isnan(corr):
                        correlations[col] = float(corr)
                except Exception:
                    continue
                    
            # Sort by absolute correlation
            sorted_corr = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
            
            return {
                "top_predictors": dict(list(sorted_corr.items())[:5]),
                "all_correlations": sorted_corr
            }
            
        except Exception as e:
            LOGGER.warning(f"Attribution analysis failed: {e}")
            return {}
            
    def _generate_recommendations(
        self,
        stats: Dict,
        clusters: Dict,
        attribution: Dict
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Check exit reasons
        if 'exit_reasons' in stats:
            sl_ratio = stats['exit_reasons'].get('sl/tp', 0) / max(stats['total_losing_trades'], 1)
            if sl_ratio > 0.7:
                recommendations.append("70%+ of losses are from SL hits. Consider widening stops or improving entry timing.")
                
        # Check feature correlations
        if attribution and 'top_predictors' in attribution:
            top_features = attribution['top_predictors']
            for feature, corr in list(top_features.items())[:2]:
                if abs(corr) > 0.3:
                    recommendations.append(f"Feature '{feature}' strongly correlated with losses (r={corr:.2f}). Review this indicator.")
                    
        # Check cluster distribution
        if clusters:
            sizes = [c['size'] for c in clusters.values()]
            if max(sizes) > sum(sizes) * 0.6:
                recommendations.append("Over 60% of losses fall into one cluster. Specific market conditions causing issues.")
                
        if not recommendations:
            recommendations.append("No clear patterns detected. Losses appear random or well-distributed.")
            
        return recommendations
        
    def _save_report(self, report: Dict):
        """Save analysis report to file."""
        try:
            path = Path("data/trade_analysis_report.json")
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w') as f:
                json.dump(report, f, indent=2)
                
            LOGGER.info(f"Trade analysis report saved to {path}")
            
        except Exception as e:
            LOGGER.error(f"Failed to save report: {e}")
