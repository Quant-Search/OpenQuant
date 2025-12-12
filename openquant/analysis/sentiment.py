"""Sentiment Analysis Module.

Fetches Fear & Greed Index and integrates with trading signals.
"""
import requests
import pandas as pd
from datetime import datetime
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)

class SentimentAnalyzer:
    """Analyzes market sentiment."""
    
    @staticmethod
    def get_crypto_fear_greed() -> dict:
        """Get Crypto Fear & Greed Index."""
        try:
            url = "https://api.alternative.me/fng/?limit=1"
            response = requests.get(url, timeout=5)
            data = response.json()
            value = int(data['data'][0]['value'])
            classification = data['data'][0]['value_classification']
            return {'value': value, 'class': classification}
        except Exception as e:
            LOGGER.error(f"Sentiment API error: {e}")
            return {'value': 50, 'class': 'Neutral'}
            
    @staticmethod
    def get_signal_modifier(sentiment_score: int) -> float:
        """
        Adjust signal confidence based on sentiment.
        
        Contrarian approach:
        - Extreme Fear (<20): Bullish modifier
        - Extreme Greed (>80): Bearish modifier
        """
        if sentiment_score < 20: return 1.2  # Buy the fear
        if sentiment_score > 80: return 0.8  # Caution on greed
        return 1.0
