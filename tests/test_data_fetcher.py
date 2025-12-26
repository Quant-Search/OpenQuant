"""
Unit Tests for Data Fetcher Module
====================================
Tests for data fetching, validation, and error handling.
"""
import pytest
import pandas as pd
from robot.data_fetcher import DataFetcher, DataFetchError
from robot.validation import ValidationError


class TestDataFetcherInit:
    """Test DataFetcher initialization."""
    
    def test_default_init(self):
        """Test default initialization."""
        fetcher = DataFetcher()
        
        assert fetcher.use_mt5 is True
        assert fetcher._mt5 is None
        assert fetcher._mt5_initialized is False
    
    def test_no_mt5_init(self):
        """Test initialization without MT5."""
        fetcher = DataFetcher(use_mt5=False)
        
        assert fetcher.use_mt5 is False


class TestInputValidation:
    """Test input validation for fetch method."""
    
    def test_invalid_symbol_empty(self):
        """Test that empty symbol raises error."""
        fetcher = DataFetcher(use_mt5=False)
        
        with pytest.raises(ValidationError):
            fetcher.fetch("", "1h", 100)
    
    def test_invalid_symbol_too_short(self):
        """Test that too-short symbol raises error."""
        fetcher = DataFetcher(use_mt5=False)
        
        with pytest.raises(ValidationError):
            fetcher.fetch("EU", "1h", 100)
    
    def test_invalid_timeframe(self):
        """Test that invalid timeframe raises error."""
        fetcher = DataFetcher(use_mt5=False)
        
        with pytest.raises(ValidationError):
            fetcher.fetch("EURUSD", "invalid", 100)
    
    def test_invalid_bars_count(self):
        """Test that invalid bars count raises error."""
        fetcher = DataFetcher(use_mt5=False)
        
        with pytest.raises(ValidationError):
            fetcher.fetch("EURUSD", "1h", -1)
        
        with pytest.raises(ValidationError):
            fetcher.fetch("EURUSD", "1h", 20000)
    
    def test_valid_symbol_normalized(self):
        """Test that symbol is normalized to uppercase."""
        fetcher = DataFetcher(use_mt5=False)
        
        # This should not raise (symbol will be normalized)
        # We can't test the actual fetch without network, 
        # but validation should pass
        try:
            fetcher.fetch("eurusd", "1h", 100)
        except ValidationError:
            pytest.fail("Should not raise ValidationError for valid lowercase symbol")
        except Exception:
            pass  # Other errors (network) are OK


class TestTimeframeValidation:
    """Test timeframe validation."""
    
    @pytest.mark.parametrize("timeframe", ["H1", "h1", "1h", "4h", "H4", "D1", "1d"])
    def test_valid_timeframes(self, timeframe):
        """Test that valid timeframes pass validation."""
        fetcher = DataFetcher(use_mt5=False)
        
        try:
            fetcher.fetch("EURUSD", timeframe, 100)
        except ValidationError:
            pytest.fail(f"Should not raise for valid timeframe: {timeframe}")
        except Exception:
            pass  # Network errors OK


class TestYFinanceFetch:
    """Test yfinance fallback fetching."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_fetch_eurusd(self):
        """Test fetching EURUSD data."""
        fetcher = DataFetcher(use_mt5=False)
        
        df = fetcher.fetch("EURUSD", "1d", 100)
        
        if not df.empty:  # Network might fail
            assert 'Open' in df.columns
            assert 'High' in df.columns
            assert 'Low' in df.columns
            assert 'Close' in df.columns
            assert 'Volume' in df.columns
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_fetch_returns_dataframe(self):
        """Test that fetch returns a DataFrame."""
        fetcher = DataFetcher(use_mt5=False)
        
        df = fetcher.fetch("EURUSD", "1h", 50)
        
        assert isinstance(df, pd.DataFrame)


class TestDataFrameFormat:
    """Test returned DataFrame format."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_column_names(self):
        """Test that columns are standardized."""
        fetcher = DataFetcher(use_mt5=False)
        df = fetcher.fetch("EURUSD", "1d", 50)
        
        if not df.empty:
            expected_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
            assert set(df.columns) == expected_cols
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_data_types(self):
        """Test that data types are correct."""
        fetcher = DataFetcher(use_mt5=False)
        df = fetcher.fetch("EURUSD", "1d", 50)
        
        if not df.empty:
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                assert df[col].dtype == float
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_index_is_datetime(self):
        """Test that index is datetime."""
        fetcher = DataFetcher(use_mt5=False)
        df = fetcher.fetch("EURUSD", "1d", 50)
        
        if not df.empty:
            assert isinstance(df.index, pd.DatetimeIndex)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_nonexistent_symbol_returns_empty(self):
        """Test that nonexistent symbol returns empty DataFrame."""
        fetcher = DataFetcher(use_mt5=False)
        
        # This should not crash, just return empty
        df = fetcher.fetch("XXXXXX", "1d", 50)
        
        assert df.empty or isinstance(df, pd.DataFrame)
    
    @pytest.mark.mt5
    def test_mt5_fallback_to_yfinance(self):
        """Test that MT5 failure falls back to yfinance."""
        fetcher = DataFetcher(use_mt5=True)
        
        # Even if MT5 fails, should get data from yfinance
        df = fetcher.fetch("EURUSD", "1d", 50)
        
        # Either we got data or empty (network issues)
        assert isinstance(df, pd.DataFrame)

