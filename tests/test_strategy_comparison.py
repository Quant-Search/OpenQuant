"""Tests for strategy comparison report module."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from openquant.backtest.engine import backtest_signals, BacktestResult
from openquant.reporting.strategy_comparison import (
    calculate_metrics,
    calculate_drawdown_series,
    t_test_comparison,
    diebold_mariano_test,
    calculate_return_correlation,
    rank_strategies,
    compare_strategies,
    generate_comparison_report,
    export_comparison_to_csv,
)


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range(start='2022-01-01', periods=n, freq='1h')
    
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high_prices = close_prices + np.abs(np.random.randn(n) * 0.3)
    low_prices = close_prices - np.abs(np.random.randn(n) * 0.3)
    open_prices = close_prices + np.random.randn(n) * 0.2
    volume = np.random.uniform(1000, 10000, n)
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)
    
    return df


@pytest.fixture
def sample_backtest_result(sample_data):
    """Create a sample backtest result."""
    signals = pd.Series(1, index=sample_data.index)
    signals.iloc[:100] = 0
    
    result = backtest_signals(
        df=sample_data,
        signals=signals,
        fee_bps=1.0,
        weight=1.0
    )
    
    return result


@pytest.fixture
def multiple_backtest_results(sample_data):
    """Create multiple backtest results for comparison."""
    np.random.seed(42)
    
    results = {}
    
    signals_long = pd.Series(1, index=sample_data.index)
    signals_long.iloc[:50] = 0
    results["Long_Strategy"] = backtest_signals(
        df=sample_data,
        signals=signals_long,
        fee_bps=1.0,
        weight=1.0
    )
    
    signals_short = pd.Series(-1, index=sample_data.index)
    signals_short.iloc[:50] = 0
    results["Short_Strategy"] = backtest_signals(
        df=sample_data,
        signals=signals_short,
        fee_bps=1.0,
        weight=1.0
    )
    
    ma_fast = sample_data['Close'].rolling(10).mean()
    ma_slow = sample_data['Close'].rolling(30).mean()
    signals_ma = pd.Series(0, index=sample_data.index)
    signals_ma[ma_fast > ma_slow] = 1
    signals_ma[ma_fast < ma_slow] = -1
    results["MA_Crossover"] = backtest_signals(
        df=sample_data,
        signals=signals_ma,
        fee_bps=1.0,
        weight=1.0
    )
    
    return results


def test_calculate_metrics(sample_backtest_result):
    """Test metrics calculation."""
    metrics = calculate_metrics(sample_backtest_result, freq="1h")
    
    assert isinstance(metrics, dict)
    assert "Sharpe Ratio" in metrics
    assert "Sortino Ratio" in metrics
    assert "Max Drawdown (%)" in metrics
    assert "Total Return (%)" in metrics
    assert "Win Rate (%)" in metrics
    assert "Profit Factor" in metrics
    assert "CAGR (%)" in metrics
    assert "Calmar Ratio" in metrics
    
    assert isinstance(metrics["Sharpe Ratio"], (int, float))
    assert isinstance(metrics["Total Trades"], int)


def test_calculate_metrics_empty_result():
    """Test metrics calculation with empty result."""
    empty_result = BacktestResult(
        equity_curve=pd.Series([]),
        returns=pd.Series([]),
        positions=pd.Series([]),
        trades=pd.Series([])
    )
    
    metrics = calculate_metrics(empty_result, freq="1h")
    
    assert metrics["Total Return (%)"] == 0.0
    assert metrics["Sharpe Ratio"] == 0.0


def test_calculate_drawdown_series(sample_backtest_result):
    """Test drawdown calculation."""
    equity = sample_backtest_result.equity_curve
    drawdown = calculate_drawdown_series(equity)
    
    assert isinstance(drawdown, pd.Series)
    assert len(drawdown) == len(equity)
    assert (drawdown <= 0).all()


def test_t_test_comparison(multiple_backtest_results):
    """Test t-test comparison."""
    returns_a = multiple_backtest_results["Long_Strategy"].returns
    returns_b = multiple_backtest_results["Short_Strategy"].returns
    
    result = t_test_comparison(returns_a, returns_b)
    
    assert isinstance(result, dict)
    assert "t_statistic" in result
    assert "p_value" in result
    assert "mean_diff" in result
    assert "significant" in result
    assert isinstance(result["significant"], bool)


def test_t_test_comparison_insufficient_data():
    """Test t-test with insufficient data."""
    returns_a = pd.Series([0.01])
    returns_b = pd.Series([0.02])
    
    result = t_test_comparison(returns_a, returns_b)
    
    assert result["p_value"] == 1.0
    assert result["significant"] is False


def test_diebold_mariano_test(multiple_backtest_results):
    """Test Diebold-Mariano test."""
    returns_a = multiple_backtest_results["Long_Strategy"].returns
    returns_b = multiple_backtest_results["Short_Strategy"].returns
    
    result = diebold_mariano_test(returns_a, returns_b, h=1)
    
    assert isinstance(result, dict)
    assert "dm_statistic" in result
    assert "p_value" in result
    assert "significant" in result
    assert "better_strategy" in result
    assert result["better_strategy"] in ["strategy_a", "strategy_b", "none"]


def test_diebold_mariano_test_insufficient_data():
    """Test DM test with insufficient data."""
    returns_a = pd.Series([0.01])
    returns_b = pd.Series([0.02])
    
    result = diebold_mariano_test(returns_a, returns_b)
    
    assert result["p_value"] == 1.0
    assert result["significant"] is False


def test_calculate_return_correlation(multiple_backtest_results):
    """Test correlation calculation."""
    correlation = calculate_return_correlation(multiple_backtest_results)
    
    assert isinstance(correlation, pd.DataFrame)
    assert correlation.shape[0] == correlation.shape[1]
    assert correlation.shape[0] == len(multiple_backtest_results)
    
    assert (correlation.values.diagonal() == 1.0).all()
    
    assert (correlation >= -1).all().all()
    assert (correlation <= 1).all().all()


def test_rank_strategies_sharpe(multiple_backtest_results):
    """Test strategy ranking by Sharpe ratio."""
    metrics_dict = {}
    for name, result in multiple_backtest_results.items():
        metrics_dict[name] = calculate_metrics(result, freq="1h")
    
    metrics_table = pd.DataFrame(metrics_dict)
    
    rankings = rank_strategies(metrics_table, ranking_method="sharpe")
    
    assert isinstance(rankings, pd.DataFrame)
    assert "Rank" in rankings.columns
    assert "Score" in rankings.columns
    assert len(rankings) == len(multiple_backtest_results)
    
    assert rankings["Rank"].min() == 1
    assert rankings["Rank"].max() == len(multiple_backtest_results)


def test_rank_strategies_composite(multiple_backtest_results):
    """Test strategy ranking with composite method."""
    metrics_dict = {}
    for name, result in multiple_backtest_results.items():
        metrics_dict[name] = calculate_metrics(result, freq="1h")
    
    metrics_table = pd.DataFrame(metrics_dict)
    
    rankings = rank_strategies(metrics_table, ranking_method="composite")
    
    assert isinstance(rankings, pd.DataFrame)
    assert "Rank" in rankings.columns
    assert "Score" in rankings.columns
    assert (rankings["Score"] >= 0).all()


def test_compare_strategies(multiple_backtest_results):
    """Test full strategy comparison."""
    comparison = compare_strategies(
        results=multiple_backtest_results,
        freq="1h",
        ranking_method="composite"
    )
    
    assert comparison.metrics_table is not None
    assert isinstance(comparison.metrics_table, pd.DataFrame)
    assert comparison.metrics_table.shape[1] == len(multiple_backtest_results)
    
    assert comparison.equity_curves is not None
    assert isinstance(comparison.equity_curves, pd.DataFrame)
    
    assert comparison.drawdown_curves is not None
    assert isinstance(comparison.drawdown_curves, pd.DataFrame)
    
    assert comparison.statistical_tests is not None
    assert isinstance(comparison.statistical_tests, dict)
    
    assert comparison.correlation_matrix is not None
    assert isinstance(comparison.correlation_matrix, pd.DataFrame)
    
    assert comparison.ranked_strategies is not None
    assert isinstance(comparison.ranked_strategies, pd.DataFrame)


def test_compare_strategies_empty():
    """Test comparison with empty results."""
    comparison = compare_strategies(
        results={},
        freq="1h",
        ranking_method="composite"
    )
    
    assert comparison.metrics_table.empty
    assert comparison.equity_curves.empty


def test_generate_comparison_report(multiple_backtest_results):
    """Test report generation."""
    comparison = compare_strategies(
        results=multiple_backtest_results,
        freq="1h",
        ranking_method="composite"
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_report.txt"
        
        report_text = generate_comparison_report(
            comparison=comparison,
            output_path=output_path,
            include_plots=False
        )
        
        assert isinstance(report_text, str)
        assert len(report_text) > 0
        assert "STRATEGY BACKTESTING COMPARISON REPORT" in report_text
        assert "STRATEGY RANKINGS" in report_text
        assert "PERFORMANCE METRICS" in report_text
        assert "STATISTICAL TESTS" in report_text
        
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            file_content = f.read()
        assert file_content == report_text


def test_export_comparison_to_csv(multiple_backtest_results):
    """Test CSV export."""
    comparison = compare_strategies(
        results=multiple_backtest_results,
        freq="1h",
        ranking_method="composite"
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        export_comparison_to_csv(
            comparison=comparison,
            output_dir=output_dir
        )
        
        assert (output_dir / "metrics_comparison.csv").exists()
        assert (output_dir / "equity_curves.csv").exists()
        assert (output_dir / "drawdown_curves.csv").exists()
        assert (output_dir / "correlation_matrix.csv").exists()
        assert (output_dir / "strategy_rankings.csv").exists()
        assert (output_dir / "statistical_tests.csv").exists()
        
        metrics_df = pd.read_csv(output_dir / "metrics_comparison.csv", index_col=0)
        assert not metrics_df.empty
        assert metrics_df.shape[1] == len(multiple_backtest_results)


def test_statistical_tests_pairwise(multiple_backtest_results):
    """Test that statistical tests are computed for all pairs."""
    comparison = compare_strategies(
        results=multiple_backtest_results,
        freq="1h",
        ranking_method="composite"
    )
    
    n_strategies = len(multiple_backtest_results)
    expected_pairs = n_strategies * (n_strategies - 1) // 2
    
    assert len(comparison.statistical_tests) == expected_pairs
    
    for test_name, tests in comparison.statistical_tests.items():
        assert "t_test" in tests
        assert "diebold_mariano" in tests
        assert "_vs_" in test_name


def test_metrics_table_structure(multiple_backtest_results):
    """Test that metrics table has expected structure."""
    comparison = compare_strategies(
        results=multiple_backtest_results,
        freq="1h",
        ranking_method="composite"
    )
    
    expected_metrics = [
        "Total Return (%)",
        "CAGR (%)",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown (%)",
        "CVaR 95% (%)",
        "Win Rate (%)",
        "Profit Factor",
        "Calmar Ratio",
        "Volatility (%)",
    ]
    
    for metric in expected_metrics:
        assert metric in comparison.metrics_table.index


def test_equity_curves_alignment(multiple_backtest_results):
    """Test that equity curves are properly aligned."""
    comparison = compare_strategies(
        results=multiple_backtest_results,
        freq="1h",
        ranking_method="composite"
    )
    
    assert comparison.equity_curves.shape[1] == len(multiple_backtest_results)
    
    for strategy_name in multiple_backtest_results.keys():
        assert strategy_name in comparison.equity_curves.columns


def test_drawdown_curves_negative(multiple_backtest_results):
    """Test that drawdown curves are non-positive."""
    comparison = compare_strategies(
        results=multiple_backtest_results,
        freq="1h",
        ranking_method="composite"
    )
    
    for col in comparison.drawdown_curves.columns:
        assert (comparison.drawdown_curves[col] <= 0).all()
