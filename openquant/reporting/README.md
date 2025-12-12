# OpenQuant Reporting Module

Comprehensive backtesting report generation with professional-grade visualizations and analysis.

## Features

### 📊 Core Metrics
- **Total Return & CAGR**: Annualized returns with compounding
- **Risk-Adjusted Returns**: Sharpe, Sortino, and Calmar ratios
- **Risk Metrics**: Maximum drawdown, CVaR (95%), volatility
- **Trade Statistics**: Win rate, profit factor, average trade metrics

### 📈 Visualizations

#### Equity Curve Analysis
- Cumulative equity curve with confidence bands
- Monte Carlo simulations (5th/50th/95th percentiles)
- Comparison with actual performance

#### Returns Analysis
- Monthly returns heatmap (color-coded)
- Yearly returns bar chart
- Return distribution histograms

#### Drawdown Analysis
- Drawdown timeline visualization
- Underwater chart (time spent in drawdown)
- Maximum drawdown periods identification

#### Trade Analysis
- **Top 10 Best Trades**: Highest P&L trades with details
- **Top 10 Worst Trades**: Largest losses with analysis
- Trade P&L distribution histogram
- Trade duration analysis
- Maximum Adverse/Favorable Excursion (MAE/MFE)

#### Regime Performance
- Performance breakdown by market regime
- Regime-specific Sharpe ratios
- Win rates and P&L by regime
- Maximum drawdown per regime

#### Stress Testing
- Market crash scenarios (10%, 20% drops)
- Volatility spike scenarios (2x vol)
- Extended drawdown simulations
- Recovery time analysis

#### Performance Attribution
- Long vs Short P&L breakdown
- Trade count distribution
- Win rate comparison
- Contribution to total returns

### 📄 Report Formats

#### PDF Reports
- Professional multi-page PDF with matplotlib
- Ready for client presentations
- Print-friendly layout
- Comprehensive metrics summary

#### HTML Reports
- Interactive Plotly visualizations
- Zoom, pan, and hover capabilities
- Modern responsive design
- Easy sharing via web

## Usage

### Basic Usage

```python
from openquant.backtest.engine import backtest_signals
from openquant.reporting import ProfitabilityReportGenerator

# Run backtest
result = backtest_signals(df, signals, fee_bps=1.0, weight=1.0)

# Generate report
generator = ProfitabilityReportGenerator(output_dir="reports")
report_files = generator.generate_report(
    result=result,
    df=df,
    strategy_name="My_Strategy",
    freq="1d",
    format="both"  # Generate both PDF and HTML
)
```

### Advanced Usage with Regime Detection

```python
import pandas as pd
from openquant.reporting import ProfitabilityReportGenerator

# Detect market regimes
volatility = df['Close'].pct_change().rolling(20).std()
regime_labels = pd.Series('NORMAL', index=df.index)
regime_labels[volatility > volatility.quantile(0.75)] = 'HIGH_VOL'
regime_labels[volatility < volatility.quantile(0.25)] = 'LOW_VOL'

# Generate report with regime analysis
generator = ProfitabilityReportGenerator()
report_files = generator.generate_report(
    result=result,
    df=df,
    strategy_name="Regime_Aware_Strategy",
    freq="1h",
    regime_labels=regime_labels,
    format="both",
    monte_carlo_runs=1000
)
```

### With Trade Details

```python
# If you have detailed trade logs
trade_details = pd.DataFrame({
    'ts': [timestamps],
    'side': ['BUY', 'SELL', ...],
    'price': [prices],
    'delta_units': [quantities]
})

report_files = generator.generate_report(
    result=result,
    df=df,
    strategy_name="Detailed_Strategy",
    freq="4h",
    trade_details=trade_details,
    format="pdf"
)
```

## Command-Line Usage

### Demo Report

```bash
python scripts/demo_report_generator.py
```

Generates a comprehensive demo report with synthetic data to showcase all features.

### Generate Report from Strategy

```bash
# Basic usage
python scripts/generate_backtest_report.py --symbol BTC/USD --strategy sma_cross

# With custom parameters
python scripts/generate_backtest_report.py \
    --symbol ETH/USD \
    --strategy stat_arb \
    --period 2y \
    --fee-bps 2.0 \
    --format pdf \
    --mc-runs 1000
```

#### Available Strategies
- `sma_cross`: Simple Moving Average crossover
- `rsi`: RSI mean reversion
- `bollinger`: Bollinger Bands breakout
- `stat_arb`: Statistical arbitrage

#### Command-Line Options
- `--symbol`: Trading symbol (e.g., BTC/USD, ETH/USD)
- `--strategy`: Strategy type
- `--period`: Historical data period (e.g., 1y, 2y, 6mo)
- `--format`: Report format (pdf, html, both)
- `--fee-bps`: Trading fee in basis points
- `--weight`: Position size fraction
- `--mc-runs`: Number of Monte Carlo simulations

## Report Sections

### 1. Summary Page
- Overall performance metrics
- Risk-adjusted returns
- Trade statistics
- Report metadata

### 2. Equity Curve Page
- Cumulative equity visualization
- Returns progression
- Performance over time

### 3. Returns Heatmap Page
- Monthly returns grid
- Yearly performance bars
- Easy identification of profitable periods

### 4. Drawdown Page
- Drawdown timeline
- Underwater periods chart
- Recovery analysis

### 5. Monte Carlo Page
- Confidence interval bands
- Sharpe ratio distribution
- Maximum drawdown distribution
- Risk assessment

### 6. Trade Analysis Page
- Best/worst trades lists
- P&L distribution
- Duration analysis
- MAE/MFE charts

### 7. Regime Analysis Page (if regime labels provided)
- Performance by market regime
- Sharpe comparison
- Win rates per regime
- Maximum drawdown analysis

### 8. Stress Test Page
- Market crash scenarios
- Volatility stress tests
- Recovery time estimates
- Risk-adjusted returns under stress

### 9. Attribution Page
- Long vs Short breakdown
- Trade count distribution
- Win rate comparison
- Contribution analysis

## Dependencies

### Required
- `pandas`: Data manipulation
- `numpy`: Numerical computations

### Optional (for full functionality)
- `matplotlib`: PDF report generation
- `plotly`: Interactive HTML reports
- `yfinance`: Market data download (for demo scripts)

## Output

Reports are saved to the `reports/` directory with timestamped filenames:

```
reports/
├── My_Strategy_20240115_143022.pdf
├── My_Strategy_20240115_143022.html
├── Regime_Aware_Strategy_20240115_144533.pdf
└── ...
```

## Performance

- **Fast Generation**: Reports generated in seconds
- **Memory Efficient**: Streaming PDF generation
- **Scalable**: Handles backtests with thousands of trades
- **Parallel Processing**: Monte Carlo simulations optimized

## Best Practices

1. **Use Regime Detection**: Provides valuable insights into strategy performance across market conditions
2. **Include Trade Details**: Enables detailed trade-by-trade analysis
3. **Run Adequate Monte Carlo Simulations**: Use at least 500 runs for robust confidence intervals
4. **Choose Appropriate Frequency**: Match `freq` parameter to your data frequency
5. **Save Reports**: Archive reports for future comparison and analysis

## Examples

See the `scripts/` directory for complete examples:
- `demo_report_generator.py`: Simple demo with synthetic data
- `generate_backtest_report.py`: Production-ready report generation

## Troubleshooting

### PDF Generation Issues
If PDF generation fails, ensure matplotlib is installed:
```bash
pip install matplotlib
```

### HTML Generation Issues
If HTML generation fails, ensure plotly is installed:
```bash
pip install plotly
```

### Memory Issues with Large Backtests
For very large backtests (>100k bars), consider:
- Reducing Monte Carlo runs
- Generating HTML-only reports (more memory efficient)
- Processing in batches

## Contributing

To add new report sections or visualizations:
1. Add visualization method to `ProfitabilityReportGenerator`
2. Update both PDF and HTML generation methods
3. Add documentation to this README
4. Include example in demo scripts

## License

Part of the OpenQuant project. See LICENSE file for details.
