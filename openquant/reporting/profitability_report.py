"""Comprehensive Backtesting Report Generator.

Generates detailed PDF/HTML reports with:
- Equity curves with confidence intervals
- Monthly/yearly returns heatmaps
- Drawdown timeline and underwater chart
- Regime-specific performance tables
- Top 10 best/worst trades
- Performance attribution breakdown
- Monte Carlo confidence intervals
- Stress test results
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import warnings

import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

from ..utils.logging import get_logger
from ..backtest.engine import BacktestResult
from ..backtest.metrics import (
    sharpe, sortino, max_drawdown, win_rate, profit_factor,
    cvar, monte_carlo_bootstrap
)

warnings.filterwarnings('ignore')

LOGGER = get_logger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.backends.backend_pdf import PdfPages
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    LOGGER.warning("matplotlib not available, PDF generation disabled")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    LOGGER.warning("plotly not available, interactive HTML reports disabled")


@dataclass
class TradeStats:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_abs: float
    duration_bars: int
    mae_pct: float
    mfe_pct: float
    side: str
    regime: Optional[str] = None


@dataclass
class RegimePerformance:
    regime_name: str
    total_trades: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl_pct: float
    sharpe: float
    max_dd: float
    profit_factor: float


@dataclass
class StressTestResult:
    scenario_name: str
    equity_change_pct: float
    max_dd_pct: float
    recovery_days: int
    sharpe_ratio: float


class ProfitabilityReportGenerator:
    """
    Generate comprehensive backtesting reports with visualizations and analysis.
    """
    
    def __init__(self, output_dir: Path = Path("reports")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_report(
        self,
        result: BacktestResult,
        df: pd.DataFrame,
        strategy_name: str = "Strategy",
        freq: str = "1d",
        trade_details: Optional[pd.DataFrame] = None,
        regime_labels: Optional[pd.Series] = None,
        format: str = "both",
        monte_carlo_runs: int = 500
    ) -> Dict[str, Path]:
        """
        Generate comprehensive backtest report.
        
        Args:
            result: BacktestResult object from backtest engine
            df: Original OHLCV DataFrame with dates
            strategy_name: Name of the strategy
            freq: Data frequency ('1d', '1h', '4h', etc.)
            trade_details: Optional detailed trade log
            regime_labels: Optional regime classification per timestamp
            format: Output format ('pdf', 'html', 'both')
            monte_carlo_runs: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with paths to generated report files
        """
        LOGGER.info(f"Generating {format.upper()} report for {strategy_name}")
        
        report_data = self._analyze_backtest(
            result, df, freq, trade_details, regime_labels, monte_carlo_runs
        )
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}
        
        if format in ("pdf", "both") and MATPLOTLIB_AVAILABLE:
            pdf_path = self.output_dir / f"{strategy_name}_{timestamp}.pdf"
            self._generate_pdf_report(report_data, strategy_name, pdf_path)
            report_files["pdf"] = pdf_path
            LOGGER.info(f"PDF report saved to {pdf_path}")
            
        if format in ("html", "both") and PLOTLY_AVAILABLE:
            html_path = self.output_dir / f"{strategy_name}_{timestamp}.html"
            self._generate_html_report(report_data, strategy_name, html_path)
            report_files["html"] = html_path
            LOGGER.info(f"HTML report saved to {html_path}")
            
        return report_files
    
    def _analyze_backtest(
        self,
        result: BacktestResult,
        df: pd.DataFrame,
        freq: str,
        trade_details: Optional[pd.DataFrame],
        regime_labels: Optional[pd.Series],
        monte_carlo_runs: int
    ) -> Dict[str, Any]:
        """Analyze backtest results and compute all metrics."""
        
        equity = result.equity_curve
        returns = result.returns
        positions = result.positions
        
        data = {
            "equity": equity,
            "returns": returns,
            "positions": positions,
            "dates": equity.index,
            "freq": freq
        }
        
        data["metrics"] = self._compute_metrics(returns, equity, freq)
        
        data["monthly_returns"] = self._compute_monthly_returns(returns)
        data["yearly_returns"] = self._compute_yearly_returns(returns)
        
        data["drawdown"] = self._compute_drawdown_series(equity)
        
        if trade_details is not None:
            data["trade_stats"] = self._analyze_trades(trade_details, df)
        else:
            data["trade_stats"] = self._extract_trades_from_positions(
                positions, df["Close"], equity.index
            )
        
        if regime_labels is not None:
            data["regime_performance"] = self._analyze_regime_performance(
                returns, equity, positions, regime_labels, freq
            )
        else:
            data["regime_performance"] = []
        
        data["monte_carlo"] = self._run_monte_carlo(returns, freq, monte_carlo_runs)
        
        data["stress_tests"] = self._run_stress_tests(equity, returns)
        
        data["attribution"] = self._compute_attribution(returns, positions)
        
        return data
    
    def _compute_metrics(
        self, returns: pd.Series, equity: pd.Series, freq: str
    ) -> Dict[str, float]:
        """Compute comprehensive performance metrics."""
        
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        cagr = self._compute_cagr(equity, freq)
        
        metrics = {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe": float(sharpe(returns, freq)),
            "sortino": float(sortino(returns, freq)),
            "max_drawdown": float(max_drawdown(equity)),
            "win_rate": float(win_rate(returns)),
            "profit_factor": float(profit_factor(returns)),
            "cvar_95": float(cvar(returns, alpha=0.95)),
            "calmar_ratio": abs(float(cagr / max_drawdown(equity))) if max_drawdown(equity) != 0 else 0.0,
            "total_trades": int((returns != 0).sum()),
            "avg_trade_return": float(returns[returns != 0].mean()) if (returns != 0).sum() > 0 else 0.0,
            "best_trade": float(returns.max()),
            "worst_trade": float(returns.min()),
            "avg_win": float(returns[returns > 0].mean()) if (returns > 0).sum() > 0 else 0.0,
            "avg_loss": float(returns[returns < 0].mean()) if (returns < 0).sum() > 0 else 0.0,
        }
        
        return metrics
    
    def _compute_cagr(self, equity: pd.Series, freq: str) -> float:
        """Compute Compound Annual Growth Rate."""
        if len(equity) < 2:
            return 0.0
        
        total_return = equity.iloc[-1] / equity.iloc[0]
        
        freq_map = {"1d": 252, "1h": 252 * 6.5, "4h": 252 * 1.625, "30m": 252 * 13}
        periods_per_year = freq_map.get(freq, 252)
        years = len(equity) / periods_per_year
        
        if years <= 0:
            return 0.0
        
        cagr = (total_return ** (1 / years)) - 1
        return cagr
    
    def _compute_monthly_returns(self, returns: pd.Series) -> pd.DataFrame:
        """Compute monthly returns matrix for heatmap."""
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)
        
        monthly = returns.groupby([returns.index.year, returns.index.month]).sum()
        
        if len(monthly) == 0:
            return pd.DataFrame()
        
        df = pd.DataFrame(monthly)
        df.columns = ['return']
        df.index.names = ['year', 'month']
        df = df.reset_index()
        
        pivot = df.pivot(index='year', columns='month', values='return')
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.columns = [month_names[int(m)-1] if m <= 12 else str(m) for m in pivot.columns]
        
        return pivot * 100
    
    def _compute_yearly_returns(self, returns: pd.Series) -> pd.Series:
        """Compute yearly returns."""
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index)
        
        yearly = returns.groupby(returns.index.year).apply(
            lambda x: (1 + x).prod() - 1
        )
        return yearly * 100
    
    def _compute_drawdown_series(self, equity: pd.Series) -> pd.DataFrame:
        """Compute drawdown series with underwater periods."""
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        
        underwater = (drawdown < 0).astype(int)
        underwater_periods = underwater.diff().fillna(0)
        
        start_indices = underwater_periods[underwater_periods == 1].index
        end_indices = underwater_periods[underwater_periods == -1].index
        
        dd_periods = []
        for i, start in enumerate(start_indices):
            if i < len(end_indices):
                end = end_indices[i]
            else:
                end = equity.index[-1]
            
            period_dd = drawdown.loc[start:end]
            max_dd = period_dd.min()
            duration = len(period_dd)
            
            dd_periods.append({
                "start": start,
                "end": end,
                "max_drawdown": max_dd,
                "duration": duration
            })
        
        return pd.DataFrame({
            "drawdown": drawdown,
            "underwater": underwater,
            "periods": dd_periods if dd_periods else None
        })
    
    def _extract_trades_from_positions(
        self, positions: pd.Series, prices: pd.Series, dates: pd.DatetimeIndex
    ) -> List[TradeStats]:
        """Extract individual trades from position series."""
        trades = []
        
        pos_changes = positions.diff().fillna(positions)
        
        in_trade = False
        entry_idx = None
        entry_price = None
        entry_side = None
        
        for i in range(len(positions)):
            pos = positions.iloc[i]
            pos_change = pos_changes.iloc[i]
            
            if not in_trade and pos != 0:
                in_trade = True
                entry_idx = i
                entry_price = prices.iloc[i]
                entry_side = "LONG" if pos > 0 else "SHORT"
                
            elif in_trade and pos == 0:
                exit_price = prices.iloc[i]
                
                if entry_side == "LONG":
                    pnl_pct = (exit_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price
                
                pnl_abs = pnl_pct * entry_price
                
                trade_prices = prices.iloc[entry_idx:i+1]
                if entry_side == "LONG":
                    mae_pct = (trade_prices.min() - entry_price) / entry_price
                    mfe_pct = (trade_prices.max() - entry_price) / entry_price
                else:
                    mae_pct = (entry_price - trade_prices.max()) / entry_price
                    mfe_pct = (entry_price - trade_prices.min()) / entry_price
                
                trades.append(TradeStats(
                    entry_time=dates[entry_idx],
                    exit_time=dates[i],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl_pct=pnl_pct,
                    pnl_abs=pnl_abs,
                    duration_bars=i - entry_idx,
                    mae_pct=mae_pct,
                    mfe_pct=mfe_pct,
                    side=entry_side
                ))
                
                in_trade = False
        
        return trades
    
    def _analyze_trades(
        self, trade_details: pd.DataFrame, df: pd.DataFrame
    ) -> List[TradeStats]:
        """Analyze trades from trade details DataFrame."""
        trades = []
        
        if "side" not in trade_details.columns or "price" not in trade_details.columns:
            return trades
        
        buys = trade_details[trade_details["side"] == "BUY"]
        sells = trade_details[trade_details["side"] == "SELL"]
        
        for i in range(min(len(buys), len(sells))):
            buy = buys.iloc[i]
            sell = sells.iloc[i]
            
            entry_price = buy["price"]
            exit_price = sell["price"]
            pnl_pct = (exit_price - entry_price) / entry_price
            
            trades.append(TradeStats(
                entry_time=buy["ts"],
                exit_time=sell["ts"],
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
                pnl_abs=pnl_pct * entry_price,
                duration_bars=0,
                mae_pct=0.0,
                mfe_pct=0.0,
                side="LONG"
            ))
        
        return trades
    
    def _analyze_regime_performance(
        self,
        returns: pd.Series,
        equity: pd.Series,
        positions: pd.Series,
        regime_labels: pd.Series,
        freq: str
    ) -> List[RegimePerformance]:
        """Analyze performance by market regime."""
        regime_perf = []
        
        regime_labels = regime_labels.reindex(returns.index).fillna("UNKNOWN")
        
        for regime in regime_labels.unique():
            mask = regime_labels == regime
            regime_returns = returns[mask]
            regime_equity = equity[mask]
            regime_positions = positions[mask]
            
            if len(regime_returns) < 2:
                continue
            
            total_trades = (regime_positions.diff().abs() > 0).sum()
            
            regime_perf.append(RegimePerformance(
                regime_name=str(regime),
                total_trades=int(total_trades),
                win_rate=float(win_rate(regime_returns)),
                avg_pnl_pct=float(regime_returns.mean() * 100),
                total_pnl_pct=float(regime_returns.sum() * 100),
                sharpe=float(sharpe(regime_returns, freq)),
                max_dd=float(max_drawdown(regime_equity)),
                profit_factor=float(profit_factor(regime_returns))
            ))
        
        return regime_perf
    
    def _run_monte_carlo(
        self, returns: pd.Series, freq: str, n_runs: int
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulations."""
        mc_results = monte_carlo_bootstrap(returns, n=n_runs, block=10, freq=freq)
        
        simulated_equity = []
        for i in range(min(n_runs, 100)):
            idx = np.random.randint(0, len(returns), len(returns))
            sim_returns = returns.iloc[idx].values
            sim_equity = (1 + pd.Series(sim_returns)).cumprod()
            simulated_equity.append(sim_equity)
        
        equity_array = np.array([eq.values for eq in simulated_equity])
        
        percentiles = {
            "p5": np.percentile(equity_array, 5, axis=0),
            "p50": np.percentile(equity_array, 50, axis=0),
            "p95": np.percentile(equity_array, 95, axis=0)
        }
        
        return {
            "sharpe_dist": mc_results["sharpe"],
            "sortino_dist": mc_results["sortino"],
            "max_dd_dist": mc_results["max_dd"],
            "equity_percentiles": percentiles,
            "n_runs": n_runs
        }
    
    def _run_stress_tests(
        self, equity: pd.Series, returns: pd.Series
    ) -> List[StressTestResult]:
        """Run stress test scenarios."""
        stress_tests = []
        
        scenarios = {
            "10% Market Crash": self._apply_crash_scenario(returns, -0.10),
            "20% Market Crash": self._apply_crash_scenario(returns, -0.20),
            "2x Volatility": self._apply_volatility_scenario(returns, 2.0),
            "Extended Drawdown": self._apply_extended_drawdown(returns, 0.15, 30),
        }
        
        for name, stressed_returns in scenarios.items():
            stressed_equity = (1 + stressed_returns).cumprod()
            equity_change = (stressed_equity.iloc[-1] - 1) * 100
            max_dd = max_drawdown(stressed_equity) * 100
            
            dd_series = (stressed_equity - stressed_equity.cummax()) / stressed_equity.cummax()
            recovery_days = self._compute_recovery_time(dd_series)
            
            sharpe_ratio = sharpe(stressed_returns, "1d")
            
            stress_tests.append(StressTestResult(
                scenario_name=name,
                equity_change_pct=equity_change,
                max_dd_pct=max_dd,
                recovery_days=recovery_days,
                sharpe_ratio=sharpe_ratio
            ))
        
        return stress_tests
    
    def _apply_crash_scenario(self, returns: pd.Series, crash_pct: float) -> pd.Series:
        """Apply market crash scenario."""
        stressed = returns.copy()
        crash_idx = len(returns) // 2
        stressed.iloc[crash_idx] = crash_pct
        return stressed
    
    def _apply_volatility_scenario(self, returns: pd.Series, vol_mult: float) -> pd.Series:
        """Apply increased volatility scenario."""
        mean_ret = returns.mean()
        stressed = mean_ret + (returns - mean_ret) * vol_mult
        return stressed
    
    def _apply_extended_drawdown(
        self, returns: pd.Series, dd_depth: float, dd_length: int
    ) -> pd.Series:
        """Apply extended drawdown scenario."""
        stressed = returns.copy()
        start_idx = len(returns) // 3
        dd_per_period = dd_depth / dd_length
        
        for i in range(min(dd_length, len(stressed) - start_idx)):
            stressed.iloc[start_idx + i] = -dd_per_period
        
        return stressed
    
    def _compute_recovery_time(self, drawdown_series: pd.Series) -> int:
        """Compute time to recovery from maximum drawdown."""
        max_dd_idx = drawdown_series.idxmin()
        
        if max_dd_idx == drawdown_series.index[-1]:
            return len(drawdown_series) - drawdown_series.index.get_loc(max_dd_idx)
        
        recovery_series = drawdown_series.loc[max_dd_idx:]
        recovery_idx = recovery_series[recovery_series >= 0].first_valid_index()
        
        if recovery_idx is None:
            return len(recovery_series)
        
        return recovery_series.index.get_loc(recovery_idx)
    
    def _compute_attribution(
        self, returns: pd.Series, positions: pd.Series
    ) -> Dict[str, float]:
        """Compute performance attribution."""
        
        long_returns = returns[positions > 0]
        short_returns = returns[positions < 0]
        
        total_return = returns.sum()
        
        long_contribution = long_returns.sum() / total_return if total_return != 0 else 0
        short_contribution = short_returns.sum() / total_return if total_return != 0 else 0
        
        attribution = {
            "long_pnl_pct": float(long_returns.sum() * 100),
            "short_pnl_pct": float(short_returns.sum() * 100),
            "long_contribution": float(long_contribution * 100),
            "short_contribution": float(short_contribution * 100),
            "long_trades": int((positions > 0).sum()),
            "short_trades": int((positions < 0).sum()),
            "long_win_rate": float(win_rate(long_returns)) if len(long_returns) > 0 else 0,
            "short_win_rate": float(win_rate(short_returns)) if len(short_returns) > 0 else 0,
        }
        
        return attribution
    
    def _generate_pdf_report(
        self, report_data: Dict[str, Any], strategy_name: str, output_path: Path
    ):
        """Generate PDF report with matplotlib."""
        
        with PdfPages(output_path) as pdf:
            self._create_summary_page(pdf, report_data, strategy_name)
            self._create_equity_curve_page(pdf, report_data)
            self._create_returns_heatmap_page(pdf, report_data)
            self._create_drawdown_page(pdf, report_data)
            self._create_monte_carlo_page(pdf, report_data)
            
            if report_data.get("trade_stats"):
                self._create_trade_analysis_page(pdf, report_data)
            
            if report_data.get("regime_performance"):
                self._create_regime_analysis_page(pdf, report_data)
            
            self._create_stress_test_page(pdf, report_data)
            self._create_attribution_page(pdf, report_data)
    
    def _create_summary_page(
        self, pdf: PdfPages, report_data: Dict, strategy_name: str
    ):
        """Create summary page with key metrics."""
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(f"{strategy_name} - Performance Summary", fontsize=16, fontweight='bold')
        
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        metrics = report_data["metrics"]
        
        summary_text = f"""
        OVERALL PERFORMANCE
        ══════════════════════════════════════════════════════════════
        Total Return:           {metrics['total_return']:>12.2%}
        CAGR:                   {metrics['cagr']:>12.2%}
        
        RISK-ADJUSTED RETURNS
        ══════════════════════════════════════════════════════════════
        Sharpe Ratio:           {metrics['sharpe']:>12.2f}
        Sortino Ratio:          {metrics['sortino']:>12.2f}
        Calmar Ratio:           {metrics['calmar_ratio']:>12.2f}
        
        RISK METRICS
        ══════════════════════════════════════════════════════════════
        Maximum Drawdown:       {metrics['max_drawdown']:>12.2%}
        CVaR (95%):             {metrics['cvar_95']:>12.2%}
        
        TRADE STATISTICS
        ══════════════════════════════════════════════════════════════
        Total Trades:           {metrics['total_trades']:>12}
        Win Rate:               {metrics['win_rate']:>12.2%}
        Profit Factor:          {metrics['profit_factor']:>12.2f}
        
        Average Trade Return:   {metrics['avg_trade_return']:>12.2%}
        Average Win:            {metrics['avg_win']:>12.2%}
        Average Loss:           {metrics['avg_loss']:>12.2%}
        
        Best Trade:             {metrics['best_trade']:>12.2%}
        Worst Trade:            {metrics['worst_trade']:>12.2%}
        
        REPORT GENERATED
        ══════════════════════════════════════════════════════════════
        Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
        
        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_equity_curve_page(self, pdf: PdfPages, report_data: Dict):
        """Create equity curve visualization."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle("Equity Curve Analysis", fontsize=14, fontweight='bold')
        
        equity = report_data["equity"]
        dates = report_data["dates"]
        
        ax1.plot(dates, equity.values, label='Equity Curve', linewidth=2, color='#2E86AB')
        ax1.fill_between(dates, 1, equity.values, alpha=0.3, color='#2E86AB')
        ax1.set_ylabel('Equity', fontsize=10)
        ax1.set_title('Cumulative Equity Curve', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        returns = report_data["returns"]
        cumulative_returns = (1 + returns).cumprod() - 1
        ax2.plot(dates, cumulative_returns.values * 100, linewidth=2, color='#A23B72')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Cumulative Return (%)', fontsize=10)
        ax2.set_title('Cumulative Returns', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_returns_heatmap_page(self, pdf: PdfPages, report_data: Dict):
        """Create monthly/yearly returns heatmap."""
        monthly_returns = report_data["monthly_returns"]
        yearly_returns = report_data["yearly_returns"]
        
        if monthly_returns.empty:
            return
        
        fig = plt.figure(figsize=(11, 8.5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        
        ax1 = plt.subplot(gs[0])
        im = ax1.imshow(monthly_returns.values, cmap='RdYlGn', aspect='auto',
                        vmin=-10, vmax=10)
        ax1.set_xticks(range(len(monthly_returns.columns)))
        ax1.set_xticklabels(monthly_returns.columns)
        ax1.set_yticks(range(len(monthly_returns.index)))
        ax1.set_yticklabels(monthly_returns.index)
        ax1.set_title('Monthly Returns Heatmap (%)', fontsize=12, fontweight='bold')
        
        for i in range(len(monthly_returns.index)):
            for j in range(len(monthly_returns.columns)):
                val = monthly_returns.iloc[i, j]
                if not np.isnan(val):
                    text_color = 'white' if abs(val) > 5 else 'black'
                    ax1.text(j, i, f'{val:.1f}', ha='center', va='center',
                            color=text_color, fontsize=8)
        
        plt.colorbar(im, ax=ax1, label='Return (%)')
        
        ax2 = plt.subplot(gs[1])
        years = yearly_returns.index
        colors = ['green' if r > 0 else 'red' for r in yearly_returns.values]
        ax2.bar(years, yearly_returns.values, color=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_xlabel('Year', fontsize=10)
        ax2.set_ylabel('Return (%)', fontsize=10)
        ax2.set_title('Yearly Returns', fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_drawdown_page(self, pdf: PdfPages, report_data: Dict):
        """Create drawdown timeline and underwater chart."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle("Drawdown Analysis", fontsize=14, fontweight='bold')
        
        drawdown_df = report_data["drawdown"]
        dates = report_data["dates"]
        
        ax1.fill_between(dates, 0, drawdown_df["drawdown"].values * 100,
                         color='red', alpha=0.5, label='Drawdown')
        ax1.set_ylabel('Drawdown (%)', fontsize=10)
        ax1.set_title('Drawdown Timeline', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        underwater = drawdown_df["underwater"].values
        ax2.fill_between(dates, 0, underwater, color='blue', alpha=0.3, label='Underwater')
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Underwater (1=Yes, 0=No)', fontsize=10)
        ax2.set_title('Underwater Periods', fontsize=11)
        ax2.set_ylim([-0.1, 1.1])
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_monte_carlo_page(self, pdf: PdfPages, report_data: Dict):
        """Create Monte Carlo simulation results."""
        mc_data = report_data["monte_carlo"]
        
        fig = plt.figure(figsize=(11, 8.5))
        gs = gridspec.GridSpec(2, 2)
        fig.suptitle(f"Monte Carlo Analysis ({mc_data['n_runs']} simulations)",
                    fontsize=14, fontweight='bold')
        
        ax1 = plt.subplot(gs[0, :])
        dates = report_data["dates"]
        equity = report_data["equity"]
        
        percentiles = mc_data["equity_percentiles"]
        ax1.fill_between(range(len(percentiles["p5"])),
                         percentiles["p5"], percentiles["p95"],
                         alpha=0.3, color='gray', label='5th-95th Percentile')
        ax1.plot(percentiles["p50"], color='blue', linewidth=2,
                label='50th Percentile (Median)', linestyle='--')
        ax1.plot(equity.values, color='green', linewidth=2, label='Actual Equity')
        ax1.set_ylabel('Equity', fontsize=10)
        ax1.set_title('Monte Carlo Equity Paths', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2 = plt.subplot(gs[1, 0])
        ax2.hist(mc_data["sharpe_dist"], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax2.axvline(report_data["metrics"]["sharpe"], color='red', linestyle='--',
                   linewidth=2, label='Actual')
        ax2.set_xlabel('Sharpe Ratio', fontsize=9)
        ax2.set_ylabel('Frequency', fontsize=9)
        ax2.set_title('Sharpe Distribution', fontsize=10)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(gs[1, 1])
        ax3.hist([abs(dd) * 100 for dd in mc_data["max_dd_dist"]], bins=30,
                color='salmon', edgecolor='black', alpha=0.7)
        ax3.axvline(abs(report_data["metrics"]["max_drawdown"]) * 100, color='red',
                   linestyle='--', linewidth=2, label='Actual')
        ax3.set_xlabel('Max Drawdown (%)', fontsize=9)
        ax3.set_ylabel('Frequency', fontsize=9)
        ax3.set_title('Max Drawdown Distribution', fontsize=10)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_trade_analysis_page(self, pdf: PdfPages, report_data: Dict):
        """Create trade analysis with top 10 best/worst trades."""
        trade_stats = report_data["trade_stats"]
        
        if not trade_stats:
            return
        
        trades_df = pd.DataFrame([asdict(t) for t in trade_stats])
        trades_df = trades_df.sort_values("pnl_pct", ascending=False)
        
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Trade Analysis", fontsize=14, fontweight='bold')
        
        gs = gridspec.GridSpec(3, 2)
        
        ax1 = plt.subplot(gs[0, :])
        ax1.axis('off')
        
        top_10 = trades_df.head(10)
        worst_10 = trades_df.tail(10)
        
        top_text = "TOP 10 BEST TRADES\n" + "=" * 80 + "\n"
        for idx, (i, row) in enumerate(top_10.iterrows(), 1):
            top_text += f"{idx:2d}. {row['entry_time']:%Y-%m-%d} {row['side']:5s} "
            top_text += f"P&L: {row['pnl_pct']:>7.2%}  Entry: ${row['entry_price']:.2f}  "
            top_text += f"Exit: ${row['exit_price']:.2f}\n"
        
        worst_text = "\n\nTOP 10 WORST TRADES\n" + "=" * 80 + "\n"
        for idx, (i, row) in enumerate(worst_10.iterrows(), 1):
            worst_text += f"{idx:2d}. {row['entry_time']:%Y-%m-%d} {row['side']:5s} "
            worst_text += f"P&L: {row['pnl_pct']:>7.2%}  Entry: ${row['entry_price']:.2f}  "
            worst_text += f"Exit: ${row['exit_price']:.2f}\n"
        
        ax1.text(0.05, 0.95, top_text + worst_text, transform=ax1.transAxes,
                fontsize=8, verticalalignment='top', family='monospace')
        
        ax2 = plt.subplot(gs[1, 0])
        pnl_values = trades_df["pnl_pct"] * 100
        ax2.hist(pnl_values, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(0, color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('P&L (%)', fontsize=9)
        ax2.set_ylabel('Frequency', fontsize=9)
        ax2.set_title('Trade P&L Distribution', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(gs[1, 1])
        durations = trades_df["duration_bars"]
        ax3.hist(durations, bins=30, color='orange', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Duration (bars)', fontsize=9)
        ax3.set_ylabel('Frequency', fontsize=9)
        ax3.set_title('Trade Duration Distribution', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(gs[2, 0])
        mae_values = trades_df["mae_pct"] * 100
        ax4.hist(mae_values, bins=30, color='crimson', edgecolor='black', alpha=0.7)
        ax4.set_xlabel('MAE (%)', fontsize=9)
        ax4.set_ylabel('Frequency', fontsize=9)
        ax4.set_title('Maximum Adverse Excursion', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        ax5 = plt.subplot(gs[2, 1])
        mfe_values = trades_df["mfe_pct"] * 100
        ax5.hist(mfe_values, bins=30, color='forestgreen', edgecolor='black', alpha=0.7)
        ax5.set_xlabel('MFE (%)', fontsize=9)
        ax5.set_ylabel('Frequency', fontsize=9)
        ax5.set_title('Maximum Favorable Excursion', fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_regime_analysis_page(self, pdf: PdfPages, report_data: Dict):
        """Create regime-specific performance analysis."""
        regime_perf = report_data["regime_performance"]
        
        if not regime_perf:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Regime-Specific Performance", fontsize=14, fontweight='bold')
        
        regimes = [r.regime_name for r in regime_perf]
        
        sharpes = [r.sharpe for r in regime_perf]
        axes[0, 0].bar(regimes, sharpes, color='steelblue', alpha=0.7)
        axes[0, 0].set_ylabel('Sharpe Ratio', fontsize=9)
        axes[0, 0].set_title('Sharpe by Regime', fontsize=10)
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        win_rates = [r.win_rate * 100 for r in regime_perf]
        axes[0, 1].bar(regimes, win_rates, color='forestgreen', alpha=0.7)
        axes[0, 1].set_ylabel('Win Rate (%)', fontsize=9)
        axes[0, 1].set_title('Win Rate by Regime', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        total_pnls = [r.total_pnl_pct for r in regime_perf]
        colors = ['green' if p > 0 else 'red' for p in total_pnls]
        axes[1, 0].bar(regimes, total_pnls, color=colors, alpha=0.7)
        axes[1, 0].set_ylabel('Total P&L (%)', fontsize=9)
        axes[1, 0].set_title('Total P&L by Regime', fontsize=10)
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        max_dds = [abs(r.max_dd) * 100 for r in regime_perf]
        axes[1, 1].bar(regimes, max_dds, color='crimson', alpha=0.7)
        axes[1, 1].set_ylabel('Max Drawdown (%)', fontsize=9)
        axes[1, 1].set_title('Max Drawdown by Regime', fontsize=10)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_stress_test_page(self, pdf: PdfPages, report_data: Dict):
        """Create stress test results visualization."""
        stress_tests = report_data["stress_tests"]
        
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Stress Test Results", fontsize=14, fontweight='bold')
        
        scenarios = [st.scenario_name for st in stress_tests]
        
        equity_changes = [st.equity_change_pct for st in stress_tests]
        colors = ['green' if e > 0 else 'red' for e in equity_changes]
        axes[0, 0].barh(scenarios, equity_changes, color=colors, alpha=0.7)
        axes[0, 0].set_xlabel('Equity Change (%)', fontsize=9)
        axes[0, 0].set_title('Equity Impact', fontsize=10)
        axes[0, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        max_dds = [st.max_dd_pct for st in stress_tests]
        axes[0, 1].barh(scenarios, max_dds, color='crimson', alpha=0.7)
        axes[0, 1].set_xlabel('Max Drawdown (%)', fontsize=9)
        axes[0, 1].set_title('Maximum Drawdown', fontsize=10)
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        recovery_days = [st.recovery_days for st in stress_tests]
        axes[1, 0].barh(scenarios, recovery_days, color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Days', fontsize=9)
        axes[1, 0].set_title('Recovery Time', fontsize=10)
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        sharpes = [st.sharpe_ratio for st in stress_tests]
        colors = ['green' if s > 1 else 'orange' if s > 0 else 'red' for s in sharpes]
        axes[1, 1].barh(scenarios, sharpes, color=colors, alpha=0.7)
        axes[1, 1].set_xlabel('Sharpe Ratio', fontsize=9)
        axes[1, 1].set_title('Risk-Adjusted Return', fontsize=10)
        axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        axes[1, 1].axvline(x=1, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_attribution_page(self, pdf: PdfPages, report_data: Dict):
        """Create performance attribution breakdown."""
        attribution = report_data["attribution"]
        
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle("Performance Attribution", fontsize=14, fontweight='bold')
        
        gs = gridspec.GridSpec(2, 2)
        
        ax1 = plt.subplot(gs[0, 0])
        labels = ['Long', 'Short']
        sizes = [attribution["long_contribution"], attribution["short_contribution"]]
        colors = ['#2E86AB', '#A23B72']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Contribution to Returns', fontsize=11)
        
        ax2 = plt.subplot(gs[0, 1])
        pnls = [attribution["long_pnl_pct"], attribution["short_pnl_pct"]]
        colors = ['green' if p > 0 else 'red' for p in pnls]
        ax2.bar(labels, pnls, color=colors, alpha=0.7)
        ax2.set_ylabel('P&L (%)', fontsize=9)
        ax2.set_title('Long vs Short P&L', fontsize=11)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(True, alpha=0.3, axis='y')
        
        ax3 = plt.subplot(gs[1, 0])
        trades = [attribution["long_trades"], attribution["short_trades"]]
        ax3.bar(labels, trades, color=['steelblue', 'orange'], alpha=0.7)
        ax3.set_ylabel('Number of Trades', fontsize=9)
        ax3.set_title('Trade Count', fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        
        ax4 = plt.subplot(gs[1, 1])
        win_rates = [attribution["long_win_rate"] * 100, attribution["short_win_rate"] * 100]
        ax4.bar(labels, win_rates, color=['forestgreen', 'crimson'], alpha=0.7)
        ax4.set_ylabel('Win Rate (%)', fontsize=9)
        ax4.set_title('Win Rate Comparison', fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(
        self, report_data: Dict[str, Any], strategy_name: str, output_path: Path
    ):
        """Generate interactive HTML report with plotly."""
        
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                'Equity Curve with Monte Carlo Bands',
                'Monthly Returns Heatmap',
                'Drawdown Timeline',
                'Trade P&L Distribution',
                'Cumulative Returns by Regime',
                'Stress Test Results',
                'Performance Attribution',
                'Risk Metrics Comparison'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "pie"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15
        )
        
        equity = report_data["equity"]
        dates = report_data["dates"]
        
        fig.add_trace(
            go.Scatter(x=dates, y=equity.values, mode='lines',
                      name='Equity', line=dict(color='#2E86AB', width=2)),
            row=1, col=1
        )
        
        mc_data = report_data["monte_carlo"]
        percentiles = mc_data["equity_percentiles"]
        fig.add_trace(
            go.Scatter(x=list(range(len(percentiles["p95"]))), y=percentiles["p95"],
                      mode='lines', name='95th Percentile',
                      line=dict(color='gray', dash='dash'), showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(range(len(percentiles["p5"]))), y=percentiles["p5"],
                      mode='lines', name='5th Percentile',
                      line=dict(color='gray', dash='dash'), fill='tonexty',
                      showlegend=False),
            row=1, col=1
        )
        
        monthly_returns = report_data["monthly_returns"]
        if not monthly_returns.empty:
            fig.add_trace(
                go.Heatmap(
                    z=monthly_returns.values,
                    x=monthly_returns.columns,
                    y=monthly_returns.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    showscale=True
                ),
                row=1, col=2
            )
        
        drawdown_df = report_data["drawdown"]
        fig.add_trace(
            go.Scatter(x=dates, y=drawdown_df["drawdown"].values * 100,
                      fill='tozeroy', name='Drawdown',
                      line=dict(color='red'), showlegend=False),
            row=2, col=1
        )
        
        trade_stats = report_data.get("trade_stats", [])
        if trade_stats:
            trades_df = pd.DataFrame([asdict(t) for t in trade_stats])
            fig.add_trace(
                go.Histogram(x=trades_df["pnl_pct"] * 100, name='Trade P&L',
                            marker_color='steelblue', showlegend=False),
                row=2, col=2
            )
        
        regime_perf = report_data.get("regime_performance", [])
        if regime_perf:
            regimes = [r.regime_name for r in regime_perf]
            total_pnls = [r.total_pnl_pct for r in regime_perf]
            fig.add_trace(
                go.Bar(x=regimes, y=total_pnls, name='Regime P&L',
                      marker_color=['green' if p > 0 else 'red' for p in total_pnls],
                      showlegend=False),
                row=3, col=1
            )
        
        stress_tests = report_data["stress_tests"]
        scenarios = [st.scenario_name for st in stress_tests]
        equity_changes = [st.equity_change_pct for st in stress_tests]
        fig.add_trace(
            go.Bar(x=scenarios, y=equity_changes, name='Stress Impact',
                  marker_color=['green' if e > 0 else 'red' for e in equity_changes],
                  showlegend=False),
            row=3, col=2
        )
        
        attribution = report_data["attribution"]
        fig.add_trace(
            go.Pie(labels=['Long', 'Short'],
                  values=[attribution["long_contribution"], attribution["short_contribution"]],
                  marker_colors=['#2E86AB', '#A23B72'],
                  showlegend=False),
            row=4, col=1
        )
        
        metrics = report_data["metrics"]
        metric_names = ['Sharpe', 'Sortino', 'Calmar', 'Profit Factor']
        metric_values = [
            metrics["sharpe"],
            metrics["sortino"],
            metrics["calmar_ratio"],
            metrics["profit_factor"]
        ]
        fig.add_trace(
            go.Bar(x=metric_names, y=metric_values, name='Metrics',
                  marker_color='steelblue', showlegend=False),
            row=4, col=2
        )
        
        fig.update_layout(
            height=1600,
            title_text=f"{strategy_name} - Comprehensive Backtest Report",
            title_font_size=20,
            showlegend=True
        )
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{strategy_name} Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #2E86AB; color: white; padding: 20px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
                .metric-card {{ background: #f5f5f5; padding: 15px; border-radius: 8px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; }}
                .metric-label {{ font-size: 14px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{strategy_name} - Backtest Report</h1>
                <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value">{metrics['total_return']:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{metrics['sharpe']:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value">{metrics['max_drawdown']:.2%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{metrics['win_rate']:.2%}</div>
                </div>
            </div>
            
            {fig.to_html(full_html=False, include_plotlyjs='cdn')}
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
