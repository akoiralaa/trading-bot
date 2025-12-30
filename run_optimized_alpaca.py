import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure local imports are resolved
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.alpaca_trader import AlpacaTrader
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
from src.backtester import AdvancedBacktester # Utilizing the refactored engine

logger = logging.getLogger("StrategyVisualizer")

class StrategyPerformanceVisualizer:
    """
    Orchestrates high-fidelity backtesting and generates interactive 
    visual performance attribution charts.
    """

    STRATEGY_CONFIG = {
        'PLTR': {'lookback': 10, 'threshold': 0.20},
        'PENN': {'lookback': 35, 'threshold': 0.15},
        'QQQ':  {'lookback': 20, 'threshold': 0.15},
        'SPY':  {'lookback': 10, 'threshold': 0.05},
    }

    def __init__(self, horizon_days: int = 365) -> None:
        self.trader = AlpacaTrader()
        self.horizon_days = horizon_days
        
        if not self.trader.connect():
            raise ConnectionError("Authentication failed with Alpaca API.")

    def fetch_standardized_data(self, ticker: str) -> pd.DataFrame:
        """Retrieves and normalizes OHLCV time-series."""
        try:
            end = datetime.now()
            start = end - timedelta(days=self.horizon_days)
            
            bars = self.trader.api.get_bars(
                ticker, '1Day',
                start=start.strftime('%Y-%m-%dT00:00:00Z'),
                end=end.strftime('%Y-%m-%dT00:00:00Z')
            ).df
            
            if bars.empty:
                return pd.DataFrame()
                
            return bars[['open', 'high', 'low', 'close', 'volume']].sort_index()
        except Exception as e:
            logger.error(f"DataError | Ticker: {ticker} | Exception: {e}")
            return pd.DataFrame()

    def run_attribution_pipeline(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Executes full backtest and generates Plotly visualization."""
        df = self.fetch_standardized_data(ticker)
        if df.empty:
            return None

        params = self.STRATEGY_CONFIG.get(ticker)
        
        # 1. Indicator Calculation
        vc = VectorCalculator(wave_period=7, lookback=params['lookback'])
        vector = vc.calculate_vector(df)
        strength = vc.get_vector_strength(df, vector)
        
        # 2. Structural Signal Detection
        fd = FractalDetector(cluster_threshold=params['threshold'])
        f_highs, f_lows = fd.detect_fractals(df)
        resistance, _ = fd.get_resistance_and_support(f_highs, f_lows)
        
        pd_detector = PatternDetector()
        entry_signals = (pd_detector.detect_table_top_b(df, vector, strength) | 
                         pd_detector.detect_table_top_a(df, vector, strength)).astype(int)
        
        # 3. Execution Logic (Stops/Targets)
        stops = vector * 0.985
        targets = self._derive_cluster_targets(df['close'].values, resistance)
        
        # 4. Metric Computation
        # Note: AdvancedBacktester handles the friction-adjusted results
        bt = AdvancedBacktester(engine=None) # Integration placeholder
        results = bt.run_backtest(df['close'].values, vector, strength, df['volume'].values)
        
        chart_path = self._generate_plotly_artifacts(ticker, df, vector, entry_signals, strength)
        
        return {
            'ticker': ticker,
            'metrics': results,
            'chart_path': chart_path
        }

    def _derive_cluster_targets(self, prices: np.ndarray, resistance: List[Any]) -> np.ndarray:
        """Projects implementation targets based on structural resistance clusters."""
        targets = np.zeros(len(prices))
        for i, price in enumerate(prices):
            upside = [r[0] for r in resistance if r[0] > price]
            targets[i] = min(upside) if upside else price * 1.03
        return targets

    def _generate_plotly_artifacts(self, ticker: str, df: pd.DataFrame, 
                                 vector: np.ndarray, signals: np.ndarray, strength: np.ndarray) -> str:
        """Constructs an interactive performance dashboard."""
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            subplot_titles=(f"{ticker} Structural Vectors", "Stochastic Signal Strength"),
            vertical_spacing=0.08, 
            row_heights=[0.7, 0.3]
        )

        # Main Price/Vector Plot
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Price', line=dict(color='#1f77b4')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=vector, name='Vector', line=dict(color='#d62728', dash='dash')), row=1, col=1)

        # Signal Markers
        sig_idx = df.index[signals == 1]
        fig.add_trace(go.Scatter(x=sig_idx, y=df.loc[sig_idx, 'close'], mode='markers', 
                                 marker=dict(symbol='triangle-up', size=10, color='#2ca02c'), name='Entry'), row=1, col=1)

        # Strength Oscillator
        fig.add_trace(go.Bar(x=df.index, y=strength, marker=dict(color=strength, colorscale='RdYlGn'), name='SignalStrength'), row=2, col=1)

        fig.update_layout(template='plotly_dark', height=900, showlegend=False)
        output_file = f"artifacts/{ticker}_report.html"
        fig.write_html(output_file)
        return output_file

if __name__ == "__main__":
    visualizer = StrategyPerformanceVisualizer()
    print(f"{'Asset':<8} {'ProfitFactor':<15} {'WinRate':<10} {'Sharpe':<10}")
    print("-" * 50)
    
    for symbol in visualizer.STRATEGY_CONFIG.keys():
        stats = visualizer.run_attribution_pipeline(symbol)
        if stats:
            m = stats['metrics']
            print(f"{symbol:<8} {m['expectancy']:<15.2f} {m['win_rate']:<10.1%} {m['sharpe_ratio']:<10.2f}")