import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.csv_data_loader import CSVDataLoader
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
from src.backtester import Backtester
import numpy as np


def get_cluster_targets(close: np.ndarray, resistance_clusters: list) -> np.ndarray:
    """Calculate cluster-based targets."""
    targets = np.zeros(len(close))
    
    for i in range(len(close)):
        current_price = close[i]
        higher_clusters = [r for r in resistance_clusters if r[0] > current_price]
        
        if higher_clusters:
            nearest = min(higher_clusters, key=lambda x: x[0] - current_price)
            targets[i] = nearest[0]
        else:
            targets[i] = current_price * 1.03
    
    return targets


def backtest_ticker(ticker: str) -> dict:
    """Backtest a single ticker."""
    print(f"\nBacktesting {ticker}...")
    
    try:
        loader = CSVDataLoader(ticker=ticker)
        df = loader.load_data(start_date="2020-01-01", end_date="2024-12-01")
        
        if df.empty:
            print(f"No data for {ticker}")
            return None
        
        vector_calc = VectorCalculator(wave_period=7, lookback=20)
        vector = vector_calc.calculate_vector(df)
        vector_strength = vector_calc.get_vector_strength(df, vector)
        
        fractal_detector = FractalDetector(cluster_threshold=0.10)
        fractal_highs, fractal_lows = fractal_detector.detect_fractals(df)
        resistance, support = fractal_detector.get_resistance_and_support(fractal_highs, fractal_lows)
        
        pattern_detector = PatternDetector()
        table_top_b = pattern_detector.detect_table_top_b(df, vector, vector_strength)
        table_top_a = pattern_detector.detect_table_top_a(df, vector, vector_strength)
        entry_signals = np.logical_or(table_top_b, table_top_a).astype(int)
        
        targets = get_cluster_targets(df['close'].values, resistance)
        stops = vector * 0.985
        
        backtester = Backtester()
        metrics = backtester.run_backtest(df, vector, entry_signals, stops, targets, vector_strength)
        
        return metrics
        
    except Exception as e:
        print(f"Error testing {ticker}: {e}")
        return None


if __name__ == "__main__":
    print("="*60)
    print("QUANTUM FRACTALS - MULTI-TICKER BACKTEST")
    print("="*60)
    
    tickers = ["QQQ", "SPY", "NVDA"]
    results = {}
    
    for ticker in tickers:
        metrics = backtest_ticker(ticker)
        if metrics:
            results[ticker] = metrics
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY - ALL TICKERS")
    print("="*60)
    
    for ticker, metrics in results.items():
        print(f"\n{ticker}")
        print(f"  Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}x")
        print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
