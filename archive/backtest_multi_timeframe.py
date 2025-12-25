import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.csv_data_loader import CSVDataLoader
from src.multi_timeframe_calculator import MultiTimeframeAnalyzer
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


if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUANTUM FRACTALS - MULTI-TIMEFRAME BACKTEST")
    print("="*60)
    
    # Load data (simulating daily data for now)
    loader = CSVDataLoader(ticker="QQQ")
    df = loader.load_data(start_date="2020-01-01", end_date="2024-12-01")
    
    if df.empty:
        print("Failed to load data.")
        exit(1)
    
    # Multi-timeframe analysis
    print("\nAnalyzing multiple timeframes...")
    mta = MultiTimeframeAnalyzer()
    
    daily_vector = mta.calculate_daily_vector(df, lookback=20)
    daily_strength = mta.get_vector_strength(df['close'].values, daily_vector)
    daily_bias = mta.get_daily_bias(daily_vector, df['close'].values)
    
    print(f"Daily Bias: {daily_bias}")
    print(f"Daily Vector: ${daily_vector[-1]:.2f}")
    
    # For now, treat hourly same as daily (in real implementation, use actual hourly data)
    hourly_vector = mta.calculate_hourly_vector(df, lookback=20)
    hourly_strength = mta.get_vector_strength(df['close'].values, hourly_vector)
    
    # Detect fractals
    print("Detecting fractals...")
    fractal_detector = FractalDetector(cluster_threshold=0.10)
    fractal_highs, fractal_lows = fractal_detector.detect_fractals(df)
    resistance, support = fractal_detector.get_resistance_and_support(fractal_highs, fractal_lows)
    
    # Detect patterns
    print("Detecting entry patterns...")
    pattern_detector = PatternDetector()
    table_top_b = pattern_detector.detect_table_top_b(df, hourly_vector, hourly_strength)
    table_top_a = pattern_detector.detect_table_top_a(df, hourly_vector, hourly_strength)
    hourly_signals = np.logical_or(table_top_b, table_top_a).astype(int)
    
    # Apply multi-timeframe confirmation
    print("Applying multi-timeframe confirmation...")
    confirmed_signals = np.zeros(len(df))
    
    for i in range(len(df)):
        if hourly_signals[i] == 1:
            # Check if daily bias allows this trade
            if mta.confirm_entry(True, daily_bias):
                confirmed_signals[i] = 1
    
    print(f"Hourly signals: {int(np.sum(hourly_signals))}")
    print(f"Confirmed signals: {int(np.sum(confirmed_signals))}")
    
    # Run backtest with confirmed signals only
    print("\nRunning multi-timeframe backtest...")
    targets = get_cluster_targets(df['close'].values, resistance)
    stops = hourly_vector * 0.985
    
    backtester = Backtester()
    metrics = backtester.run_backtest(df, hourly_vector, confirmed_signals, stops, targets, hourly_strength)
    backtester.print_results(metrics)
    
    print("\n" + "="*60)
    print("Quantum Fractals - Multi-Timeframe Analysis Complete")
    print("="*60)
