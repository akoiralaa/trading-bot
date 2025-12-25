import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.yahoo_finance_scraper import YahooFinanceScraper
from src.csv_data_loader import CSVDataLoader
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
from src.backtester import Backtester
import numpy as np


def get_cluster_targets(close: np.ndarray, resistance_clusters: list, support_clusters: list) -> np.ndarray:
    """
    Calculate targets based on actual fractal clusters, not fixed percentages.
    
    For each bar, find the next resistance cluster above current price.
    That becomes the target.
    """
    targets = np.zeros(len(close))
    
    for i in range(len(close)):
        current_price = close[i]
        
        # Find all resistance clusters above current price
        higher_clusters = [r for r in resistance_clusters if r[0] > current_price]
        
        if higher_clusters:
            # Use nearest cluster as target
            nearest = min(higher_clusters, key=lambda x: x[0] - current_price)
            targets[i] = nearest[0]  # Use low of cluster as target
        else:
            # No cluster above, use 3% above current price as fallback
            targets[i] = current_price * 1.03
    
    return targets


if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUANTUM FRACTALS - CLUSTER-BASED BACKTEST")
    print("="*60)
    
    scraper = YahooFinanceScraper(ticker="QQQ")
    df = scraper.load_data(start_date="2020-01-01", end_date="2024-12-01")
    
    if df.empty:
        print("\nFalling back to CSV loader...")
        loader = CSVDataLoader(ticker="QQQ")
        df = loader.load_data(start_date="2020-01-01", end_date="2024-12-01")
    
    if df.empty:
        print("Failed to load data.")
        exit(1)
    
    print("\nCalculating Wave 7 vector...")
    vector_calc = VectorCalculator(lookback_period=20)
    vector = vector_calc.calculate_vector(df)
    
    print("Detecting fractals and clusters...")
    fractal_detector = FractalDetector(cluster_threshold=0.10)
    fractal_highs, fractal_lows = fractal_detector.detect_fractals(df)
    resistance, support = fractal_detector.get_resistance_and_support(fractal_highs, fractal_lows)
    print(f"Found {len(resistance)} resistance clusters, {len(support)} support clusters")
    
    print("Detecting entry patterns...")
    pattern_detector = PatternDetector()
    table_top_b = pattern_detector.detect_table_top_b(df, vector)
    table_top_a = pattern_detector.detect_table_top_a(df, vector)
    entry_signals = np.logical_or(table_top_b, table_top_a).astype(int)
    print(f"Found {int(entry_signals.sum())} entry signals")
    
    # Use cluster-based targets instead of fixed %
    print("Calculating cluster-based targets...")
    targets = get_cluster_targets(df['close'].values, resistance, support)
    
    # Stops still at vector - 2%
    stops = vector * 0.98
    
    print("\nRunning backtest with real cluster targets...")
    backtester = Backtester()
    metrics = backtester.run_backtest(df, vector, entry_signals, stops, targets)
    backtester.print_results(metrics)
    
    print("\n" + "="*60)
    print("Quantum Fractals - Cluster-Based Execution")
    print("="*60)
