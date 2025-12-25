import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
from src.backtester import Backtester


def create_test_data(num_bars: int = 500) -> pd.DataFrame:
    """Generate synthetic price data for testing."""
    np.random.seed(42)
    
    prices = [420.0]
    
    for _ in range(num_bars - 1):
        change = np.random.normal(0.001, 0.01)
        prices.append(prices[-1] * (1 + change))
    
    dates = pd.date_range('2019-01-01', periods=num_bars, freq='D')
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.005 for p in prices],
        'low': [p * 0.995 for p in prices],
        'close': prices,
        'volume': [1000000] * num_bars
    }, index=dates)
    
    return df


if __name__ == "__main__":
    print("\nTesting Fractal Trading System...\n")
    
    df = create_test_data(500)
    print(f"✓ Created {len(df)} bars")
    
    vector_calc = VectorCalculator(lookback_period=20)
    vector = vector_calc.calculate_vector(df)
    print(f"✓ Vector calculated")
    
    fractal_detector = FractalDetector(cluster_threshold=0.10)
    fractal_highs, fractal_lows = fractal_detector.detect_fractals(df)
    num_fractals = int(np.sum(fractal_highs > 0)) + int(np.sum(fractal_lows > 0))
    print(f"✓ Found {num_fractals} fractals")
    
    pattern_detector = PatternDetector()
    table_top_b = pattern_detector.detect_table_top_b(df, vector)
    table_top_a = pattern_detector.detect_table_top_a(df, vector)
    num_patterns = int(np.sum(table_top_b)) + int(np.sum(table_top_a))
    print(f"✓ Found {num_patterns} patterns")
    
    entry_signals = np.logical_or(table_top_b, table_top_a).astype(int)
    stops = vector * 0.98
    targets = vector * 1.02
    
    backtester = Backtester()
    metrics = backtester.run_backtest(df, vector, entry_signals, stops, targets)
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    backtester.print_results(metrics)
    print("✓ All systems working!")
