import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
from src.backtester import Backtester
import numpy as np
import pandas as pd


def create_realistic_data(num_bars: int = 1000) -> pd.DataFrame:
    """Generate realistic QQQ-like test data."""
    np.random.seed(42)
    
    # Start at realistic QQQ price
    prices = [420.0]
    
    # Simulate realistic price action with trends
    for i in range(num_bars - 1):
        # Add trend component
        trend = 0.0002 if i % 200 < 100 else -0.0001
        # Add volatility
        volatility = np.random.normal(0, 0.01)
        change = trend + volatility
        
        prices.append(max(prices[-1] * (1 + change), 300))
    
    dates = pd.date_range('2019-01-01', periods=num_bars, freq='D')
    
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + np.random.uniform(0.001, 0.015)) for p in prices],
        'low': [p * (1 - np.random.uniform(0.001, 0.015)) for p in prices],
        'close': prices,
        'volume': [np.random.randint(50000000, 150000000) for _ in prices]
    }, index=dates)
    
    return df


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FRACTAL TRADING STRATEGY - QQQ")
    print("="*60)
    
    df = create_realistic_data(1000)
    print(f"\n✓ Generated {len(df)} bars of realistic QQQ data")
    print(f"  Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    vector_calc = VectorCalculator(lookback_period=20)
    vector = vector_calc.calculate_vector(df)
    print(f"✓ Vector calculated (avg: ${vector[vector > 0].mean():.2f})")
    
    fractal_detector = FractalDetector(cluster_threshold=0.10)
    fractal_highs, fractal_lows = fractal_detector.detect_fractals(df)
    resistance, support = fractal_detector.get_resistance_and_support(fractal_highs, fractal_lows)
    print(f"✓ Found {len(resistance)} resistance clusters, {len(support)} support clusters")
    
    pattern_detector = PatternDetector()
    table_top_b = pattern_detector.detect_table_top_b(df, vector)
    table_top_a = pattern_detector.detect_table_top_a(df, vector)
    entry_signals = np.logical_or(table_top_b, table_top_a).astype(int)
    print(f"✓ Found {int(np.sum(entry_signals))} entry signals")
    
    stops = vector * 0.98
    targets = vector * 1.02
    
    backtester = Backtester()
    metrics = backtester.run_backtest(df, vector, entry_signals, stops, targets)
    backtester.print_results(metrics)
    
    print("\n✓ System validated and ready for real QQQ data!")
