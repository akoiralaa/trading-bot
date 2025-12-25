import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.csv_data_loader import CSVDataLoader
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
import numpy as np


if __name__ == "__main__":
    print("="*60)
    print("DIAGNOSING SPY FAILURE")
    print("="*60)
    
    loader = CSVDataLoader(ticker="SPY")
    df = loader.load_data(start_date="2020-01-01", end_date="2024-12-01")
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    
    # Calculate metrics
    volatility = np.std(np.diff(close) / close[:-1]) * 100
    price_range = high.max() - low.min()
    avg_price = close.mean()
    
    print(f"\nPrice Metrics:")
    print(f"  Close range: ${low.min():.2f} - ${high.max():.2f}")
    print(f"  Avg price: ${avg_price:.2f}")
    print(f"  Daily volatility: {volatility:.2f}%")
    print(f"  Price range as % of avg: {(price_range / avg_price * 100):.2f}%")
    
    # Vector analysis
    vector_calc = VectorCalculator(wave_period=7, lookback=20)
    vector = vector_calc.calculate_vector(df)
    vector_strength = vector_calc.get_vector_strength(df, vector)
    
    print(f"\nVector Metrics:")
    print(f"  Avg vector level: ${vector[vector > 0].mean():.2f}")
    print(f"  Vector as % of price: {(vector[vector > 0].mean() / avg_price * 100):.2f}%")
    print(f"  Avg vector strength: {np.mean(vector_strength[vector_strength != 0]):.3f}")
    print(f"  Strong signals (>0.4): {np.sum(np.abs(vector_strength) > 0.4)}")
    
    # Fractal analysis
    fractal_detector = FractalDetector(cluster_threshold=0.10)
    fractal_highs, fractal_lows = fractal_detector.detect_fractals(df)
    resistance, support = fractal_detector.get_resistance_and_support(fractal_highs, fractal_lows)
    
    print(f"\nFractal Metrics:")
    print(f"  Total fractals (highs): {np.sum(fractal_highs)}")
    print(f"  Total fractals (lows): {np.sum(fractal_lows)}")
    print(f"  Resistance clusters: {len(resistance)}")
    print(f"  Support clusters: {len(support)}")
    
    if resistance:
        print(f"  Avg cluster width: {np.mean([r[1] - r[0] for r in resistance]):.2f}")
    
    # Entry signals
    pattern_detector = PatternDetector()
    table_top_b = pattern_detector.detect_table_top_b(df, vector, vector_strength)
    table_top_a = pattern_detector.detect_table_top_a(df, vector, vector_strength)
    
    print(f"\nEntry Signals:")
    print(f"  Table Top B signals: {np.sum(table_top_b)}")
    print(f"  Table Top A signals: {np.sum(table_top_a)}")
    print(f"  Total signals: {np.sum(table_top_b) + np.sum(table_top_a)}")
    
    # Compare to QQQ
    print(f"\n" + "="*60)
    print("COMPARISON: QQQ vs SPY")
    print("="*60)
    
    loader_qqq = CSVDataLoader(ticker="QQQ")
    df_qqq = loader_qqq.load_data(start_date="2020-01-01", end_date="2024-12-01")
    
    vol_qqq = np.std(np.diff(df_qqq['close'].values) / df_qqq['close'].values[:-1]) * 100
    
    print(f"SPY volatility: {volatility:.2f}%")
    print(f"QQQ volatility: {vol_qqq:.2f}%")
    print(f"QQQ is {vol_qqq / volatility:.1f}x more volatile")
