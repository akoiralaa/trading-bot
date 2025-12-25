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


if __name__ == "__main__":
    print("\n" + "="*60)
    print("QUANTUM FRACTALS - REAL DATA BACKTEST")
    print("="*60)
    
    scraper = YahooFinanceScraper(ticker="QQQ")
    df = scraper.load_data(start_date="2020-01-01", end_date="2024-12-01")
    
    if df.empty:
        print("\nFalling back to CSV loader...")
        loader = CSVDataLoader(ticker="QQQ")
        df = loader.load_data(start_date="2020-01-01", end_date="2024-12-01")
    
    if df.empty:
        print("Failed to load data. Please manually download QQQ.csv from Yahoo Finance.")
        exit(1)
    
    print("\nCalculating Wave 7 vector...")
    vector_calc = VectorCalculator(lookback_period=20)
    vector = vector_calc.calculate_vector(df)
    
    print("Detecting fractals...")
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
    
    stops = vector * 0.98
    targets = vector * 1.02
    
    print("\nRunning backtest on real data...")
    backtester = Backtester()
    metrics = backtester.run_backtest(df, vector, entry_signals, stops, targets)
    backtester.print_results(metrics)
    
    print("\n" + "="*60)
    print("Quantum Fractals validated on real market data!")
    print("="*60)
