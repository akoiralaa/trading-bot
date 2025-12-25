import numpy as np
import pandas as pd
from src.data_loader import DataLoader
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
from src.backtester import Backtester


class FractalStrategy:
    """
    RCG Fractal Trading System - complete strategy.
    
    Rules:
    1. Calculate Wave 7 vector (support/resistance)
    2. Detect fractals and cluster them (10% rule)
    3. Wait for exhaustion (price can't stay above vector)
    4. Enter on Table Top A/B pattern (vector reclaim + confirmation)
    5. Stop below vector, target at next cluster
    """
    
    def __init__(self, ticker: str = "QQQ"):
        self.ticker = ticker
        self.data_loader = DataLoader(ticker)
        self.vector_calc = VectorCalculator(lookback_period=20)
        self.fractal_detector = FractalDetector(cluster_threshold=0.10)
        self.pattern_detector = PatternDetector(tolerance=0.002)
        self.backtester = Backtester(risk_per_trade=0.01)
    
    def run(self, start_date: str, end_date: str) -> Dict:
        """
        Run complete fractal trading strategy backtest.
        
        Returns: Backtest metrics and results
        """
        print(f"\n{'='*60}")
        print(f"FRACTAL TRADING STRATEGY - {self.ticker}")
        print(f"{'='*60}")
        
        # 1. Load data
        df = self.data_loader.load_data(start_date, end_date)
        
        # 2. Calculate vector
        print("\nCalculating Wave 7 vector...")
        vector = self.vector_calc.calculate_vector(df)
        
        # 3. Detect fractals
        print("Detecting fractals...")
        fractal_highs, fractal_lows = self.fractal_detector.detect_fractals(df)
        resistance, support = self.fractal_detector.get_resistance_and_support(fractal_highs, fractal_lows)
        print(f"✓ Found {len(resistance)} resistance clusters, {len(support)} support clusters")
        
        # 4. Detect patterns
        print("Detecting entry patterns...")
        table_top_b = self.pattern_detector.detect_table_top_b(df, vector)
        table_top_a = self.pattern_detector.detect_table_top_a(df, vector)
        entry_signals = np.logical_or(table_top_b, table_top_a).astype(int)
        print(f"✓ Found {int(np.sum(entry_signals))} potential entries")
        
        # 5. Set stops and targets
        print("Calculating stops and targets...")
        stop_losses = self._calculate_stops(df, vector)
        targets = self._calculate_targets(df, vector, support, resistance)
        
        # 6. Run backtest
        print("Running backtest...")
        metrics = self.backtester.run_backtest(df, vector, entry_signals, stop_losses, targets)
        
        # 7. Print results
        self.backtester.print_results(metrics)
        
        return metrics
    
    def _calculate_stops(self, df: pd.DataFrame, vector: np.ndarray) -> np.ndarray:
        """
        Calculate stop loss levels.
        Stop = below vector + 2% buffer
        """
        stops = vector * 0.98  # 2% below vector
        return stops
    
    def _calculate_targets(self, df: pd.DataFrame, vector: np.ndarray, 
                          support: list, resistance: list) -> np.ndarray:
        """
        Calculate take profit targets.
        Target = next resistance cluster or 2:1 R:R from entry
        """
        targets = np.zeros(len(df))
        
        for i in range(len(df)):
            # Simple target: 2:1 risk reward
            # If vector is at $100, stop at $98, then target is $104 (2% up from $100)
            targets[i] = vector[i] * 1.02  # 2% above vector
        
        return targets


if __name__ == "__main__":
    strategy = FractalStrategy(ticker="QQQ")
    metrics = strategy.run(start_date="2019-01-01", end_date="2024-12-01")
