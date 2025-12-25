import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.csv_data_loader import CSVDataLoader
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
from src.backtester import Backtester
import numpy as np
import pandas as pd


def get_cluster_targets(close: np.ndarray, resistance_clusters: list) -> np.ndarray:
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


def backtest_with_params(ticker: str, vector_lookback: int, cluster_threshold: float, strength_threshold: float) -> dict:
    """Test one combination of parameters."""
    try:
        loader = CSVDataLoader(ticker=ticker)
        df = loader.load_data(start_date="2020-01-01", end_date="2024-12-01")
        
        if df.empty:
            return None
        
        vector_calc = VectorCalculator(wave_period=7, lookback=vector_lookback)
        vector = vector_calc.calculate_vector(df)
        vector_strength = vector_calc.get_vector_strength(df, vector)
        
        fractal_detector = FractalDetector(cluster_threshold=cluster_threshold)
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
        
    except:
        return None


if __name__ == "__main__":
    print("="*80)
    print("QUANTUM FRACTALS - PARAMETER OPTIMIZATION")
    print("="*80)
    
    winning_tickers = ["COIN", "PENN", "PLTR", "QQQ"]
    
    vector_lookbacks = [15, 20, 25, 30]
    cluster_thresholds = [0.08, 0.10, 0.12, 0.15]
    
    best_results = []
    
    for ticker in winning_tickers:
        print(f"\n{ticker}:")
        ticker_best = None
        
        for lookback in vector_lookbacks:
            for threshold in cluster_thresholds:
                metrics = backtest_with_params(ticker, lookback, threshold, 0.5)
                
                if metrics and metrics['total_trades'] > 0:
                    pf = metrics['profit_factor']
                    
                    if ticker_best is None or pf > ticker_best['profit_factor']:
                        ticker_best = {
                            'ticker': ticker,
                            'lookback': lookback,
                            'threshold': threshold,
                            'profit_factor': pf,
                            'trades': metrics['total_trades'],
                            'win_rate': metrics['win_rate'],
                            'sharpe': metrics['sharpe_ratio']
                        }
        
        if ticker_best:
            print(f"  Best: lookback={ticker_best['lookback']}, threshold={ticker_best['threshold']:.2f}, PF={ticker_best['profit_factor']:.2f}x ({ticker_best['trades']} trades)")
            best_results.append(ticker_best)
    
    print("\n" + "="*80)
    print("OPTIMAL PARAMETERS BY TICKER")
    print("="*80)
    df_results = pd.DataFrame(best_results)
    print(df_results[['ticker', 'lookback', 'threshold', 'profit_factor', 'trades', 'win_rate']].to_string(index=False))
