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


OPTIMAL_PARAMS = {
    'COIN': {'lookback': 30, 'threshold': 0.12},
    'PENN': {'lookback': 25, 'threshold': 0.10},
    'PLTR': {'lookback': 30, 'threshold': 0.10},
    'QQQ': {'lookback': 20, 'threshold': 0.12},
}


def backtest_optimized(ticker: str) -> dict:
    params = OPTIMAL_PARAMS[ticker]
    lookback = params['lookback']
    threshold = params['threshold']
    
    loader = CSVDataLoader(ticker=ticker)
    df = loader.load_data(start_date="2020-01-01", end_date="2024-12-01")
    
    if df.empty:
        return None
    
    vector_calc = VectorCalculator(wave_period=7, lookback=lookback)
    vector = vector_calc.calculate_vector(df)
    vector_strength = vector_calc.get_vector_strength(df, vector)
    
    fractal_detector = FractalDetector(cluster_threshold=threshold)
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


if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUANTUM FRACTALS - OPTIMIZED SYSTEM")
    print("="*70)
    
    results = []
    total_pnl = 0
    total_trades = 0
    
    for ticker in ['COIN', 'PENN', 'PLTR', 'QQQ']:
        print(f"\n{ticker} (lookback={OPTIMAL_PARAMS[ticker]['lookback']}, threshold={OPTIMAL_PARAMS[ticker]['threshold']}):")
        metrics = backtest_optimized(ticker)
        
        if metrics:
            print(f"  Trades: {metrics['total_trades']}")
            print(f"  Win Rate: {metrics['win_rate']:.1f}%")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}x")
            print(f"  Total P&L: ${metrics['total_pnl']:.2f}")
            print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
            
            results.append({
                'Ticker': ticker,
                'Trades': metrics['total_trades'],
                'Win%': f"{metrics['win_rate']:.1f}%",
                'ProfitFactor': f"{metrics['profit_factor']:.2f}x",
                'TotalPnL': f"${metrics['total_pnl']:.0f}",
                'Sharpe': f"{metrics['sharpe_ratio']:.2f}"
            })
            
            total_pnl += metrics['total_pnl']
            total_trades += metrics['total_trades']
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    print(f"\nCombined Metrics:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Total P&L: ${total_pnl:.2f}")
    print(f"  Avg P&L per Trade: ${total_pnl/total_trades:.2f}")
