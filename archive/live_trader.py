import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.alpaca_trader import AlpacaTrader
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
from src.csv_data_loader import CSVDataLoader
import numpy as np


OPTIMAL_PARAMS = {
    'QQQ': {'lookback': 20, 'threshold': 0.12},
    'PLTR': {'lookback': 30, 'threshold': 0.10},
    'PENN': {'lookback': 25, 'threshold': 0.10},
}


def get_cluster_targets(close, resistance_clusters):
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


def analyze_ticker(ticker):
    try:
        params = OPTIMAL_PARAMS.get(ticker, {'lookback': 20, 'threshold': 0.10})
        loader = CSVDataLoader(ticker=ticker)
        df = loader.load_data(start_date="2020-01-01", end_date="2024-12-01")
        
        if df.empty:
            return None
        
        vector_calc = VectorCalculator(wave_period=7, lookback=params['lookback'])
        vector = vector_calc.calculate_vector(df)
        vector_strength = vector_calc.get_vector_strength(df, vector)
        
        fractal_detector = FractalDetector(cluster_threshold=params['threshold'])
        fractal_highs, fractal_lows = fractal_detector.detect_fractals(df)
        resistance, support = fractal_detector.get_resistance_and_support(fractal_highs, fractal_lows)
        
        pattern_detector = PatternDetector()
        table_top_b = pattern_detector.detect_table_top_b(df, vector, vector_strength)
        table_top_a = pattern_detector.detect_table_top_a(df, vector, vector_strength)
        
        latest_idx = len(df) - 1
        has_signal = table_top_b[latest_idx] == 1 or table_top_a[latest_idx] == 1
        
        current_price = df['close'].iloc[-1]
        current_vector = vector[latest_idx]
        targets = get_cluster_targets(df['close'].values, resistance)
        target_price = targets[latest_idx]
        
        return {
            'ticker': ticker,
            'signal': 'BUY' if has_signal else 'NONE',
            'current_price': current_price,
            'vector': current_vector,
            'target': target_price,
            'stop_loss': current_vector * 0.985,
            'strength': vector_strength[latest_idx]
        }
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None


def main():
    print("="*70)
    print("QUANTUM FRACTALS - LIVE TRADING BOT (ALPACA)")
    print("="*70)
    
    trader = AlpacaTrader()
    
    if not trader.connect():
        return
    
    account = trader.get_account_info()
    print(f"\nAccount: ${account.get('cash', 0):,.2f} cash")
    
    print("\nANALYZING SIGNALS...")
    
    for ticker in ['QQQ', 'PLTR', 'PENN']:
        signal = analyze_ticker(ticker)
        if signal and signal['signal'] == 'BUY':
            print(f"\n{ticker}: BUY @ ${signal['current_price']:.2f}")


if __name__ == "__main__":
    main()
