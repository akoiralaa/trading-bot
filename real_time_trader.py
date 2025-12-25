import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.alpaca_trader import AlpacaTrader
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time


OPTIMIZED_PARAMS = {
    'PLTR': {'lookback': 10, 'threshold': 0.20},
    'QQQ': {'lookback': 20, 'threshold': 0.15},
    'PENN': {'lookback': 35, 'threshold': 0.15},
    'SPY': {'lookback': 10, 'threshold': 0.05},
}


def get_alpaca_data(trader, ticker: str, days: int = 365) -> pd.DataFrame:
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        
        bars = trader.api.get_bars(ticker, '1Day',
            start=start.strftime('%Y-%m-%dT00:00:00Z'),
            end=end.strftime('%Y-%m-%dT00:00:00Z'),
            limit=365)
        
        if bars is None or len(bars) == 0:
            return pd.DataFrame()
        
        data = []
        for bar in bars:
            data.append({
                'date': pd.to_datetime(bar.t),
                'open': float(bar.o), 'high': float(bar.h),
                'low': float(bar.l), 'close': float(bar.c),
                'volume': int(bar.v)
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('date').reset_index(drop=True)
    except:
        return pd.DataFrame()


def get_cluster_targets(close, resistance_clusters):
    targets = np.zeros(len(close))
    for i in range(len(close)):
        current_price = close[i]
        higher = [r for r in resistance_clusters if r[0] > current_price]
        if higher:
            targets[i] = min(higher, key=lambda x: x[0] - current_price)[0]
        else:
            targets[i] = current_price * 1.03
    return targets


def analyze_ticker(trader, ticker: str):
    df = get_alpaca_data(trader, ticker, days=365)
    
    if df.empty:
        return None
    
    params = OPTIMIZED_PARAMS.get(ticker)
    if not params:
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
        'price': current_price,
        'vector': current_vector,
        'target': target_price,
        'stop_loss': current_vector * 0.985,
        'strength': vector_strength[latest_idx]
    }


def main():
    print("="*70)
    print("QUANTUM FRACTALS - REAL-TIME TRADING BOT")
    print("="*70)
    
    trader = AlpacaTrader()
    
    if not trader.connect():
        return
    
    account = trader.get_account_info()
    print(f"\nAccount: ${account['cash']:,.2f} cash")
    
    print("\nMonitoring for signals...\n")
    
    for ticker in ['PLTR', 'QQQ', 'PENN', 'SPY']:
        signal = analyze_ticker(trader, ticker)
        
        if signal and signal['signal'] == 'BUY':
            print(f"SIGNAL: {ticker} BUY @ ${signal['price']:.2f}")
            print(f"  Target: ${signal['target']:.2f} | Stop: ${signal['stop_loss']:.2f}")


if __name__ == "__main__":
    main()
