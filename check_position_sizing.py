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


def get_alpaca_data(trader, ticker: str) -> pd.DataFrame:
    try:
        end = datetime.now()
        start = end - timedelta(days=365)
        
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


def check_sizing(trader, ticker: str, lookback: int, threshold: float):
    df = get_alpaca_data(trader, ticker)
    
    if df.empty:
        return
    
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
    
    targets = np.zeros(len(df))
    for i in range(len(df)):
        price = df['close'].iloc[i]
        higher = [r for r in resistance if r[0] > price]
        targets[i] = min(higher, key=lambda x: x[0] - price)[0] if higher else price * 1.03
    
    stops = vector * 0.985
    
    print(f"\n{ticker} POSITION SIZING")
    print(f"{'='*60}")
    
    entry_indices = np.where(entry_signals == 1)[0]
    
    for idx in entry_indices[-3:]:
        entry_price = df['close'].iloc[idx]
        stop_price = stops[idx]
        target_price = targets[idx]
        
        risk_per_share = entry_price - stop_price
        reward_per_share = target_price - entry_price
        
        account_risk = 100000 * 0.01
        quantity = account_risk / risk_per_share
        
        expected_pnl = quantity * reward_per_share
        expected_loss = quantity * risk_per_share
        
        print(f"\nEntry: ${entry_price:.2f}")
        print(f"  Stop: ${stop_price:.2f} (risk: ${risk_per_share:.2f})")
        print(f"  Target: ${target_price:.2f} (reward: ${reward_per_share:.2f})")
        print(f"  Quantity: {quantity:.0f} shares")
        print(f"  Expected Win: ${expected_pnl:.2f}")
        print(f"  Expected Loss: ${expected_loss:.2f}")


def main():
    trader = AlpacaTrader()
    
    if not trader.connect():
        print("Failed to connect")
        return
    
    check_sizing(trader, 'PLTR', 10, 0.20)
    check_sizing(trader, 'SPY', 10, 0.05)


if __name__ == "__main__":
    main()
