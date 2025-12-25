import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.alpaca_trader import AlpacaTrader
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
from src.backtester import Backtester
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


OPTIMIZED_PARAMS = {
    'PLTR': {'lookback': 10, 'threshold': 0.20},
    'PENN': {'lookback': 35, 'threshold': 0.15},
    'QQQ': {'lookback': 20, 'threshold': 0.15},
    'SPY': {'lookback': 10, 'threshold': 0.05},
}


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


def backtest_ticker(trader, ticker: str, risk_pct: float):
    df = get_alpaca_data(trader, ticker)
    
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
    entry_signals = np.logical_or(table_top_b, table_top_a).astype(int)
    
    targets = get_cluster_targets(df['close'].values, resistance)
    stops = vector * 0.985
    
    backtester = Backtester(risk_per_trade=risk_pct)
    metrics = backtester.run_backtest(df, vector, entry_signals, stops, targets, vector_strength)
    
    return metrics


def main():
    print("="*90)
    print("OPTIMAL RISK LEVEL ANALYSIS")
    print("="*90)
    
    trader = AlpacaTrader()
    
    if not trader.connect():
        return
    
    tickers = ['PLTR', 'QQQ', 'PENN', 'SPY']
    risk_levels = [0.01, 0.02, 0.03, 0.05, 0.10]
    
    for risk_pct in risk_levels:
        print(f"\n{'='*90}")
        print(f"RISK: {risk_pct*100:.0f}% per trade")
        print(f"{'='*90}\n")
        print(f"{'Ticker':<8} {'P&L':<14} {'MaxDD%':<10} {'RoR%':<10} {'Status':<10}")
        print("-" * 70)
        
        total_pnl = 0
        max_dd = 0
        
        for ticker in tickers:
            metrics = backtest_ticker(trader, ticker, risk_pct)
            if metrics:
                pnl = metrics['total_pnl']
                dd = metrics['max_drawdown']
                ror = (pnl / 100000) * 100
                
                status = "✅ Good" if dd < 10 else "⚠️ Risky" if dd < 20 else "❌ Danger"
                
                print(f"{ticker:<8} ${pnl:<13,.2f} {dd:<10.2f}% {ror:<10.2f}% {status:<10}")
                
                total_pnl += pnl
                max_dd = max(max_dd, dd)
        
        print("-" * 70)
        print(f"{'TOTAL':<8} ${total_pnl:<13,.2f} {max_dd:<10.2f}% {(total_pnl/100000)*100:<10.2f}%")


if __name__ == "__main__":
    main()
