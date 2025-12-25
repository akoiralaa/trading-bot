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


def get_alpaca_data(trader, ticker: str, days: int = 365) -> pd.DataFrame:
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        
        bars = trader.api.get_bars(
            ticker, '1Day',
            start=start.strftime('%Y-%m-%dT00:00:00Z'),
            end=end.strftime('%Y-%m-%dT00:00:00Z'),
            limit=365
        )
        
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


def analyze_ticker(trader, ticker: str, lookback: int, threshold: float):
    df = get_alpaca_data(trader, ticker, days=365)
    
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
    
    backtester = Backtester(initial_capital=100000)
    metrics = backtester.run_backtest(df, vector, entry_signals, stops, targets, vector_strength)
    
    # Calculate equity curve manually
    pnls = [t.pnl for t in backtester.trades if t.pnl is not None]
    equity_curve = np.cumsum(pnls) + 100000
    
    print(f"\n{ticker} ANALYSIS")
    print(f"{'='*60}")
    print(f"Trades: {len(pnls)}")
    print(f"P&Ls: {[f'${p:.2f}' for p in pnls]}")
    print(f"Equity: {[f'${e:.0f}' for e in equity_curve]}")
    
    # Calculate drawdown properly
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (running_max - equity_curve) / running_max * 100
    
    print(f"Max Drawdown: {np.max(drawdowns):.2f}%")
    print(f"Profit Factor: {metrics['profit_factor']:.2f}x")
    print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")


def main():
    trader = AlpacaTrader()
    
    if not trader.connect():
        return
    
    analyze_ticker(trader, 'PLTR', 10, 0.20)
    analyze_ticker(trader, 'SPY', 10, 0.05)


if __name__ == "__main__":
    main()
