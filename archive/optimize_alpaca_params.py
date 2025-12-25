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
    """Get historical data from Alpaca."""
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        
        bars = trader.api.get_bars(
            ticker,
            '1Day',
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
                'open': float(bar.o),
                'high': float(bar.h),
                'low': float(bar.l),
                'close': float(bar.c),
                'volume': int(bar.v)
            })
        
        df = pd.DataFrame(data)
        return df.sort_values('date').reset_index(drop=True)
        
    except Exception as e:
        print(f"Error: {e}")
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


def backtest_params(df, lookback, threshold):
    """Test one parameter combination."""
    if len(df) < 50:
        return None
    
    try:
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
        
    except:
        return None


def optimize_ticker(trader, ticker: str):
    """Find best parameters for a ticker on real data."""
    print(f"\nOptimizing {ticker}...")
    
    df = get_alpaca_data(trader, ticker, days=365)
    
    if df.empty:
        print(f"No data for {ticker}")
        return None
    
    print(f"Loaded {len(df)} bars")
    
    best = None
    lookbacks = [10, 15, 20, 25, 30, 35, 40]
    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    
    for lookback in lookbacks:
        for threshold in thresholds:
            metrics = backtest_params(df, lookback, threshold)
            
            if metrics and metrics['total_trades'] > 2:
                pf = metrics['profit_factor']
                
                if best is None or pf > best['pf']:
                    best = {
                        'ticker': ticker,
                        'lookback': lookback,
                        'threshold': threshold,
                        'pf': pf,
                        'trades': metrics['total_trades'],
                        'win_rate': metrics['win_rate'],
                        'sharpe': metrics['sharpe_ratio']
                    }
    
    if best:
        print(f"  Best: lookback={best['lookback']}, threshold={best['threshold']:.2f}")
        print(f"  PF: {best['pf']:.2f}x | Trades: {best['trades']} | Win: {best['win_rate']:.1f}%")
    
    return best


def main():
    print("="*70)
    print("OPTIMIZING ON REAL ALPACA DATA")
    print("="*70)
    
    trader = AlpacaTrader()
    
    if not trader.connect():
        return
    
    tickers = ['QQQ', 'PLTR', 'PENN', 'SPY']
    results = {}
    
    for ticker in tickers:
        result = optimize_ticker(trader, ticker)
        if result:
            results[ticker] = result
    
    print(f"\n{'='*70}")
    print("OPTIMIZED PARAMETERS FOR REAL DATA")
    print(f"{'='*70}")
    
    for ticker, result in sorted(results.items(), key=lambda x: x[1]['pf'], reverse=True):
        print(f"\n{ticker}:")
        print(f"  Lookback: {result['lookback']}")
        print(f"  Threshold: {result['threshold']:.2f}")
        print(f"  Profit Factor: {result['pf']:.2f}x")
        print(f"  Trades: {result['trades']}")
        print(f"  Win Rate: {result['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
