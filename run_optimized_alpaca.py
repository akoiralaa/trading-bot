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
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# REAL DATA OPTIMIZED PARAMETERS
OPTIMIZED_PARAMS = {
    'PLTR': {'lookback': 10, 'threshold': 0.20},
    'PENN': {'lookback': 35, 'threshold': 0.15},
    'QQQ': {'lookback': 20, 'threshold': 0.15},
    'SPY': {'lookback': 10, 'threshold': 0.05},
}


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


def backtest_ticker(trader, ticker: str) -> dict:
    """Backtest with optimized real data parameters."""
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
    entry_signals = np.logical_or(table_top_b, table_top_a).astype(int)
    
    targets = get_cluster_targets(df['close'].values, resistance)
    stops = vector * 0.985
    
    backtester = Backtester()
    metrics = backtester.run_backtest(df, vector, entry_signals, stops, targets, vector_strength)
    
    # Create chart
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f"{ticker} - Real Data Optimized", "Signal Strength"),
        vertical_spacing=0.15,
        row_heights=[0.7, 0.3]
    )
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=df['close'], name='Price', line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df['date'], y=vector, name='Vector', line=dict(color='red', width=1, dash='dash')),
        row=1, col=1
    )
    
    signal_dates = df.loc[entry_signals == 1, 'date']
    signal_prices = df.loc[entry_signals == 1, 'close']
    
    fig.add_trace(
        go.Scatter(x=signal_dates, y=signal_prices, mode='markers', name='Signal',
                   marker=dict(color='green', size=8, symbol='triangle-up')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=df['date'], y=vector_strength, name='Strength',
               marker=dict(color=vector_strength, colorscale='RdYlGn', cmin=-1, cmax=1)),
        row=2, col=1
    )
    
    fig.update_layout(title=f"{ticker} - Real Data Optimized", height=800, template='plotly_dark', hovermode='x unified')
    
    chart_file = f'{ticker}_optimized_real.html'
    fig.write_html(chart_file)
    
    return {
        'ticker': ticker,
        'metrics': metrics,
        'chart_file': chart_file,
        'params': params
    }


def main():
    print("="*70)
    print("QUANTUM FRACTALS - REAL DATA OPTIMIZED SYSTEM")
    print("="*70)
    
    trader = AlpacaTrader()
    
    if not trader.connect():
        return
    
    results = {}
    
    for ticker in ['PLTR', 'PENN', 'QQQ', 'SPY']:
        print(f"\nBacktesting {ticker} (optimized)...")
        result = backtest_ticker(trader, ticker)
        
        if result:
            results[ticker] = result
            m = result['metrics']
            p = result['params']
            print(f"  Params: lookback={p['lookback']}, threshold={p['threshold']:.2f}")
            print(f"  Results: {m['total_trades']} trades | PF: {m['profit_factor']:.2f}x | Win: {m['win_rate']:.1f}%")
            print(f"  Chart: {result['chart_file']}")
    
    print(f"\n{'='*70}")
    print("SUMMARY - REAL DATA OPTIMIZED")
    print(f"{'='*70}\n")
    
    for ticker, result in sorted(results.items(), key=lambda x: x[1]['metrics']['profit_factor'], reverse=True):
        m = result['metrics']
        print(f"{ticker}: {m['profit_factor']:.2f}x PF | {m['total_trades']} trades | {m['win_rate']:.1f}% win")


if __name__ == "__main__":
    main()
