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


OPTIMAL_PARAMS = {
    'QQQ': {'lookback': 20, 'threshold': 0.12},
    'PLTR': {'lookback': 30, 'threshold': 0.10},
    'PENN': {'lookback': 25, 'threshold': 0.10},
}


def get_alpaca_data(trader, ticker: str, days: int = 365) -> pd.DataFrame:
    """Get historical data from Alpaca API."""
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        
        print(f"Fetching {ticker} data from Alpaca...")
        
        bars = trader.api.get_bars(
            ticker,
            '1Day',
            start=start.strftime('%Y-%m-%dT00:00:00Z'),
            end=end.strftime('%Y-%m-%dT00:00:00Z'),
            limit=365
        )
        
        if bars is None or len(bars) == 0:
            print(f"No data for {ticker}")
            return pd.DataFrame()
        
        data = []
        for bar in bars:
            # Use t (time) instead of timestamp
            data.append({
                'date': pd.to_datetime(bar.t),
                'open': float(bar.o),
                'high': float(bar.h),
                'low': float(bar.l),
                'close': float(bar.c),
                'volume': int(bar.v)
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"Loaded {len(df)} bars for {ticker}")
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()


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


def backtest_ticker(trader, ticker: str) -> dict:
    """Backtest a ticker and return results + charts."""
    df = get_alpaca_data(trader, ticker, days=365)
    
    if df.empty:
        return None
    
    params = OPTIMAL_PARAMS.get(ticker, {'lookback': 20, 'threshold': 0.10})
    
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
        subplot_titles=(f"{ticker} - Price + Vector", "Signal Strength"),
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
    
    fig.update_layout(title=f"{ticker} Backtest", height=800, template='plotly_dark', hovermode='x unified')
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Strength", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    
    chart_file = f'{ticker}_backtest.html'
    fig.write_html(chart_file)
    print(f"Chart saved: {chart_file}")
    
    return {
        'ticker': ticker,
        'metrics': metrics,
        'chart_file': chart_file
    }


def main():
    print("="*70)
    print("QUANTUM FRACTALS - ALPACA DATA BACKTEST")
    print("="*70)
    
    trader = AlpacaTrader()
    
    if not trader.connect():
        return
    
    results = {}
    
    for ticker in ['QQQ', 'PLTR', 'PENN']:
        print(f"\nBacktesting {ticker}...")
        result = backtest_ticker(trader, ticker)
        
        if result:
            results[ticker] = result
            m = result['metrics']
            print(f"  Trades: {m['total_trades']} | PF: {m['profit_factor']:.2f}x | Win: {m['win_rate']:.1f}%")
    
    print(f"\n{'='*70}")
    print("Charts saved:")
    for ticker, result in results.items():
        print(f"  {result['chart_file']}")


if __name__ == "__main__":
    main()
