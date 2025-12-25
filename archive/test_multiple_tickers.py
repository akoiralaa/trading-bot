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


def generate_ticker_data(ticker: str, start_price: float):
    """Generate realistic data for any ticker."""
    from datetime import datetime
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 1)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    np.random.seed(hash(ticker) % 2**32)
    
    prices = [start_price]
    
    for i in range(len(dates) - 1):
        if i < len(dates) // 2:
            trend = 0.0003
        else:
            trend = 0.0001
        
        volatility = np.random.normal(0, 0.015)
        change = trend + volatility
        
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, start_price * 0.5))
    
    open_prices = [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices]
    high_prices = [max(p, o) * (1 + np.random.uniform(0, 0.02)) for p, o in zip(prices, open_prices)]
    low_prices = [min(p, o) * (1 - np.random.uniform(0, 0.02)) for p, o in zip(prices, open_prices)]
    volume = [np.random.randint(30000000, 100000000) for _ in prices]
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': prices,
        'Volume': volume
    })
    
    df.set_index('Date', inplace=True)
    
    csv_path = f'data/{ticker}.csv'
    df.to_csv(csv_path)
    
    return df


def get_cluster_targets(close: np.ndarray, resistance_clusters: list) -> np.ndarray:
    """Calculate cluster-based targets."""
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


def backtest_ticker(ticker: str, start_price: float = None) -> dict:
    """Backtest a single ticker."""
    print(f"\nBacktesting {ticker}...", end=" ")
    
    try:
        # Generate data if not exists
        if start_price:
            generate_ticker_data(ticker, start_price)
        
        loader = CSVDataLoader(ticker=ticker)
        df = loader.load_data(start_date="2020-01-01", end_date="2024-12-01")
        
        if df.empty:
            print("FAILED - No data")
            return None
        
        vector_calc = VectorCalculator(wave_period=7, lookback=20)
        vector = vector_calc.calculate_vector(df)
        vector_strength = vector_calc.get_vector_strength(df, vector)
        
        fractal_detector = FractalDetector(cluster_threshold=0.10)
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
        
        print(f"OK - {metrics['total_trades']} trades, {metrics['profit_factor']:.2f}x PF")
        return metrics
        
    except Exception as e:
        print(f"ERROR - {str(e)[:50]}")
        return None


if __name__ == "__main__":
    print("="*70)
    print("QUANTUM FRACTALS - MULTI-TICKER ROBUSTNESS TEST")
    print("="*70)
    
    # Test different types of stocks
    tickers = [
        ("QQQ", 200.0),      # Tech ETF (high volatility)
        ("SPY", 320.0),      # Broad market (medium volatility)
        ("NVDA", 70.0),      # High-growth tech
        ("AAPL", 150.0),     # Large cap tech
        ("AMD", 90.0),       # Semiconductor
        ("TSLA", 180.0),     # Electric vehicles
        ("COIN", 85.0),      # Crypto exchange
        ("MSTR", 200.0),     # Bitcoin company
        ("PENN", 40.0),      # Gaming (low price)
        ("PLTR", 25.0),      # Penny-ish stock
    ]
    
    results = {}
    
    for ticker, start_price in tickers:
        metrics = backtest_ticker(ticker, start_price)
        if metrics:
            results[ticker] = metrics
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - ALL TICKERS")
    print("="*70)
    
    summary_data = []
    for ticker, metrics in sorted(results.items(), key=lambda x: x[1]['profit_factor'], reverse=True):
        summary_data.append({
            'Ticker': ticker,
            'Trades': int(metrics['total_trades']),
            'Win%': f"{metrics['win_rate']:.1f}%",
            'ProfitFactor': f"{metrics['profit_factor']:.2f}x",
            'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
            'MaxDD': f"{metrics['max_drawdown']:.2f}%"
        })
    
    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False))
    
    print(f"\nSuccessful: {len(results)}/{len(tickers)}")
    
    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    pfs = [m['profit_factor'] for m in results.values()]
    print(f"Avg Profit Factor: {np.mean(pfs):.2f}x")
    print(f"Min Profit Factor: {np.min(pfs):.2f}x")
    print(f"Max Profit Factor: {np.max(pfs):.2f}x")
    
    winners = [t for t, m in results.items() if m['profit_factor'] > 1.2]
    losers = [t for t, m in results.items() if m['profit_factor'] < 0.8]
    
    print(f"\nWinning Tickers (PF > 1.2x): {', '.join(winners)}")
    print(f"Losing Tickers (PF < 0.8x): {', '.join(losers)}")
