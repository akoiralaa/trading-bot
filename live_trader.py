import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.webull_trader import WebullTrader
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
from src.csv_data_loader import CSVDataLoader
import numpy as np
from datetime import datetime


OPTIMAL_PARAMS = {
    'COIN': {'lookback': 30, 'threshold': 0.12},
    'PLTR': {'lookback': 30, 'threshold': 0.10},
    'PENN': {'lookback': 25, 'threshold': 0.10},
    'QQQ': {'lookback': 20, 'threshold': 0.12},
}


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


def analyze_ticker(ticker: str) -> dict:
    """Analyze a ticker for trading signals."""
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
            'above_vector': current_price > current_vector,
            'strength': vector_strength[latest_idx]
        }
        
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None


def main():
    print("="*70)
    print("QUANTUM FRACTALS - LIVE TRADING BOT")
    print("="*70)
    
    # Initialize trader
    username = input("\nEnter Webull username: ").strip()
    password = input("Enter Webull password: ").strip()
    
    trader = WebullTrader(username, password)
    
    if not trader.connect():
        print("Failed to connect to Webull")
        return
    
    # Get account info
    account = trader.get_account_info()
    print(f"\nAccount Info:")
    print(f"  Cash: ${account.get('cash', 0):,.2f}")
    print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
    print(f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
    
    # Analyze tickers
    print(f"\n" + "="*70)
    print("ANALYZING SIGNALS")
    print("="*70)
    
    tickers = ['COIN', 'PLTR', 'PENN', 'QQQ']
    buy_signals = []
    
    for ticker in tickers:
        signal = analyze_ticker(ticker)
        
        if signal:
            print(f"\n{ticker}:")
            print(f"  Signal: {signal['signal']}")
            print(f"  Price: ${signal['current_price']:.2f}")
            print(f"  Vector: ${signal['vector']:.2f}")
            print(f"  Target: ${signal['target']:.2f}")
            print(f"  Stop Loss: ${signal['stop_loss']:.2f}")
            print(f"  Strength: {signal['strength']:.2f}")
            
            if signal['signal'] == 'BUY':
                buy_signals.append(signal)
    
    # Execute trades
    if buy_signals:
        print(f"\n" + "="*70)
        print(f"FOUND {len(buy_signals)} BUY SIGNALS")
        print("="*70)
        
        for signal in buy_signals:
            print(f"\n{signal['ticker']}:")
            execute = input("  Execute trade? (y/n): ").lower() == 'y'
            
            if execute:
                # Risk 1% per trade
                risk_amount = account['buying_power'] * 0.01
                risk_per_share = signal['current_price'] - signal['stop_loss']
                quantity = int(risk_amount / risk_per_share)
                
                if quantity > 0:
                    order_id = trader.place_order(
                        ticker=signal['ticker'],
                        price=signal['current_price'],
                        quantity=quantity,
                        side='BUY',
                        order_type='LIMIT'
                    )
                    
                    if order_id:
                        print(f"  Order {order_id} placed for {quantity} shares")
                else:
                    print(f"  Insufficient buying power for {signal['ticker']}")
    
    # Save trade log
    trader.save_trade_log()
    
    print(f"\n" + "="*70)
    print("Live trading session complete")
    print("="*70)


if __name__ == "__main__":
    main()
