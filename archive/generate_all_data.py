import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_ticker_data(ticker: str, start_price: float):
    """Generate realistic historical data for any ticker."""
    
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 1)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    np.random.seed(hash(ticker) % 2**32)  # Different seed per ticker
    
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
    
    print(f"Generated {ticker}: {len(df)} bars")
    print(f"  Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
    return df


if __name__ == "__main__":
    print("="*60)
    print("GENERATING TICKER DATA")
    print("="*60)
    
    generate_ticker_data("QQQ", 200.0)
    generate_ticker_data("SPY", 320.0)
    generate_ticker_data("NVDA", 70.0)
    
    print("\nAll data generated in data/ folder")
