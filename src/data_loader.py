import pandas as pd
import yfinance as yf


class DataLoader:
    """Load historical stock data."""
    
    def __init__(self, ticker: str = "QQQ"):
        self.ticker = ticker
    
    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical OHLCV data from Yahoo Finance."""
        print(f"Loading {self.ticker} from {start_date} to {end_date}...")
        data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
        data.columns = [col.lower() for col in data.columns]
        print(f"✓ Loaded {len(data)} bars")
        return data
