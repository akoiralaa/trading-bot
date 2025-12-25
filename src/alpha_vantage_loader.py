import pandas as pd
import requests
from typing import Optional


class AlphaVantageLoader:
    """Load historical stock data from Alpha Vantage API."""
    
    def __init__(self, api_key: str, ticker: str = "QQQ"):
        self.api_key = api_key
        self.ticker = ticker
        self.base_url = "https://www.alphavantage.co/query"
    
    def load_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Load historical daily data from Alpha Vantage."""
        print(f"Loading {self.ticker} data from Alpha Vantage...")
        
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": self.ticker,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                print("Error: Could not fetch data. Check API key and rate limits.")
                return pd.DataFrame()
            
            time_series = data["Time Series (Daily)"]
            
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            for col in df.columns:
                df[col] = df[col].astype(float)
            
            print(f"✓ Loaded {len(df)} bars")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
