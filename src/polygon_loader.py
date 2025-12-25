import requests
import pandas as pd
from datetime import datetime


class PolygonLoader:
    """Load historical stock data from Polygon.io (free tier available)."""
    
    def __init__(self, api_key: str, ticker: str = "QQQ"):
        self.api_key = api_key
        self.ticker = ticker
        self.base_url = "https://api.polygon.io"
    
    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical daily data."""
        print(f"Loading {self.ticker} from Polygon.io...")
        
        url = f"{self.base_url}/v1/open-close/{self.ticker}/2024-01-01"
        params = {"apiKey": self.api_key}
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if response.status_code == 200:
                print("✓ Connected to Polygon.io")
                return pd.DataFrame()
            else:
                print(f"Error: {data}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
