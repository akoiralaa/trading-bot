import pandas as pd
import pandas_datareader as pdr
from datetime import datetime


class CSVLoader:
    """Load historical stock data."""
    
    def __init__(self, ticker: str = "QQQ"):
        self.ticker = ticker
    
    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical OHLCV data."""
        print(f"Loading {self.ticker} from {start_date} to {end_date}...")
        
        try:
            df = pdr.get_data_yahoo(
                self.ticker,
                start=start_date,
                end=end_date
            )
            
            df.columns = [col.lower() for col in df.columns]
            if 'adj close' in df.columns:
                df = df.drop('adj close', axis=1)
            
            print(f"✓ Loaded {len(df)} bars")
            return df
            
        except Exception as e:
            print(f"Error: {e}")
            return pd.DataFrame()
