import pandas as pd
import yfinance as yf


class CSVLoader:
    """Load historical stock data from Yahoo Finance and save to CSV."""
    
    def __init__(self, ticker: str = "QQQ"):
        self.ticker = ticker
    
    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical OHLCV data from Yahoo Finance."""
        print(f"Loading {self.ticker} from Yahoo Finance...")
        
        try:
            df = yf.download(
                self.ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure we have the right columns
            if 'adj close' in df.columns:
                df = df.drop('adj close', axis=1)
            
            print(f"✓ Loaded {len(df)} bars")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
