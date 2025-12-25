import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


class RealtimeDataLoader:
    """Load real historical stock data from Yahoo Finance."""
    
    def __init__(self, ticker: str = "QQQ"):
        self.ticker = ticker
    
    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load historical OHLCV data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        print(f"Loading {self.ticker} from {start_date} to {end_date}...")
        
        try:
            df = yf.download(
                self.ticker,
                start=start_date,
                end=end_date,
                progress=False,
                timeout=30
            )
            
            # Check if we got data
            if df.empty:
                print(f"Error: No data returned for {self.ticker}")
                return pd.DataFrame()
            
            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Remove adjusted close if present
            if 'adj close' in df.columns:
                df = df.drop('adj close', axis=1)
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print(f"Error: Missing required columns")
                return pd.DataFrame()
            
            print(f"Success: Loaded {len(df)} bars")
            print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def load_latest(self, days: int = 252) -> pd.DataFrame:
        """Load latest N trading days."""
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days * 1.5)
        
        return self.load_data(
            start_date=str(start_date),
            end_date=str(end_date)
        )
