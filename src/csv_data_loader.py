import pandas as pd
import os


class CSVDataLoader:
    """Load historical data from local CSV file."""
    
    def __init__(self, ticker: str = "QQQ"):
        self.ticker = ticker
        self.data_dir = "data"
    
    def load_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            start_date: Filter from this date (optional)
            end_date: Filter to this date (optional)
        
        Returns:
            DataFrame with OHLCV data
        """
        csv_path = os.path.join(self.data_dir, f"{self.ticker}.csv")
        
        print(f"Loading {self.ticker} from {csv_path}...")
        
        try:
            df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
            
            df.columns = [col.lower() for col in df.columns]
            
            df = df.sort_index()
            
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
            
            print(f"Success: Loaded {len(df)} bars")
            print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
            
            return df
            
        except FileNotFoundError:
            print(f"Error: CSV file not found at {csv_path}")
            print(f"Please download QQQ data from Yahoo Finance and save to {csv_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return pd.DataFrame()
