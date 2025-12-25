import requests
import pandas as pd
from datetime import datetime
import time


class YahooFinanceScraper:
    """Scrape historical data directly from Yahoo Finance."""
    
    def __init__(self, ticker: str = "QQQ"):
        self.ticker = ticker
        self.base_url = "https://finance.yahoo.com/quote"
    
    def load_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load historical data by scraping Yahoo Finance.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Scraping {self.ticker} from {start_date} to {end_date}...")
        
        try:
            # Convert dates to Unix timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            
            # Yahoo Finance URL with timestamps
            url = f"{self.base_url}/{self.ticker}/history?period1={start_ts}&period2={end_ts}"
            
            print(f"Fetching from: {url}")
            
            # Fetch page with headers to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                print(f"Error: Status code {response.status_code}")
                return pd.DataFrame()
            
            # Parse HTML
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the table
            table = soup.find('table')
            
            if not table:
                print("Error: Could not find data table on page")
                print("Note: Yahoo Finance may require JavaScript rendering")
                print("Try manually downloading CSV from Yahoo Finance instead")
                return pd.DataFrame()
            
            # Parse table rows
            rows = table.find_all('tr')[1:]  # Skip header
            
            data = []
            for row in rows:
                cols = row.find_all('td')
                if len(cols) < 5:
                    continue
                
                try:
                    date = cols[0].text.strip()
                    open_price = float(cols[1].text.strip().replace(',', ''))
                    high = float(cols[2].text.strip().replace(',', ''))
                    low = float(cols[3].text.strip().replace(',', ''))
                    close = float(cols[4].text.strip().replace(',', ''))
                    volume = int(cols[5].text.strip().replace(',', ''))
                    
                    data.append({
                        'Date': date,
                        'Open': open_price,
                        'High': high,
                        'Low': low,
                        'Close': close,
                        'Volume': volume
                    })
                except (ValueError, AttributeError):
                    continue
            
            if not data:
                print("Error: No data extracted from table")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df.sort_index()
            
            print(f"Success: Loaded {len(df)} bars")
            print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"Error: {e}")
            print("\nAlternative: Manually download CSV from Yahoo Finance")
            print("1. Visit: https://finance.yahoo.com/quote/QQQ/history")
            print("2. Select date range")
            print("3. Click 'Download' button")
            print("4. Save to data/QQQ.csv")
            return pd.DataFrame()
