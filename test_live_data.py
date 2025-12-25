import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.alpaca_trader import AlpacaTrader
import datetime


def main():
    print("="*70)
    print("LIVE DATA CONNECTION TEST")
    print("="*70)
    
    trader = AlpacaTrader()
    
    if not trader.connect():
        print("FAILED: Could not connect to Alpaca")
        return
    
    account = trader.get_account_info()
    print(f"\nConnected to Alpaca")
    print(f"Account: ${account['cash']:,.2f} cash")
    print(f"Buying Power: ${account['buying_power']:,.2f}")
    
    print(f"\nLive Prices ({datetime.datetime.now().strftime('%H:%M:%S')}):")
    
    for ticker in ['PLTR', 'QQQ', 'PENN', 'SPY']:
        try:
            quote = trader.api.get_latest_quote(ticker)
            if quote:
                print(f"  {ticker}: ${quote.bid} / ${quote.ask}")
        except:
            print(f"  {ticker}: Unable to fetch")
    
    print(f"\nSystem ready for live trading")


if __name__ == "__main__":
    main()
