import os
from alpaca_trade_api import REST
from datetime import datetime
import json


class AlpacaTrader:
    """Execute paper trades on Alpaca."""
    
    def __init__(self):
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = 'https://paper-api.alpaca.markets'
        self.api = None
        self.connected = False
        self.trade_log = []
    
    def connect(self) -> bool:
        """Connect to Alpaca."""
        try:
            print(f"Connecting with key: {self.api_key[:10]}...")
            self.api = REST(
                key_id=self.api_key,
                secret_key=self.secret_key,
                base_url=self.base_url
            )
            account = self.api.get_account()
            self.connected = True
            print(f"Connected to Alpaca (Paper Trading)")
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def get_account_info(self) -> dict:
        """Get account information."""
        if not self.connected:
            return {}
        
        try:
            account = self.api.get_account()
            return {
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'account_type': 'Paper Trading'
            }
        except Exception as e:
            print(f"Error: {e}")
            return {}
    
    def place_order(self, ticker: str, quantity: int, side: str = 'buy', 
                   order_type: str = 'limit', limit_price: float = None) -> dict:
        """Place a paper trade order."""
        if not self.connected:
            return {}
        
        try:
            print(f"\nPlacing {side.upper()} order:")
            print(f"  Ticker: {ticker}")
            print(f"  Quantity: {quantity}")
            print(f"  Price: ${limit_price:.2f}")
            
            order = self.api.submit_order(
                symbol=ticker,
                qty=quantity,
                side=side,
                type=order_type,
                time_in_force='day',
                limit_price=limit_price if order_type == 'limit' else None
            )
            
            self.trade_log.append({
                'timestamp': datetime.now().isoformat(),
                'ticker': ticker,
                'side': side,
                'quantity': quantity,
                'price': limit_price,
                'order_id': order.id,
                'status': 'PENDING'
            })
            
            print(f"Order {order.id} placed")
            return {'order_id': order.id, 'status': order.status}
            
        except Exception as e:
            print(f"Error placing order: {e}")
            return {}
    
    def get_positions(self) -> list:
        """Get current positions."""
        if not self.connected:
            return []
        
        try:
            return self.api.list_positions()
        except Exception as e:
            print(f"Error: {e}")
            return []
    
    def save_trade_log(self, filename: str = 'trade_log.json'):
        """Save trading log."""
        with open(filename, 'w') as f:
            json.dump(self.trade_log, f, indent=2)
        print(f"Trade log saved to {filename}")
