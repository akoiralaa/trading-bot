import os
from datetime import datetime
from typing import Dict, Optional, List
import json


class WebullTrader:
    """
    Execute real paper trades on Webull.
    """
    
    def __init__(self, username: str = None, password: str = None):
        """Initialize Webull connection."""
        self.username = username or os.getenv('WEBULL_USERNAME')
        self.password = password or os.getenv('WEBULL_PASSWORD')
        self.wb = None
        self.account_id = None
        self.connected = False
        self.trades = []
        self.trade_log = []
    
    def connect(self) -> bool:
        """Connect to Webull."""
        try:
            from webull import webull
            self.wb = webull()
            
            print(f"Connecting to Webull as {self.username}...")
            
            # Login
            self.wb.login(self.username, self.password)
            
            # Get account
            account = self.wb.get_account()
            self.account_id = account.get('accountId')
            
            self.connected = True
            print("Connected to Webull successfully")
            return True
            
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get account information."""
        if not self.connected or not self.wb:
            return {}
        
        try:
            account = self.wb.get_account()
            return {
                'account_id': account.get('accountId'),
                'cash': float(account.get('cash', 0)),
                'buying_power': float(account.get('buyingPower', 0)),
                'portfolio_value': float(account.get('portfolioValue', 0)),
                'account_type': 'Paper Trading'
            }
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}
    
    def get_quote(self, ticker: str) -> Dict:
        """Get current price quote."""
        if not self.connected or not self.wb:
            return {}
        
        try:
            quote = self.wb.get_quote(ticker)
            return {
                'ticker': ticker,
                'price': float(quote.get('lastPrice', 0)),
                'bid': float(quote.get('bidPrice', 0)),
                'ask': float(quote.get('askPrice', 0)),
                'volume': int(quote.get('volume', 0))
            }
        except Exception as e:
            print(f"Error getting quote for {ticker}: {e}")
            return {}
    
    def place_order(self, ticker: str, price: float, quantity: int, 
                   side: str = 'BUY', order_type: str = 'LIMIT') -> Optional[str]:
        """
        Place paper trading order.
        """
        if not self.connected or not self.wb:
            print("Not connected to Webull")
            return None
        
        try:
            print(f"\nPlacing {side} order:")
            print(f"  Ticker: {ticker}")
            print(f"  Price: ${price:.2f}")
            print(f"  Quantity: {quantity}")
            print(f"  Type: {order_type}")
            
            # Place order
            order = self.wb.place_order(
                stock=ticker,
                price=price,
                quantity=quantity,
                side=side,
                orderType=order_type
            )
            
            order_id = order.get('orderId')
            
            if order_id:
                print(f"Order placed: {order_id}")
                
                # Log trade
                self.trade_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'ticker': ticker,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'order_id': order_id,
                    'status': 'PENDING'
                })
                
                return order_id
            else:
                print(f"Order failed: {order}")
                return None
                
        except Exception as e:
            print(f"Error placing order: {e}")
            return None
    
    def get_orders(self) -> List[Dict]:
        """Get open orders."""
        if not self.connected or not self.wb:
            return []
        
        try:
            orders = self.wb.get_orders()
            return orders if orders else []
        except Exception as e:
            print(f"Error getting orders: {e}")
            return []
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if not self.connected or not self.wb:
            return False
        
        try:
            self.wb.cancel_order(order_id)
            print(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            print(f"Error cancelling order: {e}")
            return False
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        if not self.connected or not self.wb:
            return []
        
        try:
            positions = self.wb.get_position()
            return positions if positions else []
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    def save_trade_log(self, filename: str = 'trade_log.json'):
        """Save trading log to file."""
        with open(filename, 'w') as f:
            json.dump(self.trade_log, f, indent=2)
        print(f"Trade log saved to {filename}")
    
    def load_trade_log(self, filename: str = 'trade_log.json'):
        """Load trading log from file."""
        try:
            with open(filename, 'r') as f:
                self.trade_log = json.load(f)
            print(f"Loaded {len(self.trade_log)} trades from {filename}")
        except FileNotFoundError:
            print(f"Trade log file not found: {filename}")
