from webull import webull
import json
from typing import Dict, Optional


class WebullConnector:
    """Connect to Webull account for paper trading."""
    
    def __init__(self, username: str, password: str, mfa_code: Optional[str] = None):
        """
        Initialize Webull connection.
        
        Args:
            username: Webull username/email
            password: Webull password
            mfa_code: 2FA code if enabled (optional)
        """
        self.wb = webull()
        self.username = username
        self.password = password
        self.mfa_code = mfa_code
        self.account_id = None
        self.connected = False
    
    def login(self) -> bool:
        """
        Login to Webull account.
        
        Returns:
            True if login successful, False otherwise
        """
        try:
            print(f"Logging into Webull as {self.username}...")
            
            # Login
            self.wb.login(self.username, self.password)
            
            # Handle MFA if provided
            if self.mfa_code:
                self.wb.input_mfa(self.mfa_code)
            
            self.connected = True
            print("✓ Connected to Webull")
            return True
            
        except Exception as e:
            print(f"✗ Login failed: {e}")
            return False
    
    def get_account_info(self) -> Dict:
        """Get Webull account information."""
        try:
            account = self.wb.get_account()
            self.account_id = account.get('accountId')
            
            info = {
                'account_id': self.account_id,
                'cash': account.get('cash'),
                'buying_power': account.get('buyingPower'),
                'portfolio_value': account.get('portfolioValue'),
                'account_type': 'Paper Trading' if 'paper' in str(account).lower() else 'Live'
            }
            
            return info
            
        except Exception as e:
            print(f"Error getting account info: {e}")
            return {}
    
    def get_quote(self, ticker: str) -> Dict:
        """Get current quote for a ticker."""
        try:
            quote = self.wb.get_quote(ticker)
            return {
                'ticker': ticker,
                'price': quote.get('lastPrice'),
                'bid': quote.get('bidPrice'),
                'ask': quote.get('askPrice'),
                'volume': quote.get('volume')
            }
        except Exception as e:
            print(f"Error getting quote for {ticker}: {e}")
            return {}
    
    def place_order(self, ticker: str, price: float, quantity: int, 
                   side: str = 'BUY', order_type: str = 'LIMIT') -> Optional[str]:
        """
        Place a paper trading order.
        
        Args:
            ticker: Stock ticker (e.g., 'AAPL')
            price: Order price
            quantity: Number of shares
            side: 'BUY' or 'SELL'
            order_type: 'LIMIT', 'MARKET', etc.
        
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            print(f"\n{'='*60}")
            print(f"PLACING {side} ORDER")
            print(f"{'='*60}")
            print(f"Ticker: {ticker}")
            print(f"Price: ${price:.2f}")
            print(f"Quantity: {quantity}")
            print(f"Order Type: {order_type}")
            
            # Get current price
            quote = self.get_quote(ticker)
            if quote:
                print(f"Current Price: ${quote['price']:.2f}")
            
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
                print(f"✓ Order placed successfully!")
                print(f"Order ID: {order_id}")
                return order_id
            else:
                print(f"✗ Order failed: {order}")
                return None
                
        except Exception as e:
            print(f"✗ Error placing order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        try:
            self.wb.cancel_order(order_id)
            print(f"✓ Order {order_id} cancelled")
            return True
        except Exception as e:
            print(f"✗ Error cancelling order: {e}")
            return False
    
    def get_positions(self) -> list:
        """Get current open positions."""
        try:
            positions = self.wb.get_position()
            return positions if positions else []
        except Exception as e:
            print(f"Error getting positions: {e}")
            return []
    
    def get_orders(self) -> list:
        """Get open orders."""
        try:
            orders = self.wb.get_orders()
            return orders if orders else []
        except Exception as e:
            print(f"Error getting orders: {e}")
            return []
