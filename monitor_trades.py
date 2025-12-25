import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.alpaca_trader import AlpacaTrader
import json
from datetime import datetime


def monitor():
    """Monitor active trades and positions."""
    trader = AlpacaTrader()
    
    if not trader.connect():
        return
    
    print("="*70)
    print("QUANTUM FRACTALS - TRADE MONITOR")
    print("="*70)
    
    # Account info
    account = trader.get_account_info()
    print(f"\nAccount Status:")
    print(f"  Cash: ${account.get('cash', 0):,.2f}")
    print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
    print(f"  Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
    
    # Open positions
    positions = trader.get_positions()
    
    if positions:
        print(f"\nOpen Positions ({len(positions)}):")
        for pos in positions:
            print(f"\n  {pos.symbol}:")
            print(f"    Shares: {pos.qty}")
            print(f"    Entry Price: ${pos.avg_fill_price:.2f}")
            print(f"    Current Price: ${pos.current_price:.2f}")
            print(f"    P&L: ${float(pos.unrealized_pl):,.2f} ({float(pos.unrealized_plpc)*100:.2f}%)")
    else:
        print(f"\nNo open positions")
    
    # Open orders
    orders = trader.get_orders()
    
    if orders:
        print(f"\nOpen Orders ({len(orders)}):")
        for order in orders:
            print(f"\n  Order {order.id}:")
            print(f"    Symbol: {order.symbol}")
            print(f"    Side: {order.side.upper()}")
            print(f"    Quantity: {order.qty}")
            print(f"    Price: ${order.limit_price if order.limit_price else 'Market'}")
            print(f"    Status: {order.status}")
    else:
        print(f"\nNo open orders")
    
    # Trade log
    if os.path.exists('trade_log.json'):
        with open('trade_log.json', 'r') as f:
            trades = json.load(f)
        
        if trades:
            print(f"\nTrade History ({len(trades)} trades):")
            for trade in trades[-5:]:  # Show last 5
                print(f"\n  {trade['timestamp']}")
                print(f"    {trade['side'].upper()} {trade['quantity']} {trade['ticker']} @ ${trade['price']:.2f}")
                print(f"    Order ID: {trade['order_id']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    monitor()
