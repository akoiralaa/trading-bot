import os
from src.webull_connector import WebullConnector
from src.trading_engine import AutomatedTradingEngine
import pandas as pd
import numpy as np


def create_test_data(num_bars: int = 100) -> pd.DataFrame:
    """Generate test data for demo."""
    np.random.seed(42)
    prices = [420.0]
    for _ in range(num_bars - 1):
        change = np.random.normal(0.001, 0.01)
        prices.append(prices[-1] * (1 + change))
    
    dates = pd.date_range('2019-01-01', periods=num_bars, freq='D')
    return pd.DataFrame({
        'open': prices,
        'high': [p * 1.005 for p in prices],
        'low': [p * 0.995 for p in prices],
        'close': prices,
        'volume': [1000000] * num_bars
    }, index=dates)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("WEBULL AUTOMATED TRADING ENGINE")
    print("="*60)
    
    # Get credentials from user
    username = input("\nEnter Webull username/email: ").strip()
    password = input("Enter Webull password: ").strip()
    mfa = input("Enter 2FA code (press Enter to skip): ").strip() or None
    
    # Connect to Webull
    wb = WebullConnector(username, password, mfa)
    
    if not wb.login():
        print("Failed to connect to Webull")
        exit(1)
    
    # Get account info
    account = wb.get_account_info()
    print("\nAccount Information:")
    print(f"  Account ID: {account.get('account_id')}")
    print(f"  Cash: ${account.get('cash', 0):,.2f}")
    print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
    print(f"  Account Type: {account.get('account_type')}")
    
    # Initialize trading engine
    engine = AutomatedTradingEngine(wb, risk_per_trade=0.01)
    
    # Demo: Analyze QQQ with test data
    print("\n" + "="*60)
    print("ANALYZING QQQ")
    print("="*60)
    
    df = create_test_data(100)
    signal = engine.analyze_ticker('QQQ', df)
    
    print(f"\nSignal: {signal['signal']}")
    print(f"Current Price: ${signal['current_price']:.2f}")
    print(f"Vector Level: ${signal['vector']:.2f}")
    print(f"Stop Loss: ${signal['stop_loss']:.2f}")
    print(f"Target: ${signal['target']:.2f}")
    print(f"Confidence: {signal['confidence']}")
    
    if signal['signal'] == 'BUY':
        print("\n✓ Valid buy signal detected!")
        execute = input("Execute trade? (y/n): ").lower() == 'y'
        
        if execute:
            order_id = engine.execute_trade('QQQ', signal)
            if order_id:
                print(f"\n✓ Trade executed! Order ID: {order_id}")
    else:
        print("\n✗ No valid signal at this time")
    
    print("\n" + "="*60)
    print("Webull integration ready!")
    print("="*60)
