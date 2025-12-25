import pandas as pd
import numpy as np
from src.webull_connector import WebullConnector
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.pattern_detector import PatternDetector
from typing import Dict, Optional


class AutomatedTradingEngine:
    """Automatically execute fractal trades on Webull."""
    
    def __init__(self, wb_connector: WebullConnector, risk_per_trade: float = 0.01):
        """
        Args:
            wb_connector: Connected WebullConnector instance
            risk_per_trade: Risk % per trade (1% = $100 on $10k account)
        """
        self.wb = wb_connector
        self.risk_per_trade = risk_per_trade
        self.vector_calc = VectorCalculator(lookback_period=20)
        self.fractal_detector = FractalDetector(cluster_threshold=0.10)
        self.pattern_detector = PatternDetector()
        
        self.active_trades = {}
    
    def analyze_ticker(self, ticker: str, df: pd.DataFrame) -> Dict:
        """
        Analyze a ticker for trading signals.
        
        Returns:
            Dict with signal, entry price, stop loss, target
        """
        vector = self.vector_calc.calculate_vector(df)
        fractal_highs, fractal_lows = self.fractal_detector.detect_fractals(df)
        
        table_top_b = self.pattern_detector.detect_table_top_b(df, vector)
        table_top_a = self.pattern_detector.detect_table_top_a(df, vector)
        
        # Get latest signal
        latest_idx = len(df) - 1
        has_signal = table_top_b[latest_idx] == 1 or table_top_a[latest_idx] == 1
        
        current_price = df['close'].iloc[-1]
        current_vector = vector[latest_idx]
        
        return {
            'ticker': ticker,
            'signal': 'BUY' if has_signal else 'NONE',
            'current_price': current_price,
            'vector': current_vector,
            'stop_loss': current_vector * 0.98,
            'target': current_vector * 1.02,
            'above_vector': current_price > current_vector,
            'confidence': 'HIGH' if has_signal else 'LOW'
        }
    
    def execute_trade(self, ticker: str, signal: Dict, quantity: Optional[int] = None) -> Optional[str]:
        """
        Execute a trade based on fractal signal.
        
        Args:
            ticker: Stock ticker
            signal: Signal dict from analyze_ticker()
            quantity: Shares to buy (auto-calculate if None)
        
        Returns:
            Order ID if successful
        """
        if signal['signal'] != 'BUY':
            print(f"✗ No valid signal for {ticker}")
            return None
        
        # Get account info
        account = self.wb.get_account_info()
        if not account:
            print("✗ Unable to get account info")
            return None
        
        cash = account.get('cash', 0)
        
        # Calculate position size
        if quantity is None:
            risk_amount = (self.risk_per_trade / 100) * account.get('portfolio_value', cash)
            position_size_dollars = risk_amount
            quantity = int(position_size_dollars / signal['current_price'])
        
        if quantity <= 0:
            print(f"✗ Insufficient funds for {ticker}")
            return None
        
        # Place limit order at vector level (support)
        entry_price = signal['current_price']
        
        order_id = self.wb.place_order(
            ticker=ticker,
            price=entry_price,
            quantity=quantity,
            side='BUY',
            order_type='LIMIT'
        )
        
        if order_id:
            self.active_trades[order_id] = {
                'ticker': ticker,
                'entry_price': entry_price,
                'stop_loss': signal['stop_loss'],
                'target': signal['target'],
                'quantity': quantity
            }
        
        return order_id
