"""
Dynamic Slippage Model for Realistic Execution Cost Modeling

Replaces flat commission with microstructure-aware slippage that scales with:
1. Bid-ask spread (instrument volatility)
2. Position size relative to average volume
3. Market impact (walking the order book)

This is what Jane Street expects to see - not flat 0.1% fees everywhere.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict


class SlippageModel:
    """
    Models realistic execution costs based on market microstructure.
    
    Key principle: Larger orders relative to volume incur higher costs.
    This is why risk management matters - bigger positions cost more to exit.
    """
    
    def __init__(self, 
                 base_spread_multiplier: float = 0.5,
                 volume_threshold: float = 0.2,
                 basis_point_scaler: float = 10.0):
        """
        Args:
            base_spread_multiplier: Spread = base_spread_multiplier * ATR
                                    0.5 = tight; 1.0 = normal; 2.0 = wide
            volume_threshold: Position size % of 7-bar avg volume before impact
                             0.2 = 20% threshold (typical for limit orders)
            basis_point_scaler: How much price worsens per unit over threshold
                               10.0 = 10 basis points per 0.1 volume ratio
        """
        self.base_spread_multiplier = base_spread_multiplier
        self.volume_threshold = volume_threshold
        self.basis_point_scaler = basis_point_scaler
    
    def calculate_true_range(self, 
                            high: np.ndarray, 
                            low: np.ndarray, 
                            close: np.ndarray) -> np.ndarray:
        """
        True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        
        This is the volatility measure we use for spread sizing.
        Wider spreads in volatile markets, tighter in calm markets.
        """
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # Handle first bar
        
        return tr
    
    def calculate_atr(self, 
                     high: np.ndarray, 
                     low: np.ndarray, 
                     close: np.ndarray,
                     period: int = 14) -> np.ndarray:
        """
        Average True Range - moving average of true range
        Represents typical volatility / spread width
        """
        tr = self.calculate_true_range(high, low, close)
        atr = pd.Series(tr).rolling(period).mean().values
        return atr
    
    def calculate_execution_slippage(self,
                                     close_price: float,
                                     order_size: float,
                                     atr: float,
                                     avg_volume_7bar: float,
                                     side: str = 'buy') -> Tuple[float, float, Dict]:
        """
        Calculate bid and ask prices accounting for market impact.
        
        Args:
            close_price: Current close price
            order_size: Number of shares/contracts we want to trade
            atr: Average True Range at this point
            avg_volume_7bar: 7-bar average volume at this point
            side: 'buy' or 'sell'
        
        Returns:
            bid: Price we can sell at
            ask: Price we can buy at
            details: Dict with breakdown of slippage components
        """
        
        # Step 1: Base spread from ATR (instrument volatility)
        base_spread = self.base_spread_multiplier * atr
        
        # Step 2: Calculate volume impact (walking the book)
        volume_ratio = order_size / avg_volume_7bar if avg_volume_7bar > 0 else 0
        
        # If order is small relative to volume, use base spread
        # If order is large, spread widens significantly
        impact_spread = 0
        if volume_ratio > self.volume_threshold:
            # For every unit over threshold, widen spread by basis_point_scaler bps
            excess_ratio = volume_ratio - self.volume_threshold
            impact_basis_points = excess_ratio * self.basis_point_scaler
            impact_spread = close_price * (impact_basis_points / 10000)
        
        # Step 3: Total spread
        total_spread = base_spread + impact_spread
        
        # Step 4: Execution prices (asymmetric for buy/sell)
        bid = close_price - (total_spread / 2)
        ask = close_price + (total_spread / 2)
        
        # Step 5: Additional slippage for execution timing
        # Market orders get worse fills than limit orders
        execution_slippage = impact_spread * 0.5  # Conservative estimate
        
        if side == 'buy':
            final_execution_price = ask + execution_slippage
        else:
            final_execution_price = bid - execution_slippage
        
        # Details for transparency
        details = {
            'close_price': close_price,
            'base_spread_bps': (base_spread / close_price) * 10000,
            'volume_ratio': volume_ratio,
            'impact_spread_bps': (impact_spread / close_price) * 10000 if close_price > 0 else 0,
            'total_spread_bps': (total_spread / close_price) * 10000 if close_price > 0 else 0,
            'execution_slippage_bps': (execution_slippage / close_price) * 10000 if close_price > 0 else 0,
            'bid': bid,
            'ask': ask,
            'final_execution_price': final_execution_price,
            'side': side
        }
        
        return bid, ask, details
    
    def entry_execution_price(self,
                             close_price: float,
                             order_size: float,
                             atr: float,
                             avg_volume_7bar: float) -> Tuple[float, Dict]:
        """
        Entry price we pay when going long
        We buy at the ask (worse for us)
        """
        bid, ask, details = self.calculate_execution_slippage(
            close_price, order_size, atr, avg_volume_7bar, side='buy'
        )
        return details['final_execution_price'], details
    
    def exit_execution_price(self,
                            close_price: float,
                            order_size: float,
                            atr: float,
                            avg_volume_7bar: float) -> Tuple[float, Dict]:
        """
        Exit price we receive when closing position
        We sell at the bid (worse for us)
        """
        bid, ask, details = self.calculate_execution_slippage(
            close_price, order_size, atr, avg_volume_7bar, side='sell'
        )
        return details['final_execution_price'], details
    
    def round_trip_cost_bps(self,
                           close_price: float,
                           order_size: float,
                           atr: float,
                           avg_volume_7bar: float) -> float:
        """
        Total cost of entering AND exiting a position in basis points
        
        This is critical for position sizing decisions:
        If round trip costs 50bps, a +1% move only gives 0.5% profit
        """
        entry_price, entry_details = self.entry_execution_price(
            close_price, order_size, atr, avg_volume_7bar
        )
        exit_price, exit_details = self.exit_execution_price(
            close_price, order_size, atr, avg_volume_7bar
        )
        
        # Theoretical break-even price
        entry_cost_bps = entry_details['execution_slippage_bps'] + (entry_details['total_spread_bps'] / 2)
        exit_cost_bps = exit_details['execution_slippage_bps'] + (exit_details['total_spread_bps'] / 2)
        
        return entry_cost_bps + exit_cost_bps


class BacktesterWithSlippage:
    """
    Updated backtester that uses dynamic slippage model instead of flat commission
    
    This should replace the commission-based logic in your existing backtester.py
    """
    
    def __init__(self, 
                 initial_capital: float = 100000,
                 base_spread_multiplier: float = 0.5,
                 volume_threshold: float = 0.2):
        
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.slippage_model = SlippageModel(
            base_spread_multiplier=base_spread_multiplier,
            volume_threshold=volume_threshold
        )
        
        self.trades = []
        self.positions = []
        self.equity_curve = []
        self.slippage_costs = []
    
    def calculate_position_metrics(self,
                                   prices: pd.DataFrame,
                                   signals: pd.Series,
                                   position_size: float = None) -> Dict:
        """
        Run backtest with realistic slippage
        
        Args:
            prices: DataFrame with columns ['close', 'high', 'low', 'volume']
            signals: Series of 1 (long), -1 (short), 0 (no trade)
            position_size: If None, use 100 shares. If callable, use dynamic sizing
        
        Returns:
            metrics: Dict with P&L, slippage costs, etc
        """
        
        high = prices['high'].values
        low = prices['low'].values
        close = prices['close'].values
        volume = prices['volume'].values
        
        # Calculate ATR for spread sizing
        atr = self.slippage_model.calculate_atr(high, low, close)
        
        # Calculate 7-bar average volume
        avg_volume_7bar = pd.Series(volume).rolling(7).mean().values
        
        current_position = 0
        entry_price = None
        entry_bar = None
        pnl = 0
        
        for bar in range(1, len(close)):
            signal = signals.iloc[bar] if hasattr(signals, 'iloc') else signals[bar]
            
            # Determine position size
            if callable(position_size):
                size = position_size(bar, self.capital)
            else:
                size = position_size or 100
            
            # Entry
            if signal == 1 and current_position == 0:
                entry_price, entry_details = self.slippage_model.entry_execution_price(
                    close[bar], size, atr[bar], avg_volume_7bar[bar]
                )
                current_position = size
                entry_bar = bar
                
                self.slippage_costs.append(entry_details['execution_slippage_bps'])
            
            # Exit
            elif signal == 0 and current_position > 0:
                exit_price, exit_details = self.slippage_model.exit_execution_price(
                    close[bar], current_position, atr[bar], avg_volume_7bar[bar]
                )
                
                trade_pnl = (exit_price - entry_price) * current_position
                pnl += trade_pnl
                
                self.trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': bar,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'size': current_position,
                    'pnl': trade_pnl,
                    'entry_slippage_bps': entry_details['execution_slippage_bps'],
                    'exit_slippage_bps': exit_details['execution_slippage_bps'],
                })
                
                self.slippage_costs.append(exit_details['execution_slippage_bps'])
                current_position = 0
            
            # Track equity
            self.equity_curve.append(self.initial_capital + pnl)
        
        return {
            'total_pnl': pnl,
            'total_trades': len(self.trades),
            'avg_slippage_bps': np.mean(self.slippage_costs) if self.slippage_costs else 0,
            'total_slippage_cost': np.sum(self.slippage_costs) * self.initial_capital / 10000,
            'final_equity': self.initial_capital + pnl,
            'return_pct': (pnl / self.initial_capital) * 100,
        }


# Example usage showing how this integrates
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    prices = 100 + np.random.randn(252).cumsum()
    volumes = np.random.uniform(1000000, 2000000, 252)
    
    df = pd.DataFrame({
        'close': prices,
        'high': prices + np.abs(np.random.randn(252)),
        'low': prices - np.abs(np.random.randn(252)),
        'volume': volumes,
    })
    
    # Create slippage model
    slippage = SlippageModel(base_spread_multiplier=0.5, volume_threshold=0.2)
    
    # Calculate execution prices for a 5000 share order
    atr_value = df['high'].rolling(14).apply(
        lambda x: np.mean(np.diff(x))
    ).values[-1]
    
    avg_vol = df['volume'].rolling(7).mean().values[-1]
    
    entry_price, entry_details = slippage.entry_execution_price(
        close_price=df['close'].iloc[-1],
        order_size=5000,
        atr=atr_value,
        avg_volume_7bar=avg_vol
    )
    
    print(f"Current price: ${df['close'].iloc[-1]:.2f}")
    print(f"Entry price (buying): ${entry_price:.2f}")
    print(f"Slippage cost: {entry_details['execution_slippage_bps']:.2f} bps")
    print(f"Volume ratio: {entry_details['volume_ratio']:.2%}")