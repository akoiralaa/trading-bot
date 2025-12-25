import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class Trade:
    """Represents a single trade."""
    
    def __init__(self, entry_idx: int, entry_price: float, stop_loss: float, target: float, side: str):
        self.entry_idx = entry_idx
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.target = target
        self.side = side  # 'long' or 'short'
        self.exit_idx = None
        self.exit_price = None
        self.pnl = None
        self.pnl_pct = None
    
    def close(self, exit_idx: int, exit_price: float):
        """Close the trade at exit price."""
        self.exit_idx = exit_idx
        self.exit_price = exit_price
        
        if self.side == 'long':
            self.pnl = exit_price - self.entry_price
            self.pnl_pct = (self.pnl / self.entry_price) * 100
        else:
            self.pnl = self.entry_price - exit_price
            self.pnl_pct = (self.pnl / self.entry_price) * 100


class Backtester:
    """
    Backtest the RCG Fractal Trading System on historical data.
    """
    
    def __init__(self, risk_per_trade: float = 0.01, position_size: float = 1.0):
        """
        Args:
            risk_per_trade: Risk % per trade (1% = $100 on $10k account)
            position_size: Multiplier for position sizing
        """
        self.risk_per_trade = risk_per_trade
        self.position_size = position_size
        self.trades = []
    
    def run_backtest(self, df: pd.DataFrame, vector: np.ndarray, 
                     entry_signals: np.ndarray, stop_losses: np.ndarray, 
                     targets: np.ndarray) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            vector: Vector line values
            entry_signals: Array of entry signals (1 = enter, 0 = nothing)
            stop_losses: Array of stop loss prices for each bar
            targets: Array of target prices for each bar
        
        Returns:
            Dictionary with backtest results
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        self.trades = []
        active_trade = None
        
        for i in range(len(df)):
            # Check if we have an active trade
            if active_trade:
                # Check stop loss
                if low[i] <= active_trade.stop_loss:
                    active_trade.close(i, active_trade.stop_loss)
                    self.trades.append(active_trade)
                    active_trade = None
                    continue
                
                # Check target
                if high[i] >= active_trade.target:
                    active_trade.close(i, active_trade.target)
                    self.trades.append(active_trade)
                    active_trade = None
                    continue
            
            # Check for new entry signal
            if entry_signals[i] == 1 and not active_trade:
                entry_price = close[i]
                stop = stop_losses[i]
                target = targets[i]
                
                # Only enter if stop loss is reasonable (not too far)
                if stop > 0 and target > 0:
                    active_trade = Trade(
                        entry_idx=i,
                        entry_price=entry_price,
                        stop_loss=stop,
                        target=target,
                        side='long'
                    )
        
        # Close any remaining open trade
        if active_trade:
            active_trade.close(len(df) - 1, close[-1])
            self.trades.append(active_trade)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics."""
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl_per_trade': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
        
        pnls = [t.pnl for t in self.trades if t.pnl is not None]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': (len(winning) / len(self.trades) * 100) if self.trades else 0,
            'total_pnl': sum(pnls),
            'avg_pnl_per_trade': np.mean(pnls) if pnls else 0,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0
        }
    
    def print_results(self, metrics: Dict):
        """Print backtest results."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning: {metrics['winning_trades']} | Losing: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"Avg P&L per Trade: ${metrics['avg_pnl_per_trade']:.2f}")
        print(f"Best Trade: ${metrics['best_trade']:.2f}")
        print(f"Worst Trade: ${metrics['worst_trade']:.2f}")
        print("="*60)
