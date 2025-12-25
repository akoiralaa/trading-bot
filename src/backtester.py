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
        self.side = side
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
    Backtest the Quantum Fractals Trading System.
    """
    
    def __init__(self, risk_per_trade: float = 0.01):
        self.risk_per_trade = risk_per_trade
        self.trades = []
    
    def run_backtest(self, df: pd.DataFrame, vector: np.ndarray, 
                     entry_signals: np.ndarray, stop_losses: np.ndarray, 
                     targets: np.ndarray) -> Dict:
        """
        Run backtest using cluster-based targets.
        
        Args:
            df: DataFrame with OHLCV data
            vector: Vector line values
            entry_signals: Array of entry signals
            stop_losses: Stop loss prices
            targets: Cluster-based target prices
        
        Returns:
            Dictionary with backtest metrics
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
                
                # Only enter if setup is valid (target above entry, stop below)
                if stop > 0 and target > entry_price > stop:
                    active_trade = Trade(
                        entry_idx=i,
                        entry_price=entry_price,
                        stop_loss=stop,
                        target=target,
                        side='long'
                    )
        
        # Close any remaining open trade at last price
        if active_trade:
            active_trade.close(len(df) - 1, close[-1])
            self.trades.append(active_trade)
        
        return self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics."""
        if len(self.trades) == 0:
            return self._empty_metrics()
        
        pnls = [t.pnl for t in self.trades if t.pnl is not None]
        pnls_pct = [t.pnl_pct for t in self.trades if t.pnl_pct is not None]
        winning = [p for p in pnls if p > 0]
        losing = [p for p in pnls if p <= 0]
        
        total_pnl = sum(pnls)
        
        # Calculate risk/reward ratio for each trade
        rr_ratios = []
        for trade in self.trades:
            if trade.pnl is not None:
                risk = trade.entry_price - trade.stop_loss
                reward = trade.target - trade.entry_price
                if risk > 0:
                    rr_ratios.append(reward / risk)
        
        return {
            'total_trades': len(self.trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': (len(winning) / len(self.trades) * 100) if self.trades else 0,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': np.mean(pnls) if pnls else 0,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'avg_win': np.mean(winning) if winning else 0,
            'avg_loss': np.mean(losing) if losing else 0,
            'avg_rr_ratio': np.mean(rr_ratios) if rr_ratios else 0,
            'profit_factor': abs(sum(winning) / sum(losing)) if losing and sum(losing) != 0 else 0
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dict."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl_per_trade': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_rr_ratio': 0,
            'profit_factor': 0
        }
    
    def print_results(self, metrics: Dict):
        """Print backtest results."""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Winning: {metrics['winning_trades']} | Losing: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"\nProfitability")
        print(f"Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"Avg P&L per Trade: ${metrics['avg_pnl_per_trade']:.2f}")
        print(f"Best Trade: ${metrics['best_trade']:.2f}")
        print(f"Worst Trade: ${metrics['worst_trade']:.2f}")
        print(f"\nRisk/Reward")
        print(f"Avg Win: ${metrics['avg_win']:.2f}")
        print(f"Avg Loss: ${metrics['avg_loss']:.2f}")
        print(f"Avg RR Ratio: {metrics['avg_rr_ratio']:.2f}:1")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}x")
        print("="*60)
