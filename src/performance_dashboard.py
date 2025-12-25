import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from typing import Dict, List


class PerformanceDashboard:
    """Generate trading performance analytics and visualizations."""
    
    def __init__(self):
        self.trades = []
    
    def add_trade(self, entry_price: float, exit_price: float, 
                  entry_date: str, exit_date: str, side: str = 'LONG'):
        """Add a completed trade."""
        pnl = exit_price - entry_price if side == 'LONG' else entry_price - exit_price
        pnl_pct = (pnl / entry_price) * 100
        
        self.trades.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'side': side,
            'winner': pnl > 0
        })
    
    def get_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if not self.trades:
            return self._empty_metrics()
        
        df = pd.DataFrame(self.trades)
        winners = df[df['winner']]
        losers = df[~df['winner']]
        
        return {
            'total_trades': len(df),
            'winning_trades': len(winners),
            'losing_trades': len(losers),
            'win_rate': (len(winners) / len(df) * 100) if len(df) > 0 else 0,
            'total_pnl': df['pnl'].sum(),
            'total_pnl_pct': df['pnl_pct'].sum(),
            'avg_win': winners['pnl'].mean() if len(winners) > 0 else 0,
            'avg_loss': losers['pnl'].mean() if len(losers) > 0 else 0,
            'best_trade': df['pnl'].max(),
            'worst_trade': df['pnl'].min(),
            'avg_trade': df['pnl'].mean(),
            'profit_factor': abs(winners['pnl'].sum() / losers['pnl'].sum()) if len(losers) > 0 and losers['pnl'].sum() != 0 else 0
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dict."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_pnl_pct': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'best_trade': 0,
            'worst_trade': 0,
            'avg_trade': 0,
            'profit_factor': 0
        }
    
    def generate_report(self) -> str:
        """Generate text performance report."""
        metrics = self.get_metrics()
        
        report = f"""
{'='*60}
TRADING PERFORMANCE REPORT
{'='*60}

TRADE STATISTICS
  Total Trades:        {metrics['total_trades']}
  Winning Trades:      {metrics['winning_trades']}
  Losing Trades:       {metrics['losing_trades']}
  Win Rate:            {metrics['win_rate']:.2f}%

PROFITABILITY
  Total P&L:           ${metrics['total_pnl']:.2f}
  Total P&L %:         {metrics['total_pnl_pct']:.2f}%
  Avg P&L per Trade:   ${metrics['avg_trade']:.2f}
  Best Trade:          ${metrics['best_trade']:.2f}
  Worst Trade:         ${metrics['worst_trade']:.2f}

RISK/REWARD
  Avg Win:             ${metrics['avg_win']:.2f}
  Avg Loss:            ${metrics['avg_loss']:.2f}
  Profit Factor:       {metrics['profit_factor']:.2f}x

{'='*60}
"""
        return report
    
    def plot_equity_curve(self) -> io.BytesIO:
        """Generate equity curve chart."""
        if not self.trades:
            return None
        
        df = pd.DataFrame(self.trades)
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(range(len(df)), df['cumulative_pnl'], 'b-', linewidth=2, label='Equity Curve')
        ax.fill_between(range(len(df)), 0, df['cumulative_pnl'], alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Cumulative P&L ($)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
    def plot_drawdown(self) -> io.BytesIO:
        """Generate drawdown chart."""
        if not self.trades:
            return None
        
        df = pd.DataFrame(self.trades)
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['running_max'] = df['cumulative_pnl'].expanding().max()
        df['drawdown'] = df['cumulative_pnl'] - df['running_max']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.fill_between(range(len(df)), df['drawdown'], 0, color='red', alpha=0.3, label='Drawdown')
        ax.plot(range(len(df)), df['drawdown'], 'r-', linewidth=2)
        
        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trade Number')
        ax.set_ylabel('Drawdown ($)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
    def plot_win_loss_distribution(self) -> io.BytesIO:
        """Generate win/loss distribution chart."""
        if not self.trades:
            return None
        
        df = pd.DataFrame(self.trades)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Separate wins and losses
        wins = df[df['winner']]['pnl'].values
        losses = df[~df['winner']]['pnl'].values
        
        ax.hist(wins, bins=10, color='green', alpha=0.7, label=f'Wins (n={len(wins)})')
        ax.hist(losses, bins=10, color='red', alpha=0.7, label=f'Losses (n={len(losses)})')
        
        ax.set_title('P&L Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('P&L ($)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
