import numpy as np
import pandas as pd
from quantum_fractal_engine import QuantumFractalEngine


class AdvancedBacktester:
    """
    Backtester with institutional-grade metrics and friction modeling.
    """
    
    def __init__(self, engine, initial_equity=100000):
        self.engine = engine
        self.initial_equity = initial_equity
        self.equity_curve = [initial_equity]
        self.trades = []
    
    def run_backtest(self, price_data, vector_data, atr_data, volume_data):
        """
        Run full backtest with market friction and dynamic Kelly sizing.
        """
        for i in range(len(price_data)):
            prices = price_data[:i+1]
            vectors = vector_data[:i+1]
            strengths = np.array([v['strength'] for v in vectors])
            atr = atr_data[:i+1]
            volume = volume_data[:i+1]
            
            if len(prices) < 50:
                continue
            
            avg_vol = np.mean(volume[-10:])
            
            result = self.engine.execute_trading_cycle(
                symbol='TEST',
                prices=prices,
                vector_prices=vectors,
                vector_strengths=strengths,
                atr_values=atr,
                avg_volume=avg_vol
            )
            
            if result['trade']:
                trade = result['trade']
                self.trades.append(trade)
        
        return self.summarize_backtest()
    
    def summarize_backtest(self):
        """
        Generate professional backtest report.
        """
        if not self.trades:
            return {'error': 'No trades executed'}
        
        returns = []
        for trade in self.trades:
            entry = trade['execution_price']
            target = trade['target_price']
            ret = (target - entry) / entry
            returns.append(ret)
        
        return {
            'total_trades': len(self.trades),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns),
            'avg_return': np.mean(returns),
            'sharpe': np.mean(returns) / (np.std(returns) + 0.0001) * np.sqrt(252),
            'trades': self.trades
        }
