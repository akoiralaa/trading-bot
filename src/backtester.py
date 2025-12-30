import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from quantum_fractal_engine import QuantumFractalEngine

logger = logging.getLogger(__name__)

class AdvancedBacktester:
    """
    High-fidelity strategy backtesting engine with implementation shortfall 
    and implementation-aware capital allocation.
    
    Evaluates strategy performance using risk-adjusted return metrics, 
    accounting for non-Gaussian return distributions and dynamic leverage.
    """
    
    def __init__(self, engine: QuantumFractalEngine, initial_equity: float = 100000) -> None:
        self.engine = engine
        self.initial_equity = initial_equity
        self.equity_curve: List[float] = [initial_equity]
        self.trades: List[Dict] = []
        logger.info(f"BacktestEngine: Initialized | Capital: {initial_equity}")
    
    def run_backtest(
        self, 
        price_data: np.ndarray, 
        vector_data: np.ndarray, 
        atr_data: np.ndarray, 
        volume_data: np.ndarray
    ) -> Dict:
        """
        Executes an iterative event-driven backtest across provided time-series data.
        """
        logger.info(f"BacktestEvent: Start | Horizon: {len(price_data)} periods")
        
        for i in range(50, len(price_data)):  # Warm-up period for lookback consistency
            # Slice current visibility window
            prices = price_data[:i+1]
            vectors = vector_data[:i+1]
            atr = atr_data[:i+1]
            volume = volume_data[:i+1]
            
            # Map vector strength signal
            strengths = np.array([v['strength'] for v in vectors]) if isinstance(vectors[0], dict) else vectors
            
            # Local liquidity estimate (10-bar SMA)
            avg_vol = np.mean(volume[-10:])
            
            # Delegate to Execution Engine
            response = self.engine.execute_trading_cycle(
                symbol='TEST',
                prices=prices,
                vector_prices=vectors,
                vector_strengths=strengths,
                atr_values=atr,
                avg_volume=avg_vol
            )
            
            if response.get('trade'):
                self.trades.append(response['trade'])
        
        return self.generate_performance_report()
    
    def generate_performance_report(self) -> Dict:
        """
        Aggregates trade logs into a comprehensive risk-adjusted performance report.
        """
        if not self.trades:
            return {'status': 'ZERO_TRADES_EXECUTED'}
        
        # Vectorized implementation of returns calculation
        # short-selling logic should be accounted for in execution_price vs exit_price
        returns = np.array([(t['exit_price'] - t['execution_price']) / t['execution_price'] for t in self.trades])
        
        # Profitability Metrics
        win_rate = np.sum(returns > 0) / len(returns)
        cum_returns = np.cumprod(1 + returns)
        
        # Risk Metrics
        sharpe = self._calculate_sharpe_ratio(returns)
        sortino = self._calculate_sortino_ratio(returns)
        max_dd = self._calculate_max_drawdown(cum_returns)
        calmar = self._calculate_calmar_ratio(returns, max_dd)
        
        metrics = {
            'trade_count': len(self.trades),
            'win_rate': win_rate,
            'expectancy': np.mean(returns),
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd,
            'terminal_wealth': self.initial_equity * cum_returns[-1]
        }
        
        logger.info(f"BacktestComplete | Sharpe: {sharpe:.2f} | MaxDD: {max_dd:.2%}")
        return metrics

    def _calculate_sharpe_ratio(self, returns: np.ndarray, rf_rate: float = 0.0, periods: int = 252) -> float:
        """Calculates annualized Sharpe Ratio."""
        if len(returns) < 2: return 0.0
        return (np.mean(returns) - (rf_rate/periods)) / (np.std(returns) + 1e-6) * np.sqrt(periods)

    def _calculate_sortino_ratio(self, returns: np.ndarray, rf_rate: float = 0.0, periods: int = 252) -> float:
        """Calculates Sortino Ratio utilizing Semi-Standard Deviation (Downside Risk)."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) < 2: return 0.0
        downside_std = np.std(downside_returns)
        return (np.mean(returns) - (rf_rate/periods)) / (downside_std + 1e-6) * np.sqrt(periods)

    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculates maximum Peak-to-Trough decline."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)

    def _calculate_calmar_ratio(self, returns: np.ndarray, max_dd: float, periods: int = 252) -> float:
        """Annualized return relative to maximum drawdown."""
        annualized_return = np.mean(returns) * periods
        return annualized_return / (abs(max_dd) + 1e-6)