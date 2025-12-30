import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

from quantum_fractal_engine import QuantumFractalEngine

logger = logging.getLogger(__name__)


class AdvancedBacktester:
    """
    Backtester with institutional-grade metrics and friction modeling.
    """
    
    def __init__(self, engine: QuantumFractalEngine, initial_equity: float = 100000) -> None:
        self.engine = engine
        self.initial_equity = initial_equity
        self.equity_curve: List[float] = [initial_equity]
        self.trades: List[Dict] = []
        logger.info(f"AdvancedBacktester initialized: initial_equity={initial_equity}")
    
    def run_backtest(self, price_data: np.ndarray, vector_data: np.ndarray, 
                    atr_data: np.ndarray, volume_data: np.ndarray) -> Dict:
        """
        Run full backtest with market friction and dynamic Kelly sizing.
        
        Args:
            price_data: Array of prices
            vector_data: Array of vector line prices
            atr_data: Array of ATR values
            volume_data: Array of volumes
        
        Returns:
            Dict with backtest summary
        """
        logger.info(f"=== BACKTEST START: {len(price_data)} bars ===")
        
        for i in range(len(price_data)):
            prices: np.ndarray = price_data[:i+1]
            vectors: np.ndarray = vector_data[:i+1]
            strengths: np.ndarray = np.array([v['strength'] for v in vectors]) if isinstance(vectors[0], dict) else vectors
            atr: np.ndarray = atr_data[:i+1]
            volume: np.ndarray = volume_data[:i+1]
            
            if len(prices) < 50:
                continue
            
            avg_vol: float = np.mean(volume[-10:])
            
            result: Dict = self.engine.execute_trading_cycle(
                symbol='TEST',
                prices=prices,
                vector_prices=vectors,
                vector_strengths=strengths,
                atr_values=atr,
                avg_volume=avg_vol
            )
            
            if result['trade']:
                trade: Dict = result['trade']
                self.trades.append(trade)
                logger.debug(f"Trade {len(self.trades)}: {trade['symbol']} {trade['qty']} @ ${trade['execution_price']:.2f}")
        
        logger.info(f"=== BACKTEST COMPLETE: {len(self.trades)} trades ===")
        
        return self.summarize_backtest()
    
    def summarize_backtest(self) -> Dict:
        """
        Generate professional backtest report.
        
        Returns:
            Dict with comprehensive backtest metrics
        """
        if not self.trades:
            logger.error("No trades executed during backtest")
            return {'error': 'No trades executed'}
        
        returns: List[float] = []
        for trade in self.trades:
            entry: float = trade['execution_price']
            target: float = trade['target_price']
            ret: float = (target - entry) / entry
            returns.append(ret)
        
        returns_array: np.ndarray = np.array(returns)
        
        win_count: int = sum(1 for r in returns if r > 0)
        win_rate: float = win_count / len(returns)
        avg_return: float = np.mean(returns)
        std_return: float = np.std(returns)
        sharpe: float = (avg_return / (std_return + 0.0001)) * np.sqrt(252)
        
        # Sortino Ratio (only penalizes downside volatility)
        downside_returns: List[float] = [r for r in returns if r < 0]
        downside_std: float = np.std(downside_returns) if downside_returns else 0
        sortino: float = (avg_return / (downside_std + 0.0001)) * np.sqrt(252)
        
        # Max Drawdown
        cumulative_returns: np.ndarray = np.cumprod(1 + returns_array)
        running_max: np.ndarray = np.maximum.accumulate(cumulative_returns)
        drawdown: np.ndarray = (cumulative_returns - running_max) / running_max
        max_drawdown: float = np.min(drawdown)
        
        result: Dict = {
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'win_count': win_count,
            'avg_return': avg_return,
            'std_return': std_return,
            'sharpe': sharpe,
            'sortino': sortino,
            'max_drawdown': max_drawdown,
            'trades': self.trades
        }
        
        logger.info(f"Backtest Summary:")
        logger.info(f"  Trades: {result['total_trades']}")
        logger.info(f"  Win Rate: {result['win_rate']*100:.2f}%")
        logger.info(f"  Avg Return: {result['avg_return']*100:.2f}%")
        logger.info(f"  Sharpe: {result['sharpe']:.2f}")
        logger.info(f"  Sortino: {result['sortino']:.2f}")
        logger.info(f"  Max Drawdown: {result['max_drawdown']*100:.2f}%")
        
        return result
    
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino Ratio (downside-adjusted return).
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate (default 2%)
        
        Returns:
            Sortino Ratio
        """
        excess_returns: np.ndarray = returns - (risk_free_rate / 252)
        downside_returns: np.ndarray = np.minimum(excess_returns, 0)
        downside_std: float = np.std(downside_returns)
        
        sortino: float = (np.mean(excess_returns) / (downside_std + 0.0001)) * np.sqrt(252)
        
        logger.debug(f"Sortino Ratio: {sortino:.2f}")
        
        return sortino
    
    def calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Calmar Ratio (return / max drawdown).
        
        Args:
            returns: Array of returns
        
        Returns:
            Calmar Ratio
        """
        annual_return: float = np.mean(returns) * 252
        
        cumulative_returns: np.ndarray = np.cumprod(1 + returns)
        running_max: np.ndarray = np.maximum.accumulate(cumulative_returns)
        drawdown: np.ndarray = (cumulative_returns - running_max) / running_max
        max_drawdown: float = abs(np.min(drawdown))
        
        calmar: float = annual_return / (max_drawdown + 0.0001)
        
        logger.debug(f"Calmar Ratio: {calmar:.2f}")
        
        return calmar
