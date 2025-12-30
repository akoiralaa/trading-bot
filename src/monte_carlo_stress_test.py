import numpy as np
from scipy import stats
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class MonteCarloStressTest:
    """
    Monte Carlo simulation for stress testing trading strategies.
    Creates probability cone of future equity and identifies risk of ruin.
    """
    
    def __init__(self, initial_equity: float = 100000, simulations: int = 10000) -> None:
        self.initial_equity = initial_equity
        self.simulations = simulations
        logger.info(f"MonteCarloStressTest initialized: equity={initial_equity}, sims={simulations}")
    
    def run_probability_cone(self, trade_returns: np.ndarray, confidence_levels: List[int] = None) -> Dict:
        """
        Shuffle trade returns and run Monte Carlo to create probability cone.
        Shows best case, worst case, and median outcome.
        
        Args:
            trade_returns: Array of historical trade returns
            confidence_levels: Percentiles to calculate (default [5, 25, 50, 75, 95])
        
        Returns:
            Dict with probability cone metrics
        """
        if confidence_levels is None:
            confidence_levels = [5, 25, 50, 75, 95]
        
        logger.info(f"Running probability cone: {self.simulations} simulations on {len(trade_returns)} returns")
        
        equity_paths: List = []
        
        for sim_num in range(self.simulations):
            shuffled_returns: np.ndarray = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            equity_curve: np.ndarray = np.cumprod(1 + shuffled_returns) * self.initial_equity
            equity_paths.append(equity_curve)
        
        equity_array: np.ndarray = np.array(equity_paths)
        
        percentiles: Dict = {}
        for conf in confidence_levels:
            percentiles[f'p{conf}'] = np.percentile(equity_array, conf, axis=0)
        
        worst_case: float = np.percentile(equity_array[:, -1], 5)
        best_case: float = np.percentile(equity_array[:, -1], 95)
        median: float = np.percentile(equity_array[:, -1], 50)
        
        result: Dict = {
            'paths': equity_array,
            'final_equity': equity_array[:, -1],
            'percentiles': percentiles,
            'worst_case': worst_case,
            'best_case': best_case,
            'median': median,
            'std_dev': np.std(equity_array[:, -1])
        }
        
        logger.info(f"Probability cone: worst={worst_case:.0f}, median={median:.0f}, best={best_case:.0f}")
        
        return result
    
    def calculate_risk_of_ruin(self, trade_returns: np.ndarray, ruin_threshold: float = 0.20) -> Dict:
        """
        What % of simulations result in account decline > 20%?
        
        Args:
            trade_returns: Array of historical trade returns
            ruin_threshold: Drawdown threshold (default 20%)
        
        Returns:
            Dict with risk of ruin metrics
        """
        logger.info(f"Calculating risk of ruin: {self.simulations} sims, threshold={ruin_threshold*100}%")
        
        equity_paths: List = []
        
        for _ in range(self.simulations):
            shuffled_returns: np.ndarray = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            equity_curve: np.ndarray = np.cumprod(1 + shuffled_returns) * self.initial_equity
            equity_paths.append(equity_curve[-1])
        
        ruin_level: float = self.initial_equity * (1 - ruin_threshold)
        ruin_count: int = sum(1 for eq in equity_paths if eq < ruin_level)
        risk_of_ruin: float = ruin_count / self.simulations
        
        result: Dict = {
            'risk_of_ruin_pct': risk_of_ruin * 100,
            'ruin_threshold': ruin_threshold,
            'simulations_in_ruin': ruin_count,
            'final_equities': equity_paths
        }
        
        logger.warning(f"Risk of Ruin: {risk_of_ruin*100:.2f}% ({ruin_count}/{self.simulations} sims)")
        
        return result
    
    def stress_test_crashes(self, trade_returns: np.ndarray, crash_magnitude: float = -0.10, 
                           crash_injection_pct: float = 0.10) -> Dict:
        """
        Inject artificial market crashes and test survival.
        
        Args:
            trade_returns: Array of historical trade returns
            crash_magnitude: Severity of crash (default -10%)
            crash_injection_pct: Probability of crash injection (default 10%)
        
        Returns:
            Dict with crash stress test results
        """
        logger.info(f"Running crash stress test: magnitude={crash_magnitude*100}%, injection={crash_injection_pct*100}%")
        
        results: List = []
        crash_count: int = 0
        
        for sim in range(self.simulations):
            returns: np.ndarray = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            if np.random.random() < crash_injection_pct:
                crash_idx: int = np.random.randint(0, len(returns))
                returns[crash_idx] = crash_magnitude
                crash_count += 1
            
            equity_curve: np.ndarray = np.cumprod(1 + returns) * self.initial_equity
            results.append({
                'final_equity': equity_curve[-1],
                'min_equity': np.min(equity_curve),
                'max_drawdown': (np.min(equity_curve) - self.initial_equity) / self.initial_equity
            })
        
        final_equities: List = [r['final_equity'] for r in results]
        max_drawdowns: List = [r['max_drawdown'] for r in results]
        
        survival_rate: float = sum(1 for eq in final_equities if eq > self.initial_equity * 0.8) / len(final_equities)
        
        result: Dict = {
            'median_final_equity': np.median(final_equities),
            'worst_final_equity': np.percentile(final_equities, 5),
            'best_final_equity': np.percentile(final_equities, 95),
            'worst_drawdown': np.min(max_drawdowns),
            'avg_drawdown': np.mean(max_drawdowns),
            'survival_rate': survival_rate,
            'crashes_injected': crash_count
        }
        
        logger.info(f"Crash test: survival_rate={survival_rate*100:.2f}%, worst_dd={np.min(max_drawdowns)*100:.2f}%")
        
        return result
    
    def calculate_var_cvar(self, trade_returns: np.ndarray, confidence: float = 0.95) -> Dict:
        """
        Value at Risk & Conditional Value at Risk.
        
        Args:
            trade_returns: Array of historical trade returns
            confidence: Confidence level (default 95%)
        
        Returns:
            Dict with VaR and CVaR metrics
        """
        logger.info(f"Calculating VaR/CVaR: confidence={confidence*100}%")
        
        equity_paths: List = []
        
        for _ in range(self.simulations):
            shuffled: np.ndarray = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            equity: float = np.prod(1 + shuffled) * self.initial_equity
            equity_paths.append(equity)
        
        percentile: float = (1 - confidence) * 100
        var: float = np.percentile(equity_paths, percentile)
        cvar: float = np.mean([eq for eq in equity_paths if eq <= var])
        
        result: Dict = {
            'value_at_risk': var,
            'conditional_var': cvar,
            'confidence_level': confidence,
            'expected_loss': self.initial_equity - cvar,
            'loss_pct': ((self.initial_equity - cvar) / self.initial_equity) * 100
        }
        
        logger.info(f"VaR {confidence*100:.0f}%: {var:.0f}, CVaR: {cvar:.0f}, max loss: {result['loss_pct']:.2f}%")
        
        return result
