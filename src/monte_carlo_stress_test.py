import numpy as np
from scipy import stats
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class MonteCarloStressTest:
    """
    Stochastic risk engine for strategy validation and tail-risk quantification.
    
    Utilizes bootstrap resampling to generate multi-path equity distributions, 
    quantifying the 'Risk of Ruin' and 'Conditional Value at Risk' (CVaR) under 
    non-Gaussian return assumptions.
    """
    
    def __init__(self, initial_equity: float = 100000, simulations: int = 10000) -> None:
        """
        Initializes the simulator with specified capital and iteration depth.
        
        Args:
            initial_equity: Starting capital for path calculation.
            simulations: Number of stochastic paths to generate ($N \geq 10,000$ recommended).
        """
        self.initial_equity = initial_equity
        self.simulations = simulations
        logger.info(f"RiskEngine: Equity={initial_equity}, Iterations={simulations}")
    
    def run_probability_cone(self, trade_returns: np.ndarray, confidence_levels: Optional[List[int]] = None) -> Dict:
        """
        Generates a fan chart of potential equity trajectories via bootstrap sampling.
        
        Analyzes path dependency by aggregating the cumulative product of 
        shuffled return vectors across $N$ simulations.
        """
        if confidence_levels is None:
            confidence_levels = [5, 25, 50, 75, 95]
        
        # Vectorized bootstrapping of return paths
        # Returns shape: (simulations, trade_count)
        shuffled_indices = np.random.randint(0, len(trade_returns), size=(self.simulations, len(trade_returns)))
        path_returns = trade_returns[shuffled_indices]
        
        # Cumulative product for equity trajectory: E_t = E_0 * prod(1 + r_i)
        equity_array = np.cumprod(1 + path_returns, axis=1) * self.initial_equity
        
        percentiles = {f'p{conf}': np.percentile(equity_array, conf, axis=0) for conf in confidence_levels}
        final_dist = equity_array[:, -1]
        
        return {
            'paths': equity_array,
            'final_equity_dist': final_dist,
            'percentiles': percentiles,
            'p5_worst_case': np.percentile(final_dist, 5),
            'p50_median': np.percentile(final_dist, 50),
            'p95_best_case': np.percentile(final_dist, 95),
            'terminal_std': np.std(final_dist)
        }
    
    def calculate_risk_of_ruin(self, trade_returns: np.ndarray, ruin_threshold: float = 0.20) -> Dict:
        """
        Quantifies the probability of crossing a terminal drawdown threshold.
        
        Calculates $P(E_{final} < E_0 \cdot (1 - \theta))$ where $\theta$ is the 
        ruin threshold (e.g., 0.20 for a 20% total account impairment).
        """
        shuffled_indices = np.random.randint(0, len(trade_returns), size=(self.simulations, len(trade_returns)))
        final_equities = np.prod(1 + trade_returns[shuffled_indices], axis=1) * self.initial_equity
        
        ruin_level = self.initial_equity * (1 - ruin_threshold)
        ruin_count = np.sum(final_equities < ruin_level)
        ror = ruin_count / self.simulations
        
        logger.warning(f"TailRisk: RiskOfRuin={ror*100:.2f}% | Threshold={ruin_threshold}")
        
        return {
            'risk_of_ruin_pct': ror * 100,
            'threshold_value': ruin_level,
            'failure_count': ruin_count
        }

    

    def stress_test_shocks(self, trade_returns: np.ndarray, shock_mag: float = -0.10, 
                          shock_prob: float = 0.10) -> Dict:
        """
        Performs kurtosis injection to simulate 'Black Swan' events.
        
        Artificially introduces low-probability, high-severity negative returns 
        into the return distribution to test system robustness against tail events.
        """
        final_equities = []
        max_dds = []
        
        for _ in range(self.simulations):
            path = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            # Stochastic shock injection
            if np.random.random() < shock_prob:
                path[np.random.randint(0, len(path))] = shock_mag
                
            equity_curve = np.cumprod(1 + path) * self.initial_equity
            final_equities.append(equity_curve[-1])
            max_dds.append(np.min(equity_curve / self.initial_equity) - 1)
            
        return {
            'shock_survival_rate': np.sum(np.array(final_equities) > (self.initial_equity * 0.8)) / self.simulations,
            'median_drawdown': np.median(max_dds),
            'tail_drawdown_p5': np.percentile(max_dds, 5)
        }

    def get_tail_risk_metrics(self, trade_returns: np.ndarray, alpha: float = 0.95) -> Dict:
        """
        Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR).
        
        CVaR (Expected Shortfall) provides the average loss in the $(1-\alpha)$ 
        worst cases, offering a superior risk measure for non-normal distributions.
        """
        shuffled_indices = np.random.randint(0, len(trade_returns), size=(self.simulations, len(trade_returns)))
        final_dist = np.prod(1 + trade_returns[shuffled_indices], axis=1) * self.initial_equity
        
        # VaR is the (1-alpha) percentile of the final equity distribution
        var_threshold = np.percentile(final_dist, (1 - alpha) * 100)
        
        # CVaR is the expectation of the distribution below the VaR threshold
        cvar = final_dist[final_dist <= var_threshold].mean()
        
        return {
            'var_alpha': var_threshold,
            'cvar_expected_shortfall': cvar,
            'max_expected_loss_pct': ((self.initial_equity - cvar) / self.initial_equity) * 100
        }