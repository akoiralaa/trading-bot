import numpy as np
from scipy import stats


class MonteCarloStressTest:
    """
    Monte Carlo simulation for stress testing trading strategies.
    Creates probability cone of future equity and identifies risk of ruin.
    """
    
    def __init__(self, initial_equity=100000, simulations=10000):
        self.initial_equity = initial_equity
        self.simulations = simulations
    
    def run_probability_cone(self, trade_returns, confidence_levels=[5, 25, 50, 75, 95]):
        """
        Shuffle trade returns and run Monte Carlo to create probability cone.
        Shows best case, worst case, and median outcome.
        """
        equity_paths = []
        
        for _ in range(self.simulations):
            shuffled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            equity_curve = np.cumprod(1 + shuffled_returns) * self.initial_equity
            equity_paths.append(equity_curve)
        
        equity_array = np.array(equity_paths)
        
        percentiles = {}
        for conf in confidence_levels:
            percentiles[f'p{conf}'] = np.percentile(equity_array, conf, axis=0)
        
        return {
            'paths': equity_array,
            'final_equity': equity_array[:, -1],
            'percentiles': percentiles,
            'worst_case': np.percentile(equity_array[:, -1], 5),
            'best_case': np.percentile(equity_array[:, -1], 95),
            'median': np.percentile(equity_array[:, -1], 50),
            'std_dev': np.std(equity_array[:, -1])
        }
    
    def calculate_risk_of_ruin(self, trade_returns, ruin_threshold=0.20):
        """
        What % of simulations result in account decline > 20%?
        Jane Street: "What's your risk of ruin?"
        """
        equity_paths = []
        
        for _ in range(self.simulations):
            shuffled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            equity_curve = np.cumprod(1 + shuffled_returns) * self.initial_equity
            equity_paths.append(equity_curve[-1])
        
        ruin_level = self.initial_equity * (1 - ruin_threshold)
        ruin_count = sum(1 for eq in equity_paths if eq < ruin_level)
        risk_of_ruin = ruin_count / self.simulations
        
        return {
            'risk_of_ruin_pct': risk_of_ruin * 100,
            'ruin_threshold': ruin_threshold,
            'simulations_in_ruin': ruin_count,
            'final_equities': equity_paths
        }
    
    def stress_test_crashes(self, trade_returns, crash_magnitude=-0.10, 
                           crash_injection_pct=0.10):
        """
        Inject artificial market crashes and test survival.
        "What if a -10% gap down hit during your position?"
        """
        results = []
        
        for sim in range(self.simulations):
            returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            
            if np.random.random() < crash_injection_pct:
                crash_idx = np.random.randint(0, len(returns))
                returns[crash_idx] = crash_magnitude
            
            equity_curve = np.cumprod(1 + returns) * self.initial_equity
            results.append({
                'final_equity': equity_curve[-1],
                'min_equity': np.min(equity_curve),
                'max_drawdown': (np.min(equity_curve) - self.initial_equity) / self.initial_equity
            })
        
        final_equities = [r['final_equity'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]
        
        return {
            'median_final_equity': np.median(final_equities),
            'worst_final_equity': np.percentile(final_equities, 5),
            'best_final_equity': np.percentile(final_equities, 95),
            'worst_drawdown': np.min(max_drawdowns),
            'avg_drawdown': np.mean(max_drawdowns),
            'survival_rate': sum(1 for eq in final_equities if eq > self.initial_equity * 0.8) / len(final_equities)
        }
    
    def calculate_var_cvar(self, trade_returns, confidence=0.95):
        """
        Value at Risk & Conditional Value at Risk.
        "What's my worst case in the 5% tail?"
        """
        equity_paths = []
        
        for _ in range(self.simulations):
            shuffled = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            equity = np.prod(1 + shuffled) * self.initial_equity
            equity_paths.append(equity)
        
        percentile = (1 - confidence) * 100
        var = np.percentile(equity_paths, percentile)
        cvar = np.mean([eq for eq in equity_paths if eq <= var])
        
        return {
            'value_at_risk': var,
            'conditional_var': cvar,
            'confidence_level': confidence,
            'expected_loss': self.initial_equity - cvar,
            'loss_pct': ((self.initial_equity - cvar) / self.initial_equity) * 100
        }
