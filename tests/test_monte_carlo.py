import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monte_carlo_stress_test import MonteCarloStressTest

class TestMonteCarloStressTest(unittest.TestCase):
    """Unit tests for Monte Carlo Stress Testing"""
    
    def setUp(self) -> None:
        self.mc = MonteCarloStressTest(initial_equity=100000, simulations=1000)
    
    def test_probability_cone_structure(self) -> None:
        """Probability cone should return correct structure"""
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
        result = self.mc.run_probability_cone(returns)
        
        self.assertIn('paths', result)
        self.assertIn('final_equity_dist', result)
        self.assertIn('percentiles', result)
        self.assertIn('p5_worst_case', result)
        self.assertIn('p50_median', result)
        self.assertIn('p95_best_case', result)
    
    def test_percentiles_in_correct_order(self) -> None:
        """Percentiles should be ordered: worst < median < best"""
        returns = np.random.normal(0.001, 0.02, 100)
        result = self.mc.run_probability_cone(returns)
        
        self.assertLess(result['p5_worst_case'], result['p50_median'])
        self.assertLess(result['p50_median'], result['p95_best_case'])
    
    def test_probability_cone_worst_better_than_best(self) -> None:
        """Worst case should be less than best case"""
        returns = np.random.normal(0.001, 0.02, 100)
        result = self.mc.run_probability_cone(returns)
        
        self.assertLess(result['p5_worst_case'], result['p95_best_case'])
    
    def test_risk_of_ruin_calculation(self) -> None:
        """Risk of ruin should be percentage between 0-100"""
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
        result = self.mc.calculate_risk_of_ruin(returns, ruin_threshold=0.20)
        
        self.assertGreaterEqual(result['risk_of_ruin_pct'], 0)
        self.assertLessEqual(result['risk_of_ruin_pct'], 100)
    
    def test_risk_of_ruin_low_for_winning_strategy(self) -> None:
        """Winning strategy should have low RoR"""
        returns = np.full(100, 0.01)  # All wins
        result = self.mc.calculate_risk_of_ruin(returns)
        
        self.assertLess(result['risk_of_ruin_pct'], 5)
    
    def test_risk_of_ruin_high_for_losing_strategy(self) -> None:
        """Losing strategy should have high RoR"""
        returns = np.full(100, -0.01)  # All losses
        result = self.mc.calculate_risk_of_ruin(returns)
        
        self.assertGreater(result['risk_of_ruin_pct'], 50)
    
    def test_crash_stress_test_structure(self) -> None:
        """Crash stress test should return correct metrics"""
        returns = np.random.normal(0.001, 0.02, 100)
        result = self.mc.stress_test_shocks(returns)
        
        self.assertIn('shock_survival_rate', result)
        self.assertIn('median_drawdown', result)
        self.assertIn('tail_drawdown_p5', result)
    
    def test_crash_decreases_survival(self) -> None:
        """Crash injection should decrease survival rate"""
        returns = np.full(100, 0.01)  # All wins
        
        result_no_crash = self.mc.stress_test_shocks(returns, shock_prob=0)
        result_with_crash = self.mc.stress_test_shocks(returns, shock_prob=1.0)
        
        self.assertGreaterEqual(result_no_crash['shock_survival_rate'], result_with_crash['shock_survival_rate'])
    
    def test_var_structure(self) -> None:
        """VaR should return correct structure"""
        returns = np.random.normal(0.001, 0.02, 100)
        result = self.mc.get_tail_risk_metrics(returns, alpha=0.95)
        
        self.assertIn('var_alpha', result)
        self.assertIn('cvar_expected_shortfall', result)
        self.assertIn('max_expected_loss_pct', result)
    
    def test_var_cvar_relationship(self) -> None:
        """CVaR should be worse than or equal to VaR"""
        returns = np.random.normal(0.001, 0.02, 100)
        result = self.mc.get_tail_risk_metrics(returns, alpha=0.95)
        
        # CVaR should be <= VaR (worse or equal)
        self.assertLessEqual(result['cvar_expected_shortfall'], result['var_alpha'])
    
    def test_loss_percentage_positive(self) -> None:
        """Loss percentage should be positive for losing scenarios"""
        returns = np.full(100, -0.01)
        result = self.mc.get_tail_risk_metrics(returns, alpha=0.95)
        
        self.assertGreater(result['max_expected_loss_pct'], 0)

if __name__ == '__main__':
    unittest.main()
