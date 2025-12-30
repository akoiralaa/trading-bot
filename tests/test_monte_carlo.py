import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from monte_carlo_stress_test import MonteCarloStressTest


class TestMonteCarloStressTest(unittest.TestCase):
    """Unit tests for MonteCarloStressTest"""
    
    def setUp(self) -> None:
        self.mc = MonteCarloStressTest(initial_equity=100000, simulations=1000)
        np.random.seed(42)
    
    def test_probability_cone_structure(self) -> None:
        """Probability cone should return correct structure"""
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
        result = self.mc.run_probability_cone(returns)
        
        self.assertIn('worst_case', result)
        self.assertIn('best_case', result)
        self.assertIn('median', result)
        self.assertIn('percentiles', result)
        self.assertLess(result['worst_case'], result['median'])
        self.assertLess(result['median'], result['best_case'])
    
    def test_probability_cone_worst_better_than_best(self) -> None:
        """Worst case should be less than best case"""
        returns = np.random.normal(0.001, 0.02, 100)
        result = self.mc.run_probability_cone(returns)
        
        self.assertLess(result['worst_case'], result['best_case'])
    
    def test_percentiles_in_correct_order(self) -> None:
        """Percentiles should be in ascending order"""
        returns = np.random.normal(0.001, 0.02, 100)
        result = self.mc.run_probability_cone(returns)
        
        p5 = result['percentiles']['p5'][-1]
        p25 = result['percentiles']['p25'][-1]
        p50 = result['percentiles']['p50'][-1]
        p75 = result['percentiles']['p75'][-1]
        p95 = result['percentiles']['p95'][-1]
        
        self.assertLess(p5, p25)
        self.assertLess(p25, p50)
        self.assertLess(p50, p75)
        self.assertLess(p75, p95)
    
    def test_risk_of_ruin_calculation(self) -> None:
        """Risk of ruin should be percentage between 0-100"""
        returns = np.array([0.01, 0.02, -0.01, 0.015, 0.005])
        result = self.mc.calculate_risk_of_ruin(returns, ruin_threshold=0.20)
        
        self.assertGreaterEqual(result['risk_of_ruin_pct'], 0)
        self.assertLessEqual(result['risk_of_ruin_pct'], 100)
        self.assertEqual(result['ruin_threshold'], 0.20)
    
    def test_risk_of_ruin_high_for_losing_strategy(self) -> None:
        """Losing strategy should have high risk of ruin"""
        returns = np.full(100, -0.01)  # All losses
        result = self.mc.calculate_risk_of_ruin(returns, ruin_threshold=0.20)
        
        self.assertGreater(result['risk_of_ruin_pct'], 50)
    
    def test_risk_of_ruin_low_for_winning_strategy(self) -> None:
        """Winning strategy should have low risk of ruin"""
        returns = np.full(100, 0.02)  # All wins
        result = self.mc.calculate_risk_of_ruin(returns, ruin_threshold=0.20)
        
        self.assertLess(result['risk_of_ruin_pct'], 50)
    
    def test_crash_stress_test_structure(self) -> None:
        """Crash stress test should return correct metrics"""
        returns = np.random.normal(0.001, 0.02, 100)
        result = self.mc.stress_test_crashes(returns)
        
        self.assertIn('survival_rate', result)
        self.assertIn('worst_drawdown', result)
        self.assertIn('median_final_equity', result)
        self.assertGreaterEqual(result['survival_rate'], 0)
        self.assertLessEqual(result['survival_rate'], 1)
    
    def test_crash_decreases_survival(self) -> None:
        """Crash injection should decrease survival rate"""
        returns = np.full(100, 0.01)  # All wins
        
        result_no_crash = self.mc.stress_test_crashes(returns, crash_injection_pct=0)
        result_with_crash = self.mc.stress_test_crashes(returns, crash_injection_pct=0.50)
        
        self.assertGreaterEqual(result_no_crash['survival_rate'], result_with_crash['survival_rate'])
    
    def test_var_structure(self) -> None:
        """VaR should return correct structure"""
        returns = np.random.normal(0.001, 0.02, 100)
        result = self.mc.calculate_var_cvar(returns, confidence=0.95)
        
        self.assertIn('value_at_risk', result)
        self.assertIn('conditional_var', result)
        self.assertIn('expected_loss', result)
        self.assertIn('loss_pct', result)
        self.assertLess(result['value_at_risk'], self.mc.initial_equity)
    
    def test_var_var_relationship(self) -> None:
        """CVaR should be worse than VaR"""
        returns = np.random.normal(0.001, 0.02, 100)
        result = self.mc.calculate_var_cvar(returns, confidence=0.95)
        
        self.assertLess(result['conditional_var'], result['value_at_risk'])
    
    def test_loss_percentage_positive(self) -> None:
        """Loss percentage should be positive for losing scenarios"""
        returns = np.full(100, -0.01)  # All losses
        result = self.mc.calculate_var_cvar(returns, confidence=0.95)
        
        self.assertGreater(result['loss_pct'], 0)


if __name__ == '__main__':
    unittest.main()
