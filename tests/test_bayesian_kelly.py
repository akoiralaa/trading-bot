import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bayesian_kelly import BayesianKellyCriterion

class TestBayesianKellyCriterion(unittest.TestCase):
    """Unit tests for Bayesian Kelly Criterion"""
    
    def setUp(self) -> None:
        self.kelly = BayesianKellyCriterion(account_equity=100000)
    
    def test_at_threshold_returns_value(self) -> None:
        """At minimum threshold, should return non-zero kelly"""
        frac = self.kelly.calculate_kelly_fraction(0.51)
        self.assertGreater(frac, 0)
    
    def test_below_threshold_returns_zero(self) -> None:
        """Below minimum strength should return zero"""
        frac = self.kelly.calculate_kelly_fraction(0.50)
        self.assertEqual(frac, 0.0)
    
    def test_high_confidence_higher_kelly(self) -> None:
        """Higher confidence should produce larger Kelly fraction"""
        frac_low = self.kelly.calculate_kelly_fraction(0.60)
        frac_high = self.kelly.calculate_kelly_fraction(0.90)
        self.assertLess(frac_low, frac_high)
    
    def test_kelly_capped_at_25_percent(self) -> None:
        """Kelly fraction should be capped at 25%"""
        frac = self.kelly.calculate_kelly_fraction(0.95)
        self.assertLessEqual(frac, 0.25)
    
    def test_position_size_scales_with_confidence(self) -> None:
        """Position size should increase with vector strength"""
        qty_low = self.kelly.calculate_position_size(0.55, 2.0, 100000)
        qty_high = self.kelly.calculate_position_size(0.85, 2.0, 100000)
        self.assertLess(qty_low, qty_high)
    
    def test_position_size_respects_buying_power(self) -> None:
        """Position size should not exceed buying power"""
        qty = self.kelly.calculate_position_size(0.80, 1.0, 1000)
        self.assertLessEqual(qty * 1.0, 1000)
    
    def test_position_size_respects_concentration_limit(self) -> None:
        """Position should respect 20% concentration limit"""
        qty = self.kelly.calculate_position_size(0.90, 1.0, 100000, max_concentration=0.20)
        self.assertLessEqual(qty * 1.0, 100000 * 0.20)
    
    def test_below_minimum_strength_returns_unfavorable_ev(self) -> None:
        """Below minimum strength should return unfavorable EV"""
        ev = self.kelly.get_expected_value(0.50, 100.0, 95.0, 110.0)
        self.assertFalse(ev['is_favorable'])
    
    def test_get_expected_value_positive_for_good_trade(self) -> None:
        """Good trade should have positive EV"""
        ev = self.kelly.get_expected_value(0.70, 100.0, 95.0, 110.0)
        self.assertGreater(ev['ev'], 0)
        self.assertTrue(ev['is_favorable'])
    
    def test_get_expected_value_negative_for_bad_trade(self) -> None:
        """Bad trade should have negative EV"""
        ev = self.kelly.get_expected_value(0.55, 100.0, 95.0, 101.0)
        self.assertLess(ev['ev'], 0)
        self.assertFalse(ev['is_favorable'])

if __name__ == '__main__':
    unittest.main()
