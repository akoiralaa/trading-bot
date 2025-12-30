import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bayesian_kelly import BayesianKellyCriterion


class TestBayesianKellyCriterion(unittest.TestCase):
    """Unit tests for BayesianKellyCriterion"""
    
    def setUp(self) -> None:
        self.kelly = BayesianKellyCriterion(
            account_equity=100000,
            fractional_kelly=0.5,
            reward_risk_ratio=2.0,
            min_vector_strength=0.51
        )
    
    def test_below_threshold_returns_zero(self) -> None:
        """Vector strength below threshold should return 0 kelly"""
        kelly_frac = self.kelly.calculate_kelly_fraction(vector_strength=0.50)
        self.assertEqual(kelly_frac, 0.0)
    
    def test_at_threshold_returns_value(self) -> None:
        """Vector strength at threshold should return positive kelly"""
        kelly_frac = self.kelly.calculate_kelly_fraction(vector_strength=0.51)
        self.assertGreater(kelly_frac, 0.0)
        self.assertLess(kelly_frac, 0.25)
    
    def test_high_confidence_higher_kelly(self) -> None:
        """Higher vector strength should return higher kelly"""
        kelly_low = self.kelly.calculate_kelly_fraction(vector_strength=0.55)
        kelly_high = self.kelly.calculate_kelly_fraction(vector_strength=0.90)
        self.assertGreater(kelly_high, kelly_low)
    
    def test_kelly_capped_at_25_percent(self) -> None:
        """Kelly should never exceed 25% for safety"""
        kelly_frac = self.kelly.calculate_kelly_fraction(vector_strength=0.99)
        self.assertLessEqual(kelly_frac, 0.25)
    
    def test_position_size_scales_with_confidence(self) -> None:
        """Position size should scale with vector strength"""
        qty_low = self.kelly.calculate_position_size(
            vector_strength=0.55, risk_per_share=10.0, buying_power=50000)
        qty_high = self.kelly.calculate_position_size(
            vector_strength=0.90, risk_per_share=10.0, buying_power=50000)
        self.assertGreater(qty_high, qty_low)
    
    def test_position_size_respects_buying_power(self) -> None:
        """Position size should not exceed buying power"""
        qty = self.kelly.calculate_position_size(
            vector_strength=0.90, risk_per_share=10.0, buying_power=1000)
        cost = qty * 10.0
        self.assertLessEqual(cost, 1000)
    
    def test_position_size_respects_concentration_limit(self) -> None:
        """Position size should not exceed 20% concentration"""
        qty = self.kelly.calculate_position_size(
            vector_strength=0.90, risk_per_share=100.0, buying_power=1000000, max_concentration=0.20)
        position_value = qty * 100.0
        max_allowed = 100000 * 0.20
        self.assertLessEqual(position_value, max_allowed)
    
    def test_expected_value_positive_for_good_trade(self) -> None:
        """Good trade should have positive EV"""
        ev = self.kelly.expected_value(
            vector_strength=0.70, entry_price=100.0, stop_price=95.0, target_price=110.0)
        self.assertTrue(ev['favorable'])
        self.assertGreater(ev['expected_value'], 0)
    
    def test_expected_value_negative_for_bad_trade(self) -> None:
        """Bad trade should have negative EV"""
        ev = self.kelly.expected_value(
            vector_strength=0.55, entry_price=100.0, stop_price=95.0, target_price=101.0)
        self.assertFalse(ev['favorable'])
        self.assertLess(ev['expected_value'], 0)
    
    def test_below_minimum_strength_returns_zero_ev(self) -> None:
        """Below minimum strength should return zero EV dict"""
        ev = self.kelly.expected_value(
            vector_strength=0.50, entry_price=100.0, stop_price=95.0, target_price=110.0)
        self.assertEqual(ev['expected_value'], 0)
        self.assertFalse(ev['favorable'])


if __name__ == '__main__':
    unittest.main()
