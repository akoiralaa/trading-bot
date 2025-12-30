import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from market_friction_model import MarketFrictionModel


class TestMarketFrictionModel(unittest.TestCase):
    """Unit tests for MarketFrictionModel"""
    
    def setUp(self) -> None:
        self.model = MarketFrictionModel(market_impact_coeff=0.1, bid_ask_spread_bps=2.0)
    
    def test_small_order_low_impact(self) -> None:
        """Small order should have minimal impact"""
        result = self.model.calculate_dynamic_slippage(qty=100, avg_volume=10000, price=100.0)
        
        self.assertEqual(result['volume_ratio'], 1.0)
        self.assertLess(result['impact_bps'], 1.0)
        self.assertAlmostEqual(result['execution_price'], 100.0, places=2)
    
    def test_large_order_high_impact(self) -> None:
        """Large order relative to volume should have significant impact"""
        result = self.model.calculate_dynamic_slippage(qty=2000, avg_volume=10000, price=100.0)
        
        self.assertEqual(result['volume_ratio'], 20.0)
        self.assertGreater(result['impact_bps'], 5.0)
        self.assertGreater(result['execution_price'], 100.0)
    
    def test_total_friction_buy_side(self) -> None:
        """Buy side should increase execution price"""
        result = self.model.calculate_total_friction(qty=500, avg_volume=10000, price=100.0, side='buy')
        
        self.assertGreater(result['execution_price'], 100.0)
        self.assertGreater(result['total_friction_bps'], 0)
    
    def test_total_friction_sell_side(self) -> None:
        """Sell side should decrease execution price"""
        result = self.model.calculate_total_friction(qty=500, avg_volume=10000, price=100.0, side='sell')
        
        self.assertLess(result['execution_price'], 100.0)
    
    def test_max_position_constraint(self) -> None:
        """Max position should be 5% of daily volume"""
        max_size = self.model.get_max_position_size(avg_volume=10000, account_equity=100000)
        
        self.assertEqual(max_size, 500)  # 5% of 10000
    
    def test_zero_quantity(self) -> None:
        """Zero quantity should produce zero slippage"""
        result = self.model.calculate_dynamic_slippage(qty=0, avg_volume=10000, price=100.0)
        
        self.assertEqual(result['volume_ratio'], 0.0)
        self.assertEqual(result['impact_bps'], 0.5)


if __name__ == '__main__':
    unittest.main()
