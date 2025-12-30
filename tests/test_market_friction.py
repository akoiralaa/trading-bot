import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from market_friction_model import MarketFrictionModel

class TestMarketFrictionModel(unittest.TestCase):
    """Unit tests for Market Friction Model"""
    
    def setUp(self) -> None:
        self.model = MarketFrictionModel(market_impact_coeff=0.1, bid_ask_spread_bps=2.0)
    
    def test_small_order_low_impact(self) -> None:
        """Small order should have low impact"""
        result = self.model.calculate_dynamic_slippage(qty=100, avg_volume=100000, price=100.0)
        
        self.assertLess(result['impact_bps'], 1.0)
        self.assertGreater(result['impact_bps'], 0)
    
    def test_large_order_high_impact(self) -> None:
        """Large order should have higher impact"""
        small = self.model.calculate_dynamic_slippage(qty=100, avg_volume=100000, price=100.0)
        large = self.model.calculate_dynamic_slippage(qty=20000, avg_volume=100000, price=100.0)
        
        self.assertGreater(large['impact_bps'], small['impact_bps'])
    
    def test_zero_quantity(self) -> None:
        """Zero quantity should produce minimal impact"""
        result = self.model.calculate_dynamic_slippage(qty=0, avg_volume=10000, price=100.0)
        
        self.assertEqual(result['volume_ratio'], 0.0)
        # With zero qty, impact should be half the bid-ask spread
        self.assertEqual(result['impact_bps'], self.model.bid_ask_spread_bps / 2)
    
    def test_total_friction_buy_side(self) -> None:
        """Buy side should include bid-ask spread cost"""
        result = self.model.calculate_total_friction(qty=1000, avg_volume=100000, price=100.0, side='buy')
        
        self.assertIn('total_friction_bps', result)
        self.assertGreater(result['execution_price'], 100.0)
    
    def test_total_friction_sell_side(self) -> None:
        """Sell side should reduce execution price"""
        result = self.model.calculate_total_friction(qty=1000, avg_volume=100000, price=100.0, side='sell')
        
        self.assertIn('total_friction_bps', result)
        self.assertLess(result['execution_price'], 100.0)
    
    def test_max_position_constraint(self) -> None:
        """Max position should be 5% of daily volume"""
        max_size = self.model.get_liquidity_constrained_size(avg_volume=10000, max_participation_rate=0.05)
        
        self.assertEqual(max_size, 500)

if __name__ == '__main__':
    unittest.main()
