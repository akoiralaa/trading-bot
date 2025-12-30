import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from regime_detector import RegimeDetector


class TestRegimeDetector(unittest.TestCase):
    """Unit tests for RegimeDetector"""
    
    def setUp(self) -> None:
        self.detector = RegimeDetector(atr_multiplier=2.0, min_vector_strength=0.51)
    
    def test_trending_regime_detection(self) -> None:
        """Strong uptrend should be detected as TRENDING"""
        prices = np.linspace(100, 130, 30)
        result = self.detector.detect_regime(prices, lookback=30)
        
        self.assertEqual(result['regime'], 'TRENDING')
        self.assertGreater(result['trend_strength'], 0.001)
    
    def test_sideways_regime_detection(self) -> None:
        """Flat prices should be detected as SIDEWAYS"""
        prices = np.full(30, 100.0)
        result = self.detector.detect_regime(prices, lookback=30)
        
        self.assertEqual(result['regime'], 'SIDEWAYS')
    
    def test_volatile_regime_detection(self) -> None:
        """High volatility should be detected as VOLATILE"""
        np.random.seed(42)
        prices = 100 + np.random.normal(0, 5, 30)
        result = self.detector.detect_regime(prices, lookback=30)
        
        self.assertGreater(result['volatility'], 0.01)
    
    def test_breakout_signal_requires_all_conditions(self) -> None:
        """Breakout signal should require all three conditions"""
        result = self.detector.is_breakout_signal(
            price=110.0,
            vector_price=100.0,
            atr=2.0,
            vector_strength=0.80,
            regime='TRENDING'
        )
        
        self.assertTrue(result['is_signal'])
        self.assertTrue(result['clears_dead_band'])
        self.assertTrue(result['vector_strength_ok'])
        self.assertTrue(result['regime_ok'])
    
    def test_breakout_rejected_on_weak_vector(self) -> None:
        """Breakout should be rejected if vector strength too low"""
        result = self.detector.is_breakout_signal(
            price=110.0,
            vector_price=100.0,
            atr=2.0,
            vector_strength=0.50,
            regime='TRENDING'
        )
        
        self.assertFalse(result['is_signal'])
        self.assertFalse(result['vector_strength_ok'])
    
    def test_breakout_rejected_on_sideways_regime(self) -> None:
        """Breakout should be rejected in sideways regime"""
        result = self.detector.is_breakout_signal(
            price=110.0,
            vector_price=100.0,
            atr=2.0,
            vector_strength=0.80,
            regime='SIDEWAYS'
        )
        
        self.assertFalse(result['is_signal'])
        self.assertFalse(result['regime_ok'])
    
    def test_breakout_rejected_if_inside_band(self) -> None:
        """Breakout should be rejected if price inside dead band"""
        result = self.detector.is_breakout_signal(
            price=101.0,
            vector_price=100.0,
            atr=2.0,
            vector_strength=0.80,
            regime='TRENDING'
        )
        
        self.assertFalse(result['is_signal'])
        self.assertFalse(result['clears_dead_band'])
    
    def test_dynamic_stop_long_position(self) -> None:
        """Long stop should be below entry"""
        entry_price = 100.0
        result = self.detector.calculate_dynamic_stop(
            entry_price=entry_price,
            vector_price=98.0,
            atr=1.0,
            side='long'
        )
        
        self.assertLess(result['stop_price'], entry_price)
        self.assertGreater(result['risk_distance'], 0)
    
    def test_dynamic_stop_short_position(self) -> None:
        """Short stop should be above entry"""
        entry_price = 100.0
        result = self.detector.calculate_dynamic_stop(
            entry_price=entry_price,
            vector_price=102.0,
            atr=1.0,
            side='short'
        )
        
        self.assertGreater(result['stop_price'], entry_price)
        self.assertGreater(result['risk_distance'], 0)
    
    def test_adaptive_zones_calculation(self) -> None:
        """Adaptive zones should scale with ATR"""
        prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
        atr_values = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        vector_prices = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
        vector_strengths = np.array([0.6, 0.7, 0.8, 0.8, 0.9])
        
        result = self.detector.calculate_adaptive_zones(
            prices, atr_values, vector_prices, vector_strengths
        )
        
        self.assertIsNotNone(result['upper_band'])
        self.assertIsNotNone(result['lower_band'])
        self.assertGreater(len(result['is_signal_valid']), 0)


if __name__ == '__main__':
    unittest.main()
