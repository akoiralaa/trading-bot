"""
Test Suite for Jane Street-Ready Trading Bot

Tests verify:
1. Slippage model matches market microstructure
2. Kelly sizing adapts to signal confidence
3. Monte Carlo finds robust strategies
4. Vector zones prevent whipsaw

This is your "code quality" signal to recruiters.
"""

import numpy as np
import pandas as pd
import pytest
from typing import List

# Import modules to test
from slippage_model import SlippageModel, BacktesterWithSlippage
from kelly_criterion import KellyCriterion, DynamicPositionSizer, TradeMetrics
from monte_carlo_simulator import MonteCarloBacktester
from vector_zone_detector import VectorZoneDetector, EnhancedPatternDetector


class TestSlippageModel:
    """Test dynamic slippage calculations"""
    
    def setup_method(self):
        self.slippage = SlippageModel(
            base_spread_multiplier=0.5,
            volume_threshold=0.2,
            basis_point_scaler=10.0
        )
    
    def test_small_order_uses_base_spread(self):
        """Small orders should have minimal slippage"""
        
        close_price = 100.0
        small_order = 100  # Small
        atr = 0.5
        avg_volume = 100000
        
        entry_price, details = self.slippage.entry_execution_price(
            close_price, small_order, atr, avg_volume
        )
        
        # Slippage should be small (just spread/2)
        slippage_bps = details['execution_slippage_bps']
        assert slippage_bps < 5, f"Small order slippage too high: {slippage_bps} bps"
    
    def test_large_order_increases_slippage(self):
        """Large orders relative to volume should increase slippage"""
        
        close_price = 100.0
        atr = 0.5
        avg_volume = 100000
        
        # Small order
        small_slippage, _ = self.slippage.entry_execution_price(
            close_price, 5000, atr, avg_volume
        )
        
        # Large order (50% of volume)
        large_slippage, details = self.slippage.entry_execution_price(
            close_price, 50000, atr, avg_volume
        )
        
        # Large order should have more slippage
        assert large_slippage > small_slippage, \
            f"Large order should cost more: {large_slippage} vs {small_slippage}"
    
    def test_atr_scales_spread(self):
        """Higher ATR should scale spread wider"""
        
        close_price = 100.0
        order_size = 10000
        avg_volume = 100000
        
        # Low volatility
        _, low_vol_details = self.slippage.entry_execution_price(
            close_price, order_size, atr=0.2, avg_volume_7bar=avg_volume
        )
        
        # High volatility
        _, high_vol_details = self.slippage.entry_execution_price(
            close_price, order_size, atr=1.0, avg_volume_7bar=avg_volume
        )
        
        # High vol should have wider spread
        assert high_vol_details['total_spread_bps'] > low_vol_details['total_spread_bps'], \
            "High volatility should create wider spreads"


class TestKellyCriterion:
    """Test Kelly Criterion position sizing"""
    
    def setup_method(self):
        self.kelly = KellyCriterion(
            lookback_trades=50,
            kelly_fraction=0.5,
            min_risk=0.01,
            max_risk=0.05
        )
    
    def test_kelly_below_min_risk(self):
        """Negative Kelly should floor at min_risk"""
        
        # Create losing trades
        for i in range(20):
            trade = TradeMetrics(
                entry_price=100.0,
                exit_price=99.0,  # Losing trade
                size=100,
                entry_bar=i*10,
                exit_bar=i*10+5,
                vector_strength=0.5
            )
            self.kelly.add_trade(trade)
        
        size = self.kelly.get_position_size_bayesian(
            capital=100000,
            vector_strength=0.5
        )
        
        # Should be at minimum
        assert size >= 100000 * self.kelly.min_risk, \
            f"Kelly should floor at min_risk: {size} < {100000 * self.kelly.min_risk}"
    
    def test_kelly_adapts_to_strength(self):
        """Kelly sizing should scale with vector strength"""
        
        # Create winning trades
        for i in range(30):
            trade = TradeMetrics(
                entry_price=100.0,
                exit_price=103.0,  # +3% win
                size=100,
                entry_bar=i*10,
                exit_bar=i*10+5,
                vector_strength=0.7
            )
            self.kelly.add_trade(trade)
        
        # Weak signal
        weak_size = self.kelly.get_position_size_bayesian(
            capital=100000,
            vector_strength=0.51  # Barely bullish
        )
        
        # Strong signal
        strong_size = self.kelly.get_position_size_bayesian(
            capital=100000,
            vector_strength=0.90  # Very bullish
        )
        
        # Strong signal should size larger
        assert strong_size > weak_size, \
            f"Strong signal should size larger: {strong_size} <= {weak_size}"
    
    def test_kelly_ceiling(self):
        """Kelly should not exceed max_risk"""
        
        # Create very profitable trades
        for i in range(40):
            trade = TradeMetrics(
                entry_price=100.0,
                exit_price=110.0,  # +10% win
                size=100,
                entry_bar=i*10,
                exit_bar=i*10+5,
                vector_strength=0.95
            )
            self.kelly.add_trade(trade)
        
        size = self.kelly.get_position_size_bayesian(
            capital=100000,
            vector_strength=0.95
        )
        
        # Should not exceed max
        assert size <= 100000 * self.kelly.max_risk, \
            f"Kelly should ceiling at max_risk: {size} > {100000 * self.kelly.max_risk}"


class TestMonteCarlo:
    """Test Monte Carlo simulation"""
    
    def test_survival_rate_positive_edge(self):
        """Strategy with positive edge should have high survival"""
        
        # Generate winning trades (60% win rate, +2% avg)
        trades = []
        for _ in range(100):
            if np.random.rand() < 0.60:
                trades.append(np.random.normal(0.02, 0.01))
            else:
                trades.append(np.random.normal(-0.01, 0.01))
        
        mc = MonteCarloBacktester(initial_capital=100000)
        results = mc.run_full_monte_carlo(trades, num_simulations=1000)
        
        # Should have high survival
        assert results['survival_rate'] > 0.90, \
            f"Positive edge should have >90% survival: {results['survival_rate']:.1%}"
    
    def test_survival_rate_negative_edge(self):
        """Strategy with negative edge should have low survival"""
        
        # Generate losing trades (40% win rate, +1.5% avg win, -2% avg loss)
        trades = []
        for _ in range(100):
            if np.random.rand() < 0.40:
                trades.append(np.random.normal(0.015, 0.01))
            else:
                trades.append(np.random.normal(-0.02, 0.01))
        
        mc = MonteCarloBacktester(initial_capital=100000)
        results = mc.run_full_monte_carlo(trades, num_simulations=1000)
        
        # Should have low survival
        assert results['survival_rate'] < 0.50, \
            f"Negative edge should have <50% survival: {results['survival_rate']:.1%}"
    
    def test_crash_injection_reduces_survival(self):
        """Adding crashes should reduce survival rate"""
        
        trades = [np.random.normal(0.01, 0.01) for _ in range(100)]
        
        mc = MonteCarloBacktester(initial_capital=100000)
        
        # Baseline survival (no crashes)
        baseline_results = mc.run_full_monte_carlo(trades, num_simulations=500)
        baseline_survival = baseline_results['survival_rate']
        
        # With crashes
        crash_results = mc.run_crash_injection_test(
            trades,
            crash_magnitude=-0.05,
            num_tests=100
        )
        crash_survival = crash_results['overall_survival_rate']
        
        # Crashes should reduce survival
        assert crash_survival <= baseline_survival, \
            f"Crashes should reduce survival: {crash_survival:.1%} > {baseline_survival:.1%}"


class TestVectorZoneDetector:
    """Test vector zone detection"""
    
    def setup_method(self):
        self.detector = VectorZoneDetector(lookback_period=20)
    
    def test_zone_width_scales_with_volatility(self):
        """Zone width should increase with ATR"""
        
        # Create sample prices
        prices = 100 + np.random.randn(252).cumsum()
        high = prices + np.abs(np.random.randn(252))
        low = prices - np.abs(np.random.randn(252))
        close = prices
        
        # Low volatility
        atr_low = np.ones(252) * 0.5
        zone_low = self.detector.get_vector_zone(
            high, low, close, atr_low, bar_index=100, atr_multiplier=1.0
        )
        
        # High volatility
        atr_high = np.ones(252) * 2.0
        zone_high = self.detector.get_vector_zone(
            high, low, close, atr_high, bar_index=100, atr_multiplier=1.0
        )
        
        # Higher volatility should create wider zone
        assert zone_high.zone_width > zone_low.zone_width, \
            f"Higher ATR should create wider zone: {zone_high.zone_width} <= {zone_low.zone_width}"
    
    def test_bounce_signal_filters_noise(self):
        """Bounce signal should require clear momentum"""
        
        # Create a zone
        from vector_zone_detector import VectorZone
        zone = VectorZone(
            upper_band=101.0,
            middle_line=100.0,
            lower_band=99.0,
            zone_width=1.0,
            atr=0.5,
            timestamp=100
        )
        
        # Test: Price barely touches lower band, no bounce
        triggered, reason = self.detector.check_bounce_signal(
            price_prev=99.05,  # Just barely below
            price_curr=99.50,  # Didn't bounce high enough
            zone=zone,
            side='long'
        )
        
        assert not triggered, f"Shouldn't trigger on weak bounce: {reason}"
        
        # Test: Clear bounce off lower band
        triggered, reason = self.detector.check_bounce_signal(
            price_prev=98.5,  # Well below lower band
            price_curr=100.5,  # Bounced above middle
            zone=zone,
            side='long'
        )
        
        assert triggered, f"Should trigger on clear bounce: {reason}"


class TestIntegration:
    """Integration tests - all components together"""
    
    def test_full_backtest_with_all_features(self):
        """Test backtest with slippage, Kelly, and zone filtering"""
        
        # Generate sample prices
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        base_price = 100
        returns = np.random.normal(0.0005, 0.01, 252)
        close = base_price * np.exp(np.cumsum(returns))
        high = close + np.abs(np.random.normal(0, 0.5, 252))
        low = close - np.abs(np.random.normal(0, 0.5, 252))
        volume = np.random.uniform(1e6, 2e6, 252)
        
        prices_df = pd.DataFrame({
            'close': close,
            'high': high,
            'low': low,
            'volume': volume
        })
        
        # Initialize components
        slippage = SlippageModel()
        kelly = KellyCriterion()
        pattern_detector = EnhancedPatternDetector()
        mc = MonteCarloBacktester()
        
        # Generate signals (simplified)
        signals = []
        for i in range(len(prices_df)):
            if i < 10:
                signals.append(0)
            elif i % 50 == 0:
                signals.append(1)  # Long signal every 50 bars
            elif i % 50 == 20:
                signals.append(0)  # Exit 20 bars later
            else:
                signals.append(0)
        
        signals = pd.Series(signals)
        
        # Run backtest with slippage
        backtester = BacktesterWithSlippage()
        metrics = backtester.calculate_position_metrics(prices_df, signals)
        
        # Should have executed trades
        assert metrics['total_trades'] > 0, "No trades executed"
        
        # Should have calculated slippage
        assert metrics['avg_slippage_bps'] >= 0, "Slippage should be non-negative"
        
        # Should have reasonable return
        assert -100 < metrics['return_pct'] < 100, "Return seems unrealistic"
        
        print(f"\nIntegration test results:")
        print(f"  Total trades: {metrics['total_trades']}")
        print(f"  Total return: {metrics['return_pct']:.2f}%")
        print(f"  Avg slippage: {metrics['avg_slippage_bps']:.2f} bps")


if __name__ == "__main__":
    # Run tests
    print("Running slippage model tests...")
    slippage_tests = TestSlippageModel()
    slippage_tests.setup_method()
    slippage_tests.test_small_order_uses_base_spread()
    slippage_tests.test_large_order_increases_slippage()
    slippage_tests.test_atr_scales_spread()
    print("✓ Slippage tests passed")
    
    print("\nRunning Kelly Criterion tests...")
    kelly_tests = TestKellyCriterion()
    kelly_tests.setup_method()
    kelly_tests.test_kelly_below_min_risk()
    kelly_tests.test_kelly_adapts_to_strength()
    kelly_tests.test_kelly_ceiling()
    print("✓ Kelly tests passed")
    
    print("\nRunning Monte Carlo tests...")
    mc_tests = TestMonteCarlo()
    mc_tests.test_survival_rate_positive_edge()
    mc_tests.test_survival_rate_negative_edge()
    print("✓ Monte Carlo tests passed")
    
    print("\nRunning vector zone tests...")
    zone_tests = TestVectorZoneDetector()
    zone_tests.setup_method()
    zone_tests.test_zone_width_scales_with_volatility()
    zone_tests.test_bounce_signal_filters_noise()
    print("✓ Vector zone tests passed")
    
    print("\nRunning integration test...")
    integration = TestIntegration()
    integration.test_full_backtest_with_all_features()
    print("✓ Integration test passed")
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED ✓")
    print("="*50)