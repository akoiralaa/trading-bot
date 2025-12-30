import numpy as np
import pandas as pd
from scipy import stats


class RegimeDetector:
    """
    Volatility-adjusted dead band for signal vs noise detection.
    Uses ATR zones to confirm momentum, not just direction.
    """
    
    def __init__(self, atr_multiplier=2.0, min_vector_strength=0.51):
        self.atr_multiplier = atr_multiplier
        self.min_vector_strength = min_vector_strength
    
    def calculate_adaptive_zones(self, prices, atr_values, vector_prices, 
                                 vector_strengths):
        """
        Create ATR-based dead bands around vector line.
        Only trade if price breaks outside the band (proof of momentum).
        """
        dead_band_width = self.atr_multiplier * atr_values
        
        upper_band = vector_prices + dead_band_width
        lower_band = vector_prices - dead_band_width
        
        is_above_band = prices > upper_band
        is_below_band = prices < lower_band
        is_signal_valid = (is_above_band | is_below_band) & (vector_strengths >= self.min_vector_strength)
        
        return {
            'upper_band': upper_band,
            'lower_band': lower_band,
            'dead_band_width': dead_band_width,
            'is_above_band': is_above_band,
            'is_below_band': is_below_band,
            'is_signal_valid': is_signal_valid,
            'signal_strength': vector_strengths * is_signal_valid.astype(float)
        }
    
    def detect_regime(self, prices, lookback=30):
        """
        Identify current market regime: Trending vs Sideways vs Volatile
        """
        recent_prices = prices[-lookback:]
        
        volatility = np.std(np.diff(recent_prices) / recent_prices[:-1])
        
        slope, intercept, r_val, p_val, std_err = stats.linregress(
            np.arange(lookback), recent_prices
        )
        
        trend_strength = abs(slope) / np.mean(recent_prices)
        
        if p_val < 0.05 and trend_strength > 0.001:
            regime = 'TRENDING'
        elif volatility > 0.02:
            regime = 'VOLATILE'
        else:
            regime = 'SIDEWAYS'
        
        return {
            'regime': regime,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'p_value': p_val,
            'regime_confidence': 1.0 - p_val if regime == 'TRENDING' else 1.0 - volatility
        }
    
    def is_breakout_signal(self, price, vector_price, atr, 
                          vector_strength, regime):
        """
        Multi-factor confirmation:
        1. Price clears dead band (momentum)
        2. Vector strength is high (confidence)
        3. Regime is favorable (not sideways noise)
        """
        dead_band = self.atr_multiplier * atr
        
        clears_band = abs(price - vector_price) > dead_band
        strong_vector = vector_strength >= self.min_vector_strength
        favorable_regime = regime != 'SIDEWAYS'
        
        all_confirmed = clears_band and strong_vector and favorable_regime
        
        return {
            'is_signal': all_confirmed,
            'clears_dead_band': clears_band,
            'vector_strength_ok': strong_vector,
            'regime_ok': favorable_regime,
            'signal_quality': float(all_confirmed) * vector_strength,
            'regime': regime
        }
    
    def calculate_dynamic_stop(self, entry_price, vector_price, atr, 
                              side='long'):
        """
        Stop loss scales with volatility and position in vector zone.
        Wider stops in high volatility, tighter in low volatility.
        """
        if side == 'long':
            stop = entry_price - (self.atr_multiplier * atr)
            stop = min(stop, vector_price - atr)
        else:
            stop = entry_price + (self.atr_multiplier * atr)
            stop = max(stop, vector_price + atr)
        
        risk_distance = abs(entry_price - stop)
        
        return {
            'stop_price': stop,
            'risk_distance': risk_distance,
            'atr_based': self.atr_multiplier * atr,
            'side': side
        }
