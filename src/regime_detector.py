import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Volatility-adjusted dead band for signal vs noise detection.
    Uses ATR zones to confirm momentum, not just direction.
    """
    
    def __init__(self, atr_multiplier: float = 2.0, min_vector_strength: float = 0.51) -> None:
        self.atr_multiplier = atr_multiplier
        self.min_vector_strength = min_vector_strength
        logger.info(f"RegimeDetector initialized: atr_mult={atr_multiplier}, min_strength={min_vector_strength}")
    
    def calculate_adaptive_zones(self, prices: np.ndarray, atr_values: np.ndarray, 
                                vector_prices: np.ndarray, vector_strengths: np.ndarray) -> Dict:
        """
        Create ATR-based dead bands around vector line.
        Only trade if price breaks outside the band (proof of momentum).
        
        Args:
            prices: Array of prices
            atr_values: Array of ATR values
            vector_prices: Array of vector line prices
            vector_strengths: Array of vector strength values
        
        Returns:
            Dict with zone metrics
        """
        dead_band_width: np.ndarray = self.atr_multiplier * atr_values
        
        upper_band: np.ndarray = vector_prices + dead_band_width
        lower_band: np.ndarray = vector_prices - dead_band_width
        
        is_above_band: np.ndarray = prices > upper_band
        is_below_band: np.ndarray = prices < lower_band
        is_signal_valid: np.ndarray = (is_above_band | is_below_band) & (vector_strengths >= self.min_vector_strength)
        
        result: Dict = {
            'upper_band': upper_band,
            'lower_band': lower_band,
            'dead_band_width': dead_band_width,
            'is_above_band': is_above_band,
            'is_below_band': is_below_band,
            'is_signal_valid': is_signal_valid,
            'signal_strength': vector_strengths * is_signal_valid.astype(float)
        }
        
        logger.debug(f"Adaptive zones: {np.sum(is_signal_valid)} valid signals out of {len(prices)} bars")
        
        return result
    
    def detect_regime(self, prices: np.ndarray, lookback: int = 30) -> Dict:
        """
        Identify current market regime: Trending vs Sideways vs Volatile
        
        Args:
            prices: Array of prices
            lookback: Number of bars to analyze (default 30)
        
        Returns:
            Dict with regime classification and metrics
        """
        recent_prices: np.ndarray = prices[-lookback:]
        
        volatility: float = np.std(np.diff(recent_prices) / recent_prices[:-1])
        
        slope, intercept, r_val, p_val, std_err = stats.linregress(
            np.arange(lookback), recent_prices
        )
        
        trend_strength: float = abs(slope) / np.mean(recent_prices)
        
        if p_val < 0.05 and trend_strength > 0.001:
            regime: str = 'TRENDING'
        elif volatility > 0.02:
            regime = 'VOLATILE'
        else:
            regime = 'SIDEWAYS'
        
        result: Dict = {
            'regime': regime,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'p_value': p_val,
            'regime_confidence': 1.0 - p_val if regime == 'TRENDING' else 1.0 - volatility
        }
        
        logger.info(f"Regime detected: {regime} (vol={volatility:.4f}, trend={trend_strength:.4f}, p={p_val:.4f})")
        
        return result
    
    def is_breakout_signal(self, price: float, vector_price: float, atr: float, 
                          vector_strength: float, regime: str) -> Dict:
        """
        Multi-factor confirmation:
        1. Price clears dead band (momentum)
        2. Vector strength is high (confidence)
        3. Regime is favorable (not sideways noise)
        
        Args:
            price: Current price
            vector_price: Vector line price
            atr: Average True Range
            vector_strength: Signal confidence (0-1)
            regime: Current regime (TRENDING, VOLATILE, SIDEWAYS)
        
        Returns:
            Dict with signal confirmation metrics
        """
        dead_band: float = self.atr_multiplier * atr
        
        clears_band: bool = abs(price - vector_price) > dead_band
        strong_vector: bool = vector_strength >= self.min_vector_strength
        favorable_regime: bool = regime != 'SIDEWAYS'
        
        all_confirmed: bool = clears_band and strong_vector and favorable_regime
        
        result: Dict = {
            'is_signal': all_confirmed,
            'clears_dead_band': clears_band,
            'vector_strength_ok': strong_vector,
            'regime_ok': favorable_regime,
            'signal_quality': float(all_confirmed) * vector_strength,
            'regime': regime
        }
        
        if all_confirmed:
            logger.info(f"BREAKOUT SIGNAL: price={price:.2f}, vector={vector_price:.2f}, strength={vector_strength:.3f}, regime={regime}")
        else:
            logger.debug(f"Signal rejected: band={clears_band}, vector={strong_vector}, regime={favorable_regime}")
        
        return result
    
    def calculate_dynamic_stop(self, entry_price: float, vector_price: float, 
                              atr: float, side: str = 'long') -> Dict:
        """
        Stop loss scales with volatility and position in vector zone.
        Wider stops in high volatility, tighter in low volatility.
        
        Args:
            entry_price: Entry price
            vector_price: Vector line price
            atr: Average True Range
            side: Trade direction ('long' or 'short')
        
        Returns:
            Dict with stop loss metrics
        """
        if side == 'long':
            stop: float = entry_price - (self.atr_multiplier * atr)
            stop = min(stop, vector_price - atr)
        else:
            stop = entry_price + (self.atr_multiplier * atr)
            stop = max(stop, vector_price + atr)
        
        risk_distance: float = abs(entry_price - stop)
        
        result: Dict = {
            'stop_price': stop,
            'risk_distance': risk_distance,
            'atr_based': self.atr_multiplier * atr,
            'side': side
        }
        
        logger.info(f"Dynamic stop: entry={entry_price:.2f}, stop={stop:.2f}, risk_dist={risk_distance:.2f} ({side})")
        
        return result
