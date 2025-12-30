import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Implements a volatility-adaptive signal filter and market state classifier.
    
    Utilizes ATR-weighted 'Dead Bands' to differentiate between stochastic noise 
    and structural momentum. Employs OLS regression for objective regime 
    classification and statistical significance testing.
    """
    
    def __init__(self, atr_multiplier: float = 2.0, min_vector_strength: float = 0.51) -> None:
        """
        Initializes detector with volatility-scaling and confidence parameters.
        
        Args:
            atr_multiplier: Width of the volatility-adjusted dead band.
            min_vector_strength: Lower bound for signal probability threshold.
        """
        self.atr_multiplier = atr_multiplier
        self.min_vector_strength = min_vector_strength
        logger.info(f"RegimeEngine: ATR_Mult={atr_multiplier}, StrengthThreshold={min_vector_strength}")
    
    def calculate_adaptive_zones(self, prices: np.ndarray, atr: np.ndarray, 
                                vector: np.ndarray, strengths: np.ndarray) -> Dict:
        """
        Derives dynamic dead bands around the primary vector line.
        
        Signals are categorized as valid only upon breaching the $\pm(k \cdot ATR)$ 
        threshold, effectively filtering out sub-volatility mean-reversion.
        """
        band_offset = self.atr_multiplier * atr
        
        upper_bound = vector + band_offset
        lower_bound = vector - band_offset
        
        # Binary state mapping for breakout conditions
        valid_mask = ((prices > upper_bound) | (prices < lower_bound)) & (strengths >= self.min_vector_strength)
        
        return {
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'is_valid_signal': valid_mask,
            'filtered_strength': strengths * valid_mask.astype(float)
        }
    
    def detect_regime(self, prices: np.ndarray, lookback: int = 30) -> Dict:
        """
        Classifies the current state of the price process via OLS regression.
        
        State space:
        - TRENDING: Statistical significance ($p < 0.05$) with localized slope.
        - VOLATILE: Elevated standard deviation of logarithmic returns.
        - SIDEWAYS: Low-momentum consolidation within stochastic noise.
        """
        window = prices[-lookback:]
        returns = np.diff(np.log(window))
        volatility = np.std(returns)
        
        # Ordinary Least Squares (OLS) for trend estimation
        x = np.arange(lookback)
        slope, _, _, p_val, _ = stats.linregress(x, window)
        
        # Normalized drift estimate
        drift = abs(slope) / np.mean(window)
        
        # State classification logic
        if p_val < 0.05 and drift > 0.001:
            state = 'TRENDING'
        elif volatility > 0.02: # Log-return volatility threshold
            state = 'VOLATILE'
        else:
            state = 'SIDEWAYS'
        
        logger.info(f"MarketState: {state} | Drift: {drift:.5f} | Vol: {volatility:.4f} | P: {p_val:.4f}")
        
        return {
            'state': state,
            'volatility': volatility,
            'p_value': p_val,
            'confidence': 1.0 - p_val if state == 'TRENDING' else 1.0 - volatility
        }

    

    def validate_execution_signal(self, price: float, vector: float, atr: float, 
                                 strength: float, state: str) -> Dict:
        """
        Performs multi-factor confirmation of directional alpha.
        
        Validates the confluence of volatility-weighted clearance, 
        probabilistic confidence, and favorable macro-state (State != SIDEWAYS).
        """
        band_width = self.atr_multiplier * atr
        clears_dead_band = abs(price - vector) > band_width
        
        is_confirmed = (clears_dead_band and 
                        strength >= self.min_vector_strength and 
                        state != 'SIDEWAYS')
        
        return {
            'is_confirmed': is_confirmed,
            'dead_band_breach': clears_dead_band,
            'state': state,
            'quality_score': float(is_confirmed) * strength
        }

    def get_volatility_adjusted_stop(self, entry: float, vector: float, 
                                    atr: float, side: str = 'long') -> Dict:
        """
        Calculates dynamic protective stops based on local volatility.
        
        Adjusts the risk-distance to ensure stops reside outside the 
        stochastic noise zone defined by ATR and the structural vector line.
        """
        offset = self.atr_multiplier * atr
        
        if side == 'long':
            # Stop must be below vector and outside ATR noise
            stop = min(entry - offset, vector - atr)
        else:
            # Stop must be above vector and outside ATR noise
            stop = max(entry + offset, vector + atr)
        
        return {
            'stop_price': stop,
            'risk_per_share': abs(entry - stop),
            'volatility_buffer': offset
        }