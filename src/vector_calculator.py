import numpy as np
import pandas as pd


class VectorCalculator:
    """
    Calculate Wave 7 vector - dynamic support/resistance line.
    
    Above vector = bullish (price needs to stay here for bulls to maintain control)
    Below vector = bearish (price needs to stay here for bears to maintain control)
    """
    
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period
    
    def calculate_vector(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate Wave 7 vector using recent highs and lows.
        
        Vector = midpoint between recent high and recent low
        Smoothed with exponential weighting so recent bars matter more
        """
        high = df['high'].values
        low = df['low'].values
        
        vector = np.zeros(len(df))
        
        for i in range(self.lookback_period, len(df)):
            # Get recent price range
            recent_high = np.max(high[i - self.lookback_period:i])
            recent_low = np.min(low[i - self.lookback_period:i])
            
            # Vector = midpoint (fair value)
            vector_level = (recent_high + recent_low) / 2
            
            # Smooth it
            if i > self.lookback_period:
                vector[i] = 0.7 * vector_level + 0.3 * vector[i-1]
            else:
                vector[i] = vector_level
        
        return vector
    
    def is_above_vector(self, price: float, vector: float) -> bool:
        """Check if price is above vector (bullish)"""
        return price > vector
    
    def is_below_vector(self, price: float, vector: float) -> bool:
        """Check if price is below vector (bearish)"""
        return price < vector
