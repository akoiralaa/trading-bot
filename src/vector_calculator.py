import numpy as np
import pandas as pd


class VectorCalculator:
    """Calculate Wave 7 vector - dynamic support/resistance line."""
    
    def __init__(self, wave_period: int = 7, lookback: int = 20):
        self.wave_period = wave_period
        self.lookback = lookback
    
    def calculate_vector(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate Wave 7 vector."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        vector = np.zeros(len(df))
        
        for i in range(self.lookback, len(df)):
            recent_high = np.max(high[i - self.lookback:i])
            recent_low = np.min(low[i - self.lookback:i])
            
            midpoint = (recent_high + recent_low) / 2
            
            recent_range = recent_high - recent_low
            current_distance_from_low = close[i] - recent_low
            
            if recent_range > 0:
                price_position = current_distance_from_low / recent_range
            else:
                price_position = 0.5
            
            momentum_adjustment = (price_position - 0.5) * (recent_range * 0.1)
            vector_level = midpoint + momentum_adjustment
            
            if i > self.lookback:
                alpha = 0.7
                vector[i] = alpha * vector_level + (1 - alpha) * vector[i - 1]
            else:
                vector[i] = vector_level
        
        return vector
    
    def get_vector_strength(self, df: pd.DataFrame, vector: np.ndarray, lookback: int = 5) -> np.ndarray:
        """
        Measure how strongly price is maintaining above/below vector.
        
        Returns: Array from -1 (very bearish) to +1 (very bullish)
        """
        close = df['close'].values
        strength = np.zeros(len(close))
        
        for i in range(lookback, len(close)):
            bars_above = np.sum(close[i-lookback:i] > vector[i-lookback:i])
            bars_below = lookback - bars_above
            
            if bars_above > bars_below:
                strength[i] = bars_above / lookback
            else:
                strength[i] = -(bars_below / lookback)
        
        return strength
    
    def is_above_vector(self, price: float, vector: float) -> bool:
        return price > vector
    
    def is_below_vector(self, price: float, vector: float) -> bool:
        return price < vector


class PivotCalculator:
    """Calculate Pivot Line."""
    
    def __init__(self, period: int = 20):
        self.period = period
    
    def calculate_pivot(self, df: pd.DataFrame) -> np.ndarray:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        pivot = np.zeros(len(df))
        
        for i in range(self.period, len(df)):
            period_high = np.max(high[i - self.period:i])
            period_low = np.min(low[i - self.period:i])
            period_close = close[i]
            
            pivot_level = (period_high + period_low + period_close) / 3
            
            if i > self.period:
                pivot[i] = 0.8 * pivot_level + 0.2 * pivot[i - 1]
            else:
                pivot[i] = pivot_level
        
        return pivot
    
    def is_above_pivot(self, price: float, pivot: float) -> bool:
        return price > pivot
    
    def is_below_pivot(self, price: float, pivot: float) -> bool:
        return price < pivot
