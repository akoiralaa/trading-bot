import numpy as np
import pandas as pd


class PatternDetector:
    """Detect entry patterns with strength confirmation."""
    
    def __init__(self, tolerance: float = 0.002):
        self.tolerance = tolerance
    
    def detect_table_top_b(self, df: pd.DataFrame, vector: np.ndarray, 
                          vector_strength: np.ndarray, lookback: int = 5) -> np.ndarray:
        """Table Top B with STRICT strength confirmation."""
        close = df['close'].values
        low = df['low'].values
        
        signals = np.zeros(len(df))
        
        for i in range(lookback, len(df)):
            dipped_below = np.any(low[i-lookback:i] < vector[i-lookback:i])
            above_now = close[i] > vector[i]
            was_below = close[i-1] <= vector[i-1] * (1 + self.tolerance)
            
            # STRICT: Only very strong bullish signals (>0.6)
            strength_strong = vector_strength[i] > 0.6
            
            if dipped_below and above_now and was_below and strength_strong:
                signals[i] = 1
        
        return signals
    
    def detect_table_top_a(self, df: pd.DataFrame, vector: np.ndarray,
                          vector_strength: np.ndarray, lookback: int = 5) -> np.ndarray:
        """Table Top A with STRICT strength confirmation."""
        close = df['close'].values
        high = df['high'].values
        
        signals = np.zeros(len(df))
        
        for i in range(lookback, len(df)):
            tapped_vector = np.any(high[i-lookback:i] >= vector[i-lookback:i] * (1 - self.tolerance))
            above_now = close[i] > vector[i]
            
            if vector[i-1] > 0:
                was_at_vector = abs(close[i-1] - vector[i-1]) / vector[i-1] <= self.tolerance
            else:
                was_at_vector = False
            
            # STRICT: Only very strong bullish signals (>0.6)
            strength_strong = vector_strength[i] > 0.6
            
            if tapped_vector and above_now and was_at_vector and strength_strong:
                signals[i] = 1
        
        return signals
