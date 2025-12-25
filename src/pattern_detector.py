import numpy as np
import pandas as pd


class PatternDetector:
    """
    Detect entry patterns: Table Top A and Table Top B.
    
    Table Top A: Price taps vector from above, then breaks back above (bullish)
    Table Top B: Price dips below vector, then breaks back above (bullish)
    
    These patterns confirm entry after exhaustion phase.
    """
    
    def __init__(self, tolerance: float = 0.002):
        """
        tolerance: How close price needs to be to vector to count as "tap" (0.2%)
        """
        self.tolerance = tolerance
    
    def detect_table_top_b(self, df: pd.DataFrame, vector: np.ndarray, lookback: int = 5) -> np.ndarray:
        """
        Table Top B Pattern:
        1. Price dips BELOW vector
        2. Then reverses and breaks BACK ABOVE vector
        3. This is the entry signal
        
        Returns: Array with 1 where Table Top B forms, 0 elsewhere
        """
        close = df['close'].values
        low = df['low'].values
        
        signals = np.zeros(len(df))
        
        for i in range(lookback, len(df)):
            # Check if price recently dipped below vector
            dipped_below = np.any(low[i-lookback:i] < vector[i-lookback:i])
            
            # Check if current bar is now above vector
            above_now = close[i] > vector[i]
            
            # Check if previous bar was below/near vector
            was_below = close[i-1] <= vector[i-1] * (1 + self.tolerance)
            
            if dipped_below and above_now and was_below:
                signals[i] = 1
        
        return signals
    
    def detect_table_top_a(self, df: pd.DataFrame, vector: np.ndarray, lookback: int = 5) -> np.ndarray:
        """
        Table Top A Pattern:
        1. Price taps TOP of vector from above
        2. Bounces down slightly
        3. Then breaks back above vector
        4. This is the entry signal (weaker than B)
        
        Returns: Array with 1 where Table Top A forms, 0 elsewhere
        """
        close = df['close'].values
        high = df['high'].values
        
        signals = np.zeros(len(df))
        
        for i in range(lookback, len(df)):
            # Check if price tapped vector (high touched it)
            tapped_vector = np.any(high[i-lookback:i] >= vector[i-lookback:i] * (1 - self.tolerance))
            
            # Check if currently above vector
            above_now = close[i] > vector[i]
            
            # Check if previous bar was near/at vector (avoid divide by zero)
            if vector[i-1] > 0:
                was_at_vector = abs(close[i-1] - vector[i-1]) / vector[i-1] <= self.tolerance
            else:
                was_at_vector = False
            
            if tapped_vector and above_now and was_at_vector:
                signals[i] = 1
        
        return signals
    
    def detect_exhaustion(self, df: pd.DataFrame, vector: np.ndarray, lookback: int = 5) -> np.ndarray:
        """
        Exhaustion Signal: Price struggles to stay above vector.
        
        Signs of exhaustion:
        1. Multiple failed attempts to break above vector
        2. Price keeps testing but can't hold
        3. Precedes big reversals
        
        Returns: Array with 1 where exhaustion detected, 0 elsewhere
        """
        close = df['close'].values
        
        signals = np.zeros(len(df))
        
        for i in range(lookback, len(df)):
            # Count how many times price touched vector but failed
            touches = 0
            for j in range(i - lookback, i):
                if vector[j] > 0 and abs(close[j] - vector[j]) / vector[j] <= self.tolerance:
                    touches += 1
            
            # Exhaustion = multiple touches with no clean break
            if touches >= 2:
                signals[i] = 1
        
        return signals
