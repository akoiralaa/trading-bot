import numpy as np
import pandas as pd


class MultiTimeframeAnalyzer:
    """Analyze price action across multiple timeframes."""
    
    def __init__(self):
        self.daily_vector = None
        self.hourly_vector = None
    
    def calculate_daily_vector(self, df_daily: pd.DataFrame, lookback: int = 20) -> np.ndarray:
        """Calculate vector on daily timeframe."""
        high = df_daily['high'].values
        low = df_daily['low'].values
        close = df_daily['close'].values
        
        vector = np.zeros(len(df_daily))
        
        for i in range(lookback, len(df_daily)):
            recent_high = np.max(high[i - lookback:i])
            recent_low = np.min(low[i - lookback:i])
            
            midpoint = (recent_high + recent_low) / 2
            recent_range = recent_high - recent_low
            current_distance_from_low = close[i] - recent_low
            
            if recent_range > 0:
                price_position = current_distance_from_low / recent_range
            else:
                price_position = 0.5
            
            momentum_adjustment = (price_position - 0.5) * (recent_range * 0.1)
            vector_level = midpoint + momentum_adjustment
            
            if i > lookback:
                alpha = 0.7
                vector[i] = alpha * vector_level + (1 - alpha) * vector[i - 1]
            else:
                vector[i] = vector_level
        
        self.daily_vector = vector
        return vector
    
    def calculate_hourly_vector(self, df_hourly: pd.DataFrame, lookback: int = 20) -> np.ndarray:
        """Calculate vector on hourly timeframe."""
        high = df_hourly['high'].values
        low = df_hourly['low'].values
        close = df_hourly['close'].values
        
        vector = np.zeros(len(df_hourly))
        
        for i in range(lookback, len(df_hourly)):
            recent_high = np.max(high[i - lookback:i])
            recent_low = np.min(low[i - lookback:i])
            
            midpoint = (recent_high + recent_low) / 2
            recent_range = recent_high - recent_low
            current_distance_from_low = close[i] - recent_low
            
            if recent_range > 0:
                price_position = current_distance_from_low / recent_range
            else:
                price_position = 0.5
            
            momentum_adjustment = (price_position - 0.5) * (recent_range * 0.1)
            vector_level = midpoint + momentum_adjustment
            
            if i > lookback:
                alpha = 0.7
                vector[i] = alpha * vector_level + (1 - alpha) * vector[i - 1]
            else:
                vector[i] = vector_level
        
        self.hourly_vector = vector
        return vector
    
    def get_vector_strength(self, close: np.ndarray, vector: np.ndarray, lookback: int = 5) -> np.ndarray:
        """Calculate strength metric."""
        strength = np.zeros(len(close))
        
        for i in range(lookback, len(close)):
            bars_above = np.sum(close[i-lookback:i] > vector[i-lookback:i])
            bars_below = lookback - bars_above
            
            if bars_above > bars_below:
                strength[i] = bars_above / lookback
            else:
                strength[i] = -(bars_below / lookback)
        
        return strength
    
    def get_daily_bias(self, daily_vector: np.ndarray, daily_close: np.ndarray, lookback: int = 20) -> str:
        """
        Determine daily trend.
        Only return BULLISH if clearly bullish (80%+ above).
        Otherwise NEUTRAL to allow more trades.
        """
        recent_close = daily_close[-lookback:]
        recent_vector = daily_vector[-lookback:]
        
        bars_above = np.sum(recent_close > recent_vector)
        
        if bars_above >= 16:  # 80%+ above = strong bullish
            return 'BULLISH'
        else:
            return 'NEUTRAL'  # Default to neutral, allow trading
    
    def confirm_entry(self, hourly_signal: bool, daily_bias: str) -> bool:
        """
        Confirmation logic:
        - BULLISH daily: Take all hourly signals (high confidence)
        - NEUTRAL daily: Take hourly signals (medium confidence)
        - Never take if daily is extremely bearish
        """
        if not hourly_signal:
            return False
        
        if daily_bias == 'BULLISH':
            return True  # Full confidence
        elif daily_bias == 'NEUTRAL':
            return True  # Still trade, just be aware
        
        return False
