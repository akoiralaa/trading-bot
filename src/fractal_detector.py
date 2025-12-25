import numpy as np
import pandas as pd
from typing import List, Tuple


class FractalDetector:
    """
    Detect fractals and cluster them into zones.
    
    Fractal = local high or low (5-bar pattern)
    Cluster = multiple fractals within 10% price range = high probability zone
    """
    
    def __init__(self, cluster_threshold: float = 0.10):
        self.cluster_threshold = cluster_threshold
    
    def detect_fractals(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect fractal highs and lows.
        
        Fractal High: bar higher than 2 bars before AND 2 bars after
        Fractal Low: bar lower than 2 bars before AND 2 bars after
        """
        high = df['high'].values
        low = df['low'].values
        
        fractal_highs = np.zeros(len(df))
        fractal_lows = np.zeros(len(df))
        
        for i in range(2, len(df) - 2):
            # Fractal High
            if (high[i] > high[i-2] and high[i] > high[i-1] and 
                high[i] > high[i+1] and high[i] > high[i+2]):
                fractal_highs[i] = high[i]
            
            # Fractal Low
            if (low[i] < low[i-2] and low[i] < low[i-1] and 
                low[i] < low[i+1] and low[i] < low[i+2]):
                fractal_lows[i] = low[i]
        
        return fractal_highs, fractal_lows
    
    def cluster_fractals(self, fractals: np.ndarray) -> List[Tuple[float, float, int]]:
        """
        Group fractals into clusters if within 10% of each other.
        
        Returns:
            List of (cluster_low, cluster_high, fractal_count)
        """
        # Get non-zero fractals
        fractal_values = fractals[fractals > 0]
        
        if len(fractal_values) == 0:
            return []
        
        clusters = []
        sorted_fractals = np.sort(fractal_values)
        current_cluster = [sorted_fractals[0]]
        
        for i in range(1, len(sorted_fractals)):
            cluster_avg = np.mean(current_cluster)
            pct_distance = abs(sorted_fractals[i] - cluster_avg) / cluster_avg
            
            if pct_distance <= self.cluster_threshold:
                # Add to current cluster
                current_cluster.append(sorted_fractals[i])
            else:
                # Save cluster and start new one
                clusters.append((
                    min(current_cluster),
                    max(current_cluster),
                    len(current_cluster)
                ))
                current_cluster = [sorted_fractals[i]]
        
        # Save last cluster
        if len(current_cluster) > 0:
            clusters.append((
                min(current_cluster),
                max(current_cluster),
                len(current_cluster)
            ))
        
        return clusters
    
    def get_resistance_and_support(self, fractal_highs: np.ndarray, fractal_lows: np.ndarray) -> Tuple[List, List]:
        """
        Get resistance clusters (from highs) and support clusters (from lows).
        """
        resistance = self.cluster_fractals(fractal_highs)
        support = self.cluster_fractals(fractal_lows)
        
        return resistance, support
