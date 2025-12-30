import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from scipy import stats
from dataclasses import dataclass


@dataclass
class VectorZoneWithStats:
    """Vector zone with statistical significance metrics"""
    upper_band: float
    middle_line: float
    lower_band: float
    zone_width: float
    atr: float
    atr_multiplier: float
    volume_multiplier: float
    effective_multiplier: float
    
    # Statistical significance
    slope: float
    intercept: float
    r_squared: float
    slope_std_error: float
    t_statistic: float
    p_value: float
    is_significant: bool  # p_value < 0.05
    significance_level: str  # 'high', 'medium', 'low', 'not_significant'
    
    # Volume metrics
    current_volume: float
    avg_volume: float
    volume_ratio: float
    
    # Metadata
    timestamp: int
    lookback_period: int
    num_observations: int


class StatisticalVectorZoneDetector:

    def __init__(self, 
                 lookback_period: int = 20,
                 base_atr_multiplier: float = 1.0,
                 volume_lookback: int = 50,
                 significance_threshold: float = 0.05):
       
        self.lookback_period = lookback_period
        self.base_atr_multiplier = base_atr_multiplier
        self.volume_lookback = volume_lookback
        self.significance_threshold = significance_threshold
    
    def calculate_true_range(self, 
                            high: np.ndarray, 
                            low: np.ndarray, 
                            close: np.ndarray) -> np.ndarray:
        """True Range for volatility"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        return tr
    
    def calculate_atr(self, 
                     high: np.ndarray, 
                     low: np.ndarray, 
                     close: np.ndarray,
                     period: int = 14) -> np.ndarray:
        """Average True Range"""
        tr = self.calculate_true_range(high, low, close)
        atr = pd.Series(tr).rolling(period).mean().values
        return atr
    
    def fit_vector_with_stats(self, 
                             prices: np.ndarray) -> Dict:
   
        n = len(prices)
        if n < 3:
            return {
                'slope': 0,
                'intercept': prices[-1] if len(prices) > 0 else 0,
                'r_squared': 0,
                'slope_std_error': np.inf,
                't_statistic': 0,
                'p_value': 1.0,
                'is_significant': False,
                'residual_std': 0,
                'num_observations': n
            }
        
        # X = [0, 1, 2, ..., n-1] (time indices)
        x = np.arange(n)
        
        # Calculate regression using numpy
        coeffs = np.polyfit(x, prices, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Calculate fitted values and residuals
        y_pred = slope * x + intercept
        residuals = prices - y_pred
        
        # Calculate R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((prices - np.mean(prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate standard error of slope
        # SE(slope) = sqrt(SS_res / (n-2)) / sqrt(SS_x)
        ms_res = ss_res / (n - 2) if n > 2 else np.inf
        ss_x = np.sum((x - np.mean(x)) ** 2)
        
        if ss_x > 0:
            slope_std_error = np.sqrt(ms_res / ss_x)
        else:
            slope_std_error = np.inf
        
        # Calculate t-statistic
        # t = slope / SE(slope)
        # Under H0 (slope=0), follows t-distribution with n-2 df
        if slope_std_error > 0:
            t_stat = slope / slope_std_error
        else:
            t_stat = 0
        
        # Calculate p-value (two-tailed)
        df = n - 2
        if df > 0:
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        else:
            p_value = 1.0
        
        is_significant = p_value < self.significance_threshold
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'slope_std_error': slope_std_error,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': is_significant,
            'residual_std': np.sqrt(ms_res),
            'num_observations': n
        }
    
    def get_significance_level(self, p_value: float) -> str:
        """Categorize significance level"""
        if p_value < 0.01:
            return 'high'  # p < 0.01 (99% confidence)
        elif p_value < 0.05:
            return 'medium'  # p < 0.05 (95% confidence)
        elif p_value < 0.10:
            return 'low'  # p < 0.10 (90% confidence)
        else:
            return 'not_significant'  # Not significant
    
    def get_volume_multiplier(self,
                             current_volume: float,
                             avg_volume: float,
                             max_multiplier: float = 3.0,
                             min_multiplier: float = 0.5) -> Tuple[float, Dict]:
       
        if current_volume <= 0:
            return 1.0, {'volume_ratio': np.inf, 'reason': 'zero_volume'}
        
        volume_ratio = avg_volume / current_volume
        log_ratio = np.log(max(volume_ratio, 0.1))
        multiplier = 1.0 + (log_ratio / 2.3)
        multiplier = np.clip(multiplier, min_multiplier, max_multiplier)
        
        metrics = {
            'volume_ratio': volume_ratio,
            'multiplier': multiplier,
        }
        
        return multiplier, metrics
    
    def get_vector_zone(self,
                       high: np.ndarray,
                       low: np.ndarray,
                       close: np.ndarray,
                       volume: np.ndarray,
                       atr_values: np.ndarray,
                       bar_index: int) -> VectorZoneWithStats:
    
        # Get lookback period
        start_idx = max(0, bar_index - self.lookback_period)
        end_idx = bar_index + 1
        
        # Extract prices for vector line
        lookback_highs = high[start_idx:end_idx]
        
        # FIT VECTOR WITH STATISTICS (NEW)
        stats_dict = self.fit_vector_with_stats(lookback_highs)
        
        slope = stats_dict['slope']
        intercept = stats_dict['intercept']
        t_stat = stats_dict['t_statistic']
        p_value = stats_dict['p_value']
        is_significant = stats_dict['is_significant']
        
        # Project vector at current bar
        bars_back = bar_index - start_idx
        vector_value = slope * bars_back + intercept
        
        # Get ATR
        atr = atr_values[bar_index] if bar_index < len(atr_values) else np.mean(atr_values[-5:])
        
        # Get volume metrics
        vol_start = max(0, bar_index - self.volume_lookback)
        avg_volume = np.mean(volume[vol_start:bar_index + 1])
        current_volume = volume[bar_index]
        vol_multiplier, vol_metrics = self.get_volume_multiplier(current_volume, avg_volume)
        
        # Zone width with volume adaptation
        effective_multiplier = self.base_atr_multiplier * vol_multiplier
        zone_width = atr * effective_multiplier
        
        upper_band = vector_value + zone_width
        lower_band = vector_value - zone_width
        
        # Determine significance level
        sig_level = self.get_significance_level(p_value)
        
        return VectorZoneWithStats(
            upper_band=upper_band,
            middle_line=vector_value,
            lower_band=lower_band,
            zone_width=zone_width,
            atr=atr,
            atr_multiplier=self.base_atr_multiplier,
            volume_multiplier=vol_multiplier,
            effective_multiplier=effective_multiplier,
            
            # NEW: Statistical significance
            slope=slope,
            intercept=intercept,
            r_squared=stats_dict['r_squared'],
            slope_std_error=stats_dict['slope_std_error'],
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=is_significant,
            significance_level=sig_level,
            
            current_volume=current_volume,
            avg_volume=avg_volume,
            volume_ratio=vol_metrics['volume_ratio'],
            timestamp=bar_index,
            lookback_period=self.lookback_period,
            num_observations=len(lookback_highs)
        )
    
    def check_bounce_signal(self,
                           price_prev: float,
                           price_curr: float,
                           zone: VectorZoneWithStats,
                           side: str = 'long',
                           min_bounce_pct: float = 0.001,
                           require_significance: bool = True) -> Tuple[bool, str, Dict]:
    
        # NEW: Filter out non-significant zones
        if require_significance and not zone.is_significant:
            return False, f"Zone lacks statistical significance (p={zone.p_value:.3f})", {
                'p_value': zone.p_value,
                'is_significant': False,
                'reason': 'slope_not_significant'
            }
        
        if side == 'long':
            below_lower = price_prev < zone.lower_band
            above_middle = price_curr > zone.middle_line
            
            bounce_magnitude = (price_curr - price_prev) / price_prev if price_prev > 0 else 0
            has_momentum = bounce_magnitude > min_bounce_pct
            
            if below_lower and above_middle and has_momentum:
                confidence = {
                    'p_value': zone.p_value,
                    't_statistic': zone.t_statistic,
                    'significance': zone.significance_level,
                    'r_squared': zone.r_squared,
                    'volume_regime': 'thin' if zone.volume_ratio > 1.5 else 'normal',
                    'bounce_magnitude': bounce_magnitude,
                    'signal_quality': self._rate_signal_quality(zone)
                }
                
                reason = (f"Long bounce: {price_prev:.2f} < {zone.lower_band:.2f} → "
                         f"{price_curr:.2f} > {zone.middle_line:.2f} "
                         f"(slope p-value={zone.p_value:.3f}, t={zone.t_statistic:.2f})")
                
                return True, reason, confidence
            else:
                return False, f"No valid long bounce", {'bounce_magnitude': bounce_magnitude}
        
        elif side == 'short':
            above_upper = price_prev > zone.upper_band
            below_middle = price_curr < zone.middle_line
            
            bounce_magnitude = (price_prev - price_curr) / price_prev if price_prev > 0 else 0
            has_momentum = bounce_magnitude > min_bounce_pct
            
            if above_upper and below_middle and has_momentum:
                confidence = {
                    'p_value': zone.p_value,
                    't_statistic': zone.t_statistic,
                    'significance': zone.significance_level,
                    'r_squared': zone.r_squared,
                    'volume_regime': 'thin' if zone.volume_ratio > 1.5 else 'normal',
                    'bounce_magnitude': bounce_magnitude,
                    'signal_quality': self._rate_signal_quality(zone)
                }
                
                reason = (f"Short bounce: {price_prev:.2f} > {zone.upper_band:.2f} → "
                         f"{price_curr:.2f} < {zone.middle_line:.2f} "
                         f"(slope p-value={zone.p_value:.3f}, t={zone.t_statistic:.2f})")
                
                return True, reason, confidence
            else:
                return False, f"No valid short bounce", {'bounce_magnitude': bounce_magnitude}
        
        else:
            raise ValueError(f"Unknown side: {side}")
    
    def _rate_signal_quality(self, zone: VectorZoneWithStats) -> str:
      
        # Significance score (higher p-value = lower score)
        if zone.p_value < 0.01:
            stat_score = 1.0
        elif zone.p_value < 0.05:
            stat_score = 0.8
        elif zone.p_value < 0.10:
            stat_score = 0.6
        else:
            stat_score = 0.3
        
        # R² score (higher = better fit)
        r2_score = min(zone.r_squared * 2, 1.0)  # Scale 0-1
        
        # Volume score (normal volume = higher)
        if zone.volume_ratio < 1.1:
            vol_score = 1.0
        elif zone.volume_ratio < 1.5:
            vol_score = 0.8
        else:
            vol_score = 0.6
        
        # Combined score
        combined = (stat_score * 0.5) + (r2_score * 0.3) + (vol_score * 0.2)
        
        if combined > 0.85:
            return 'high'
        elif combined > 0.70:
            return 'medium'
        elif combined > 0.50:
            return 'low'
        else:
            return 'poor'
    
    def get_zone_quality_score(self, zone: VectorZoneWithStats) -> float:
     
        # Significance penalty
        if zone.p_value < 0.05:
            sig_score = 1.0
        elif zone.p_value < 0.10:
            sig_score = 0.7
        else:
            sig_score = 0.4  # Not significant = low quality
        
        # Volume score
        volume_score = 1.0 / (1.0 + max(0, zone.volume_ratio - 1.0))
        
        # Fit quality
        r2_score = min(zone.r_squared, 1.0)
        
        # Combined
        combined = (sig_score * 0.5) + (volume_score * 0.3) + (r2_score * 0.2)
        
        return np.clip(combined, 0.2, 1.0)


# Example usage and testing
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    np.random.seed(42)
    
    print("\n=== Testing Statistical Vector Zone Detector ===\n")
    
    # Scenario 1: Strong trend (should be significant)
    print("SCENARIO 1: Strong Uptrend")
    strong_trend = 100 + np.arange(20) * 0.3  # Clear upward trend
    noise = np.random.normal(0, 0.1, 20)
    prices_strong = strong_trend + noise
    
    detector = StatisticalVectorZoneDetector()
    stats_strong = detector.fit_vector_with_stats(prices_strong)
    
    print(f"  Slope: {stats_strong['slope']:.4f}")
    print(f"  t-stat: {stats_strong['t_statistic']:.2f}")
    print(f"  p-value: {stats_strong['p_value']:.4f}")
    print(f"  Significant? {stats_strong['is_significant']}")
    print(f"  R²: {stats_strong['r_squared']:.3f}")
    
    # Scenario 2: Random walk (should NOT be significant)
    print("\nSCENARIO 2: Random Walk (No Trend)")
    random_walk = 100 + np.cumsum(np.random.normal(0, 0.2, 20))
    
    stats_random = detector.fit_vector_with_stats(random_walk)
    
    print(f"  Slope: {stats_random['slope']:.4f}")
    print(f"  t-stat: {stats_random['t_statistic']:.2f}")
    print(f"  p-value: {stats_random['p_value']:.4f}")
    print(f"  Significant? {stats_random['is_significant']}")
    print(f"  R²: {stats_random['r_squared']:.3f}")
    
    # Scenario 3: Weak trend (borderline significant)
    print("\nSCENARIO 3: Weak Trend (Borderline)")
    weak_trend = 100 + np.arange(20) * 0.05
    noise_weak = np.random.normal(0, 0.3, 20)
    prices_weak = weak_trend + noise_weak
    
    stats_weak = detector.fit_vector_with_stats(prices_weak)
    
    print(f"  Slope: {stats_weak['slope']:.4f}")
    print(f"  t-stat: {stats_weak['t_statistic']:.2f}")
    print(f"  p-value: {stats_weak['p_value']:.4f}")
    print(f"  Significant? {stats_weak['is_significant']}")
    print(f"  R²: {stats_weak['r_squared']:.3f}")
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    print(f"\nStrong trend: p-value = {stats_strong['p_value']:.4f}")
    print("  → Slope is significantly different from zero")
    print("  → Real directional bias, can trade it")
    
    print(f"\nRandom walk: p-value = {stats_random['p_value']:.4f}")
    print("  → Slope is NOT significantly different from zero")
    print("  → Could be random noise, avoid trading")
    
    print(f"\nWeak trend: p-value = {stats_weak['p_value']:.4f}")
    if stats_weak['p_value'] < 0.05:
        print("  → Just barely significant at 95% confidence")
        print("  → Can trade but with caution")
    else:
        print("  → Not quite significant at 95% confidence")
        print("  → Skip this one")
    
    print("\n" + "="*60)
    print("WHAT THIS MEANS FOR TRADING")
    print("="*60)
    
    print()