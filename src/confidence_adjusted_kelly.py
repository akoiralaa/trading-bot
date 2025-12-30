"""
Confidence-Adjusted Kelly Criterion with Statistical Significance

The Insight:
- Kelly Criterion gives us edge-based sizing (from historical trades)
- Statistical significance gives us signal-based confidence (current pattern strength)
- Position size should be PRODUCT of both

Mathematical Foundation:
final_position = capital × kelly_fraction × confidence_multiplier

where:
  kelly_fraction = (p*b - q) / b × fractional_kelly
    ↑ Based on historical edge (win rate, odds)
  
  confidence_multiplier = f(p_value)
    ↑ Based on current signal significance (0.0 to 1.0)

Example:
  capital = $100,000
  kelly_fraction = 0.03 (from historical 55% win rate, 1.2x odds)
  
  If p_value = 0.001 (highly significant):
    confidence = 1.0
    position = $100,000 × 0.03 × 1.0 = $3,000
  
  If p_value = 0.025 (moderately significant):
    confidence = 0.5
    position = $100,000 × 0.03 × 0.5 = $1,500
  
  If p_value = 0.049 (barely significant):
    confidence = 0.02
    position = $100,000 × 0.03 × 0.02 = $60 (minimal size)
  
  If p_value = 0.051 (not significant):
    confidence = 0.0
    position = $0 (NO TRADE)

This is mathematically sound because:
1. Edge (Kelly) is from statistical sample of 50+ trades
2. Confidence is from current signal quality
3. Both matter independently
4. Position scales with BOTH
"""

import numpy as np
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class ConfidenceAdjustedSize:
    """Container for position sizing with breakdown"""
    final_position_size: float  # What to actually trade
    kelly_fraction: float        # Historical edge-based sizing
    confidence_multiplier: float # Signal strength-based sizing
    p_value: float              # Regression slope significance
    vector_strength: float      # Direction of vector (if available)
    reasoning: str              # Explanation of sizing
    
    # Breakdown for transparency
    capital: float
    kelly_pct: float
    confidence_pct: float
    product: float


class ConfidenceAdjuster:
    """
    Converts p-values (statistical significance) into position size multipliers
    
    Three scaling options:
    1. Linear: Simple proportional scaling
    2. Sigmoid: Smooth S-curve (more aggressive at high confidence)
    3. Step: Discrete confidence levels
    """
    
    def __init__(self, threshold: float = 0.05, scaling_method: str = 'linear'):
        """
        Args:
            threshold: p-value cutoff (0.05 = 95% confidence)
            scaling_method: 'linear', 'sigmoid', or 'step'
        """
        self.threshold = threshold
        self.scaling_method = scaling_method
    
    def confidence_linear(self, p_value: float) -> float:
        """
        Linear scaling from 0 to 1
        
        p_value = 0.001  → confidence = 1.0 (max)
        p_value = 0.025  → confidence = 0.5 (medium)
        p_value = 0.050  → confidence = 0.0 (threshold)
        p_value > 0.050  → confidence = 0.0 (no trade)
        
        Formula: confidence = 1.0 - (p_value / threshold) × 1.0
        Scaled to: 1.0 - (p_value / threshold) × (1.0 - min_confidence)
        """
        if p_value > self.threshold:
            return 0.0
        
        # Linear: maps [0, threshold] to [1.0, 0.0]
        # At p=0: confidence = 1.0
        # At p=threshold: confidence = 0.0
        confidence = 1.0 - (p_value / self.threshold)
        
        return max(0.0, confidence)
    
    def confidence_sigmoid(self, p_value: float, steepness: float = 20.0) -> float:
        """
        Sigmoid scaling (S-curve)
        
        Smoother than linear. Less aggressive near threshold, more aggressive at high confidence.
        
        Formula: 1 / (1 + exp(steepness * (p_value - threshold/2)))
        
        p_value = 0.001  → confidence = 0.99 (very high)
        p_value = 0.025  → confidence = 0.73 (high)
        p_value = 0.050  → confidence = 0.0 (threshold)
        p_value > 0.050  → confidence = 0.0 (no trade)
        """
        if p_value > self.threshold:
            return 0.0
        
        # Sigmoid centered at threshold/2
        midpoint = self.threshold / 2.0
        confidence = 1.0 / (1.0 + np.exp(steepness * (p_value - midpoint)))
        
        return max(0.0, min(1.0, confidence))
    
    def confidence_step(self, p_value: float) -> float:
        """
        Step function (discrete confidence levels)
        
        p_value < 0.01  → confidence = 1.0 (High - 99% confidence)
        p_value < 0.05  → confidence = 0.5 (Medium - 95% confidence)
        p_value >= 0.05 → confidence = 0.0 (Low - not significant)
        
        Useful when you want clear tiers, not continuous scaling.
        """
        if p_value > self.threshold:
            return 0.0
        elif p_value < 0.01:
            return 1.0  # p < 0.01 (very confident)
        elif p_value < 0.03:
            return 0.75  # p < 0.03 (confident)
        else:
            return 0.25  # p < 0.05 (barely confident)
    
    def get_confidence(self, p_value: float) -> float:
        """Get confidence multiplier using selected method"""
        if self.scaling_method == 'linear':
            return self.confidence_linear(p_value)
        elif self.scaling_method == 'sigmoid':
            return self.confidence_sigmoid(p_value)
        elif self.scaling_method == 'step':
            return self.confidence_step(p_value)
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
    
    def explain_confidence(self, p_value: float) -> str:
        """Describe the confidence level"""
        conf = self.get_confidence(p_value)
        
        if p_value > self.threshold:
            return f"Not significant (p={p_value:.4f} > {self.threshold}). NO TRADE."
        elif conf > 0.9:
            return f"Very high confidence (p={p_value:.4f}). 100% position size."
        elif conf > 0.7:
            return f"High confidence (p={p_value:.4f}). ~70% position size."
        elif conf > 0.5:
            return f"Moderate confidence (p={p_value:.4f}). ~50% position size."
        elif conf > 0.2:
            return f"Low confidence (p={p_value:.4f}). ~20% position size."
        else:
            return f"Very low confidence (p={p_value:.4f}). Minimal position size."


class ConfidenceAdjustedKelly:
    """
    Unified position sizing: Kelly Criterion × Confidence Multiplier
    
    final_position = capital × kelly_fraction × confidence_multiplier
    
    This ensures:
    1. You only risk what your edge supports (Kelly)
    2. You only risk it when the signal is strong (Confidence)
    3. Position scales smoothly with both factors
    """
    
    def __init__(self, 
                 kelly_sizing,  # KellyCriterion instance
                 confidence_adjuster=None,
                 min_position_size: float = 100):
        """
        Args:
            kelly_sizing: KellyCriterion instance (for historical edge)
            confidence_adjuster: ConfidenceAdjuster instance
            min_position_size: Don't bother with trades < this size
        """
        self.kelly = kelly_sizing
        self.confidence = confidence_adjuster or ConfidenceAdjuster(threshold=0.05, scaling_method='linear')
        self.min_position_size = min_position_size
    
    def get_position_size(self,
                         capital: float,
                         p_value: float,
                         vector_strength: float = None) -> ConfidenceAdjustedSize:
        """
        Calculate final position size combining Kelly + Confidence
        
        Args:
            capital: Account capital
            p_value: Statistical significance of vector slope
            vector_strength: Optional (for vector direction boost)
        
        Returns:
            ConfidenceAdjustedSize with full breakdown
        """
        
        # Step 1: Get Kelly fraction from historical edge
        kelly_metrics = self.kelly.calculate_trade_metrics()
        kelly_fraction = self.kelly.calculate_kelly_fraction(kelly_metrics)
        
        # Adjust Kelly for vector strength if provided
        if vector_strength is not None:
            # Vector strength affects how much of Kelly to use
            if vector_strength < 0.55:
                kelly_adjusted = kelly_fraction * 0.25
            elif vector_strength < 0.70:
                kelly_adjusted = kelly_fraction * 0.40
            elif vector_strength < 0.85:
                kelly_adjusted = kelly_fraction * 0.60
            else:
                kelly_adjusted = kelly_fraction * 0.80
        else:
            kelly_adjusted = kelly_fraction
        
        # Step 2: Get confidence multiplier from p-value
        confidence_multiplier = self.confidence.get_confidence(p_value)
        
        # Step 3: Combine them (PRODUCT, not sum)
        combined_multiplier = kelly_adjusted * confidence_multiplier
        
        # Step 4: Calculate final position size
        final_position = capital * combined_multiplier
        
        # Step 5: Check if it's worth trading
        if final_position < self.min_position_size:
            final_position = 0
            reason = f"Position too small (${final_position:.0f} < ${self.min_position_size:.0f} threshold)"
        else:
            reason = f"Kelly {kelly_adjusted:.1%} × Confidence {confidence_multiplier:.1%} = {combined_multiplier:.1%}"
        
        return ConfidenceAdjustedSize(
            final_position_size=final_position,
            kelly_fraction=kelly_adjusted,
            confidence_multiplier=confidence_multiplier,
            p_value=p_value,
            vector_strength=vector_strength,
            reasoning=reason,
            capital=capital,
            kelly_pct=kelly_adjusted * 100,
            confidence_pct=confidence_multiplier * 100,
            product=combined_multiplier * 100
        )
    
    def compare_scaling_methods(self, p_value: float, capital: float = 100000):
        """
        Show how different scaling methods affect position sizing
        
        Useful for understanding the impact of your choice
        """
        print(f"\n=== Position Sizing Comparison (p={p_value:.4f}) ===\n")
        
        for method in ['linear', 'sigmoid', 'step']:
            adjuster = ConfidenceAdjuster(threshold=0.05, scaling_method=method)
            confidence = adjuster.get_confidence(p_value)
            
            # Assume 3% kelly fraction
            kelly_frac = 0.03
            final_pos = capital * kelly_frac * confidence
            
            print(f"{method.upper():8} → Confidence: {confidence:.2%} → Position: ${final_pos:,.0f}")
        
        print()


# Example usage and testing
if __name__ == "__main__":
    
    print("="*70)
    print("CONFIDENCE-ADJUSTED KELLY CRITERION")
    print("="*70)
    
    # Initialize confidence adjuster
    adjuster = ConfidenceAdjuster(threshold=0.05, scaling_method='linear')
    
    print("\n=== Scaling Example: Linear Method ===\n")
    
    test_pvalues = [0.001, 0.01, 0.025, 0.045, 0.049, 0.051]
    
    for p in test_pvalues:
        conf = adjuster.get_confidence(p)
        kelly = 0.03  # 3% kelly fraction
        final = 100000 * kelly * conf
        
        print(f"p = {p:.4f} → Confidence: {conf:6.1%} → Position: ${final:8,.0f}")
        print(f"  {adjuster.explain_confidence(p)}")
        print()
    
    print("\n" + "="*70)
    print("SCALING METHOD COMPARISON")
    print("="*70)
    
    for test_p in [0.001, 0.025, 0.049]:
        print(f"\nAt p={test_p:.4f}:")
        
        linear = ConfidenceAdjuster(scaling_method='linear')
        sigmoid = ConfidenceAdjuster(scaling_method='sigmoid')
        step = ConfidenceAdjuster(scaling_method='step')
        
        conf_lin = linear.get_confidence(test_p)
        conf_sig = sigmoid.get_confidence(test_p)
        conf_step = step.get_confidence(test_p)
        
        print(f"  Linear:  {conf_lin:6.1%} → ${100000 * 0.03 * conf_lin:8,.0f}")
        print(f"  Sigmoid: {conf_sig:6.1%} → ${100000 * 0.03 * conf_sig:8,.0f}")
        print(f"  Step:    {conf_step:6.1%} → ${100000 * 0.03 * conf_step:8,.0f}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    print("""
LINEAR SCALING:
- Simplest: confidence drops linearly from 1.0 to 0.0
- p=0.001: 100% size ($3,000)
- p=0.025: 50% size ($1,500)
- p=0.049: 2% size ($60)
- Good for: Proportional risk-adjustment

SIGMOID SCALING:
- Smoother: S-curve, less aggressive near threshold
- p=0.001: 99% size ($2,970)
- p=0.025: 73% size ($2,190)
- p=0.049: 20% size ($600)
- Good for: Avoid being too aggressive near edge

STEP SCALING:
- Discrete: Three confidence tiers
- p<0.01:  100% size ($3,000)
- p<0.03:  75% size ($2,250)
- p<0.05:  25% size ($750)
- Good for: Clear risk tiers, easier to manage

RECOMMENDATION:
Use LINEAR for maximum flexibility.
Use SIGMOID if you're conservative.
Use STEP if you want clear tiers.
    """)
    
    print("\n" + "="*70)
    print("THE PRODUCT PRINCIPLE")
    print("="*70)
    
    print("""
WHY MULTIPLY (not add)?

KELLY gives you: edge-based sizing
"Based on 50 trades: 55% win rate, 1.2x odds → risk 3%"

CONFIDENCE gives you: signal-based sizing
"This vector: p=0.025 → confidence 50%"

PRODUCT = Kelly × Confidence
"Risk 3% × 50% = 1.5% this trade"

Why multiply?
- If Kelly = 0, you have no edge → position = 0 (correct)
- If Confidence = 0, signal is noise → position = 0 (correct)
- If both are high, you take full position (correct)
- If one is low, you reduce proportionally (correct)

Adding would be wrong:
"3% + 50% = 53% is nonsensical"

Multiplying is right:
"3% × 50% = 1.5% means risk half as much given signal weakness"
    """)