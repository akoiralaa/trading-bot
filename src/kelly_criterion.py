"""
Kelly Criterion Position Sizing for Dynamic Risk Management

Theory:
f* = (p*b - q) / b
where:
  p = probability of winning (win rate)
  q = probability of losing (1 - p)
  b = odds = (avg win size) / (avg loss size)
  f* = fraction of capital to risk

Example:
  p = 0.55 (55% win rate)
  q = 0.45
  b = 1.2 (average winner is 1.2x average loser)
  f* = (0.55 * 1.2 - 0.45) / 1.2 = 0.075 = 7.5% of capital

Why use it:
- Fixed 3% is safe but suboptimal
- Kelly adapts to actual strategy edge
- Signal strength (from vector_calculator) tells us when to size up/down
- Fractional Kelly (25-75% of Kelly) reduces ruin probability

For Jane Street:
- Shows you understand portfolio mathematics
- Risk scales with confidence (higher vector strength = bigger position)
- Backed by century-old theory from information theory
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class TradeMetrics:
    """Simple container for trade statistics"""
    entry_price: float
    exit_price: float
    size: int
    entry_bar: int
    exit_bar: int
    vector_strength: float = 0.5  # Confidence of signal


class KellyCriterion:
    """
    Implements fractional Kelly Criterion position sizing
    
    Key principle: Don't risk the Kelly fraction directly.
    Use fractional Kelly (25-75% of f*) to reduce drawdown risk.
    """
    
    def __init__(self, 
                 lookback_trades: int = 50,
                 kelly_fraction: float = 0.5,
                 min_risk: float = 0.01,
                 max_risk: float = 0.05):
        """
        Args:
            lookback_trades: Use last N trades to calculate win rate and odds
            kelly_fraction: Fraction of full Kelly to use (0.25-1.0 typical)
                           0.25 = very conservative
                           0.5 = reasonable
                           1.0 = aggressive (not recommended)
            min_risk: Floor on position sizing (1% of capital)
            max_risk: Ceiling on position sizing (5% of capital)
        """
        self.lookback_trades = lookback_trades
        self.kelly_fraction = kelly_fraction
        self.min_risk = min_risk
        self.max_risk = max_risk
        
        self.trade_history: List[TradeMetrics] = []
    
    def add_trade(self, trade: TradeMetrics):
        """Record a completed trade"""
        self.trade_history.append(trade)
    
    def calculate_trade_metrics(self) -> Dict[str, float]:
        """
        Analyze recent trades to get win probability and odds
        
        Returns:
            {
                'win_rate': 0.55,
                'avg_win': 150.0,
                'avg_loss': 100.0,
                'odds': 1.5,
                'sample_size': 40
            }
        """
        
        # Use only recent trades
        recent_trades = self.trade_history[-self.lookback_trades:]
        
        if len(recent_trades) < 5:
            # Not enough data - return neutral defaults
            return {
                'win_rate': 0.5,
                'avg_win': 1.0,
                'avg_loss': 1.0,
                'odds': 1.0,
                'sample_size': len(recent_trades),
                'insufficient_data': True
            }
        
        # Calculate returns (not profit in dollars, but R-multiples)
        returns = []
        for trade in recent_trades:
            ret = (trade.exit_price - trade.entry_price) / trade.entry_price
            returns.append(ret)
        
        returns = np.array(returns)
        wins = returns[returns > 0]
        losses = returns[returns <= 0]
        
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0.5
        
        avg_win = np.mean(wins) if len(wins) > 0 else 1.0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1.0
        
        # Odds = win size / loss size
        odds = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'odds': odds,
            'sample_size': len(recent_trades),
            'insufficient_data': False
        }
    
    def calculate_kelly_fraction(self, trade_metrics: Dict) -> float:
        """
        Calculate the Kelly fraction f*
        
        f* = (p*b - q) / b
        
        Returns fraction of capital to risk
        """
        
        if trade_metrics.get('insufficient_data'):
            return 0.03  # Default to 3%
        
        p = trade_metrics['win_rate']
        q = 1 - p
        b = trade_metrics['odds']
        
        if b <= 0:
            return 0.03  # Safety: avoid division errors
        
        # Full Kelly
        f_kelly = (p * b - q) / b
        
        # Handle edge cases
        if f_kelly < 0:
            # Negative Kelly means the strategy is losing
            return self.min_risk
        
        if f_kelly > 0.5:
            # Kelly can suggest very large positions; cap it
            f_kelly = 0.5
        
        # Apply fractional Kelly
        f_star = f_kelly * self.kelly_fraction
        
        # Clamp to min/max
        f_star = np.clip(f_star, self.min_risk, self.max_risk)
        
        return f_star
    
    def get_position_size_bayesian(self,
                                   capital: float,
                                   vector_strength: float,
                                   trade_metrics: Dict = None) -> float:
        """
        Calculate position size with Bayesian confidence adjustment
        
        Args:
            capital: Current account capital
            vector_strength: 0.0 to 1.0 (from vector_calculator.py)
                            This represents our confidence in the signal
                            0.51 = barely bullish, risk minimum
                            0.90 = very strong, risk more
            trade_metrics: Use provided metrics, or calculate from history
        
        Returns:
            position_size: Capital to risk in this trade
        
        Logic:
        1. Calculate Kelly from historical trades
        2. Adjust Kelly based on current signal confidence
        3. Size position accordingly
        """
        
        if trade_metrics is None:
            trade_metrics = self.calculate_trade_metrics()
        
        # Get base Kelly
        f_kelly = self.calculate_kelly_fraction(trade_metrics)
        
        # Adjust for signal confidence
        # Confidence = vector_strength (0.5 to 1.0)
        # If vector_strength < 0.6: Low confidence, use 25% of Kelly
        # If vector_strength < 0.8: Medium confidence, use 50% of Kelly
        # If vector_strength >= 0.8: High confidence, use 75% of Kelly
        
        if vector_strength < 0.55:
            confidence_multiplier = 0.25
        elif vector_strength < 0.70:
            confidence_multiplier = 0.4
        elif vector_strength < 0.85:
            confidence_multiplier = 0.6
        else:
            confidence_multiplier = 0.8
        
        # Apply confidence adjustment
        adjusted_risk = f_kelly * confidence_multiplier
        
        # Clamp to min/max
        adjusted_risk = np.clip(adjusted_risk, self.min_risk, self.max_risk)
        
        # Convert to dollars
        position_size = capital * adjusted_risk
        
        return position_size
    
    def get_position_size_fixed(self,
                               capital: float,
                               vector_strength: float = None) -> float:
        """
        Simplified: Just scale fixed 3% by vector strength
        
        For when you don't have enough trade history yet.
        This is a stepping stone to full Kelly.
        """
        
        base_risk = 0.03  # Start with 3%
        
        if vector_strength is None:
            return capital * base_risk
        
        # Scale between 1-2x based on strength
        if vector_strength < 0.55:
            multiplier = 0.5  # 1.5% when weak
        elif vector_strength < 0.75:
            multiplier = 1.0  # 3% when medium
        else:
            multiplier = 1.5  # 4.5% when strong
        
        adjusted_risk = base_risk * multiplier
        adjusted_risk = np.clip(adjusted_risk, self.min_risk, self.max_risk)
        
        return capital * adjusted_risk


class DynamicPositionSizer:
    """
    Wrapper that decides between fixed and Kelly sizing based on sample size
    
    This is what you actually use in your trading system.
    """
    
    def __init__(self, kelly_criterion: KellyCriterion):
        self.kelly = kelly_criterion
        self.warmup_trades = 30  # Need 30 trades before switching to Kelly
    
    def get_position_size(self,
                         capital: float,
                         vector_strength: float,
                         risk_mode: str = 'adaptive') -> Tuple[float, str]:
        """
        Get position size with reasoning
        
        Args:
            capital: Account capital
            vector_strength: Signal confidence (0.0-1.0)
            risk_mode: 'fixed', 'kelly', or 'adaptive'
        
        Returns:
            position_size: Amount to risk
            sizing_method: Description of how it was calculated
        """
        
        if risk_mode == 'fixed':
            size = self.kelly.get_position_size_fixed(capital, vector_strength)
            reason = f"Fixed sizing: ${size:.2f} (vector={vector_strength:.2f})"
        
        elif risk_mode == 'kelly':
            metrics = self.kelly.calculate_trade_metrics()
            size = self.kelly.get_position_size_bayesian(capital, vector_strength, metrics)
            reason = f"Kelly sizing: ${size:.2f} (win_rate={metrics['win_rate']:.1%}, odds={metrics['odds']:.2f})"
        
        elif risk_mode == 'adaptive':
            # Start with fixed, switch to Kelly when we have enough trades
            if len(self.kelly.trade_history) < self.warmup_trades:
                size = self.kelly.get_position_size_fixed(capital, vector_strength)
                reason = f"Warmup phase ({len(self.kelly.trade_history)}/{self.warmup_trades}): ${size:.2f}"
            else:
                metrics = self.kelly.calculate_trade_metrics()
                size = self.kelly.get_position_size_bayesian(capital, vector_strength, metrics)
                reason = f"Kelly sizing: ${size:.2f}"
        
        else:
            raise ValueError(f"Unknown risk_mode: {risk_mode}")
        
        return size, reason
    
    def add_completed_trade(self, trade: TradeMetrics):
        """Record a trade for Kelly calculations"""
        self.kelly.add_trade(trade)


# Example usage and testing
if __name__ == "__main__":
    
    # Scenario 1: Strategy with 55% win rate, 1.2x odds
    kelly = KellyCriterion(kelly_fraction=0.5)
    
    # Simulate 50 trades
    np.random.seed(42)
    for i in range(50):
        # 55% win rate
        is_win = np.random.rand() < 0.55
        
        if is_win:
            ret = np.random.uniform(0.01, 0.04)  # 1-4% wins
        else:
            ret = np.random.uniform(-0.03, 0.0)  # 0-3% losses
        
        entry = 100.0
        exit_price = entry * (1 + ret)
        
        trade = TradeMetrics(
            entry_price=entry,
            exit_price=exit_price,
            size=100,
            entry_bar=i*10,
            exit_bar=i*10+5,
            vector_strength=0.7 + np.random.uniform(-0.1, 0.1)
        )
        kelly.add_trade(trade)
    
    # Check metrics
    metrics = kelly.calculate_trade_metrics()
    print("\n=== Trade Metrics ===")
    print(f"Win rate: {metrics['win_rate']:.1%}")
    print(f"Avg win: {metrics['avg_win']:.2%}")
    print(f"Avg loss: {metrics['avg_loss']:.2%}")
    print(f"Odds: {metrics['odds']:.2f}")
    print(f"Sample size: {metrics['sample_size']}")
    
    # Kelly fraction
    kelly_frac = kelly.calculate_kelly_fraction(metrics)
    print(f"\nKelly fraction: {kelly_frac:.2%}")
    
    # Position sizing with different confidences
    capital = 100000
    print(f"\n=== Position Sizing (${capital:,.0f} capital) ===")
    
    sizer = DynamicPositionSizer(kelly)
    
    for strength in [0.5, 0.6, 0.75, 0.85, 0.95]:
        size, reason = sizer.get_position_size(capital, strength, risk_mode='kelly')
        print(f"Vector strength {strength:.2f}: ${size:.2f} ({reason})")