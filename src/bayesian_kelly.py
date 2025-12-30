import logging
from typing import Dict

logger = logging.getLogger(__name__)

class BayesianKellyCriterion:
    """
    Implements a Bayesian-adjusted Fractional Kelly Criterion for capital allocation.
    
    Dynamically scales position size based on signal confidence (vector strength)
    and expected utility, applying a volatility buffer (fractional Kelly) to 
    mitigate tail-risk and estimation error.
    """
    
    def __init__(
        self, 
        account_equity: float, 
        fractional_kelly: float = 0.5, 
        reward_risk_ratio: float = 2.0,
        min_vector_strength: float = 0.51
    ) -> None:
        self.account_equity = account_equity
        self.fractional_kelly = fractional_kelly
        self.reward_risk_ratio = reward_risk_ratio
        self.min_vector_strength = min_vector_strength
        
        logger.info(f"Initialized Kelly Engine | Equity: {account_equity} | Multiplier: {fractional_kelly}")

    def calculate_kelly_fraction(self, vector_strength: float) -> float:
        """
        Derives the optimal growth fraction f* using the Kelly formula:
        f* = (p*b - q) / b
        """
        if vector_strength < self.min_vector_strength:
            return 0.0
        
        p = vector_strength # Win probability proxy
        q = 1.0 - p
        b = self.reward_risk_ratio
        
        raw_kelly = (p * b - q) / b
        
        if raw_kelly <= 0:
            return 0.0
        
        # Apply fractional safety buffer and hard concentration cap
        safe_kelly = raw_kelly * self.fractional_kelly
        return max(0.0, min(safe_kelly, 0.25))

    def calculate_position_size(
        self, 
        vector_strength: float, 
        risk_per_share: float,
        buying_power: float,
        max_concentration: float = 0.20
    ) -> int:
        """
        Returns share quantity constrained by Kelly allocation, available 
        liquidity, and portfolio concentration limits.
        """
        f_star = self.calculate_kelly_fraction(vector_strength)
        
        if f_star == 0:
            return 0
        
        # Risk-based capital allocation
        kelly_cap = self.account_equity * f_star
        
        # Constraint mapping
        shares_kelly = kelly_cap / (risk_per_share + 1e-6)
        shares_limit = (self.account_equity * max_concentration) / (risk_per_share + 1e-6)
        shares_liquid = buying_power / (risk_per_share + 1e-6)
        
        qty = int(min(shares_kelly, shares_limit, shares_liquid))
        
        logger.debug(f"Sizing Logic | Strength: {vector_strength:.2f} | f*: {f_star:.4f} | Qty: {qty}")
        return max(0, qty)

    def get_expected_value(
        self, 
        vector_strength: float, 
        entry: float,
        stop: float, 
        target: float
    ) -> Dict:
        """Calculates probabilistic trade expectancy (EV)."""
        p_win = vector_strength
        p_loss = 1.0 - p_win
        
        win_amt = target - entry
        loss_amt = entry - stop
        
        ev = (p_win * win_amt) - (p_loss * loss_amt)
        
        return {
            'ev': ev,
            'is_favorable': ev > 0 and vector_strength >= self.min_vector_strength
        }