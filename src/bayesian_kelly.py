import numpy as np


class BayesianKellyCriterion:
    """
    Fractional Kelly Criterion with Bayesian confidence scaling.
    Uses vector strength as win probability proxy.
    """
    
    def __init__(self, account_equity, fractional_kelly=0.5, 
                 reward_risk_ratio=2.0, min_vector_strength=0.51):
        self.account_equity = account_equity
        self.fractional_kelly = fractional_kelly
        self.reward_risk_ratio = reward_risk_ratio
        self.min_vector_strength = min_vector_strength
    
    def calculate_kelly_fraction(self, vector_strength, win_rate=None, loss_rate=None):
        """
        Kelly Criterion: f* = (p * b - q) / b
        where:
        - p = probability of win (vector strength as proxy)
        - q = probability of loss (1 - p)
        - b = reward/risk ratio
        """
        if vector_strength < self.min_vector_strength:
            return 0.0
        
        p = vector_strength
        q = 1.0 - p
        b = self.reward_risk_ratio
        
        raw_kelly = (p * b - q) / b
        
        if raw_kelly <= 0:
            return 0.0
        
        safe_kelly = raw_kelly * self.fractional_kelly
        
        return max(0, min(safe_kelly, 0.25))
    
    def calculate_position_size(self, vector_strength, risk_per_share, 
                                buying_power, max_concentration=0.20):
        """
        Position size scales with vector strength (Bayesian confidence).
        
        High confidence (0.9) = higher % Kelly
        Low confidence (0.51) = minimal Kelly
        """
        kelly_fraction = self.calculate_kelly_fraction(vector_strength)
        
        if kelly_fraction == 0:
            return 0
        
        kelly_dollars = self.account_equity * kelly_fraction
        shares_by_kelly = kelly_dollars / risk_per_share
        
        shares_by_buying_power = buying_power / (risk_per_share + 0.01)
        shares_by_concentration = (self.account_equity * max_concentration) / (risk_per_share + 0.01)
        
        final_shares = int(min(shares_by_kelly, shares_by_buying_power, shares_by_concentration))
        
        return max(0, final_shares)
    
    def expected_value(self, vector_strength, entry_price, stop_price, target_price):
        """
        Calculate expected value of trade based on vector strength.
        EV = P(win) * Win - P(loss) * Loss
        """
        if vector_strength < self.min_vector_strength:
            return 0
        
        p_win = vector_strength
        p_loss = 1 - vector_strength
        
        win_amount = target_price - entry_price
        loss_amount = entry_price - stop_price
        
        ev = (p_win * win_amount) - (p_loss * loss_amount)
        
        return {
            'expected_value': ev,
            'probability_win': p_win,
            'probability_loss': p_loss,
            'win_amount': win_amount,
            'loss_amount': loss_amount,
            'favorable': ev > 0
        }
