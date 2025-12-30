import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BayesianKellyCriterion:
    """
    Fractional Kelly Criterion with Bayesian confidence scaling.
    Uses vector strength as win probability proxy.
    """
    
    def __init__(self, account_equity: float, fractional_kelly: float = 0.5, 
                 reward_risk_ratio: float = 2.0, min_vector_strength: float = 0.51) -> None:
        self.account_equity = account_equity
        self.fractional_kelly = fractional_kelly
        self.reward_risk_ratio = reward_risk_ratio
        self.min_vector_strength = min_vector_strength
        logger.info(f"BayesianKellyCriterion initialized: equity={account_equity}, kelly_frac={fractional_kelly}, r:r={reward_risk_ratio}")
    
    def calculate_kelly_fraction(self, vector_strength: float, win_rate: Optional[float] = None, 
                                loss_rate: Optional[float] = None) -> float:
        """
        Kelly Criterion: f* = (p * b - q) / b
        """
        if vector_strength < self.min_vector_strength:
            logger.debug(f"Vector strength {vector_strength:.3f} below threshold {self.min_vector_strength}, kelly=0")
            return 0.0
        
        p: float = vector_strength
        q: float = 1.0 - p
        b: float = self.reward_risk_ratio
        
        raw_kelly: float = (p * b - q) / b
        
        if raw_kelly <= 0:
            logger.debug(f"Raw kelly {raw_kelly:.4f} <= 0, returning 0")
            return 0.0
        
        safe_kelly: float = raw_kelly * self.fractional_kelly
        final_kelly: float = max(0, min(safe_kelly, 0.25))
        
        logger.debug(f"Kelly calculation: p={p:.3f}, q={q:.3f}, raw={raw_kelly:.4f}, safe={safe_kelly:.4f}, final={final_kelly:.4f}")
        
        return final_kelly
    
    def calculate_position_size(self, vector_strength: float, risk_per_share: float, 
                                buying_power: float, max_concentration: float = 0.20) -> int:
        """
        Position size scales with vector strength (Bayesian confidence).
        """
        kelly_fraction: float = self.calculate_kelly_fraction(vector_strength)
        
        if kelly_fraction == 0:
            logger.info(f"Kelly fraction is 0, position size = 0")
            return 0
        
        kelly_dollars: float = self.account_equity * kelly_fraction
        shares_by_kelly: float = kelly_dollars / risk_per_share
        
        shares_by_buying_power: float = buying_power / (risk_per_share + 0.01)
        shares_by_concentration: float = (self.account_equity * max_concentration) / (risk_per_share + 0.01)
        
        final_shares: int = int(min(shares_by_kelly, shares_by_buying_power, shares_by_concentration))
        
        logger.info(f"Position sizing: vector_strength={vector_strength:.3f}, kelly_fraction={kelly_fraction:.4f}, final_shares={final_shares}")
        logger.debug(f"  by_kelly={shares_by_kelly:.0f}, by_bp={shares_by_buying_power:.0f}, by_conc={shares_by_concentration:.0f}")
        
        return max(0, final_shares)
    
    def expected_value(self, vector_strength: float, entry_price: float, 
                      stop_price: float, target_price: float) -> Dict:
        """
        Calculate expected value of trade based on vector strength.
        EV = P(win) * Win - P(loss) * Loss
        """
        if vector_strength < self.min_vector_strength:
            logger.debug(f"Vector strength {vector_strength:.3f} < threshold, EV rejected")
            return {'expected_value': 0, 'favorable': False}
        
        p_win: float = vector_strength
        p_loss: float = 1 - vector_strength
        
        win_amount: float = target_price - entry_price
        loss_amount: float = entry_price - stop_price
        
        ev: float = (p_win * win_amount) - (p_loss * loss_amount)
        
        result: Dict = {
            'expected_value': ev,
            'probability_win': p_win,
            'probability_loss': p_loss,
            'win_amount': win_amount,
            'loss_amount': loss_amount,
            'favorable': ev > 0
        }
        
        logger.info(f"EV calculation: ev={ev:.4f}, p_win={p_win:.3f}, favorable={ev > 0}")
        
        return result
