import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class MarketFrictionModel:
    """
    Advanced market friction modeling for institutional trading.
    Accounts for slippage, market impact, and liquidity constraints.
    """
    
    def __init__(self, market_impact_coeff: float = 0.1, bid_ask_spread_bps: float = 2.0) -> None:
        self.market_impact_coeff = market_impact_coeff
        self.bid_ask_spread_bps = bid_ask_spread_bps
        logger.info(f"MarketFrictionModel initialized: impact_coeff={market_impact_coeff}, spread_bps={bid_ask_spread_bps}")
    
    def calculate_dynamic_slippage(self, qty: int, avg_volume: float, price: float) -> Dict:
        """
        Calculate slippage based on order size relative to liquidity.
        Jane Street Logic: Walking the book degrades entry price.
        """
        volume_ratio: float = (qty / avg_volume) * 100
        
        if volume_ratio < 1:
            impact_bps: float = 0.5
        elif volume_ratio < 5:
            impact_bps = volume_ratio * self.market_impact_coeff
        elif volume_ratio < 10:
            impact_bps = 5 + (volume_ratio - 5) * 0.2
        else:
            impact_bps = 6 + (volume_ratio - 10) * 0.5
        
        slippage_result: Dict = {
            'volume_ratio': volume_ratio,
            'impact_bps': impact_bps,
            'slippage_dollars': (price * impact_bps) / 10000,
            'execution_price': price * (1 + (impact_bps / 10000))
        }
        
        logger.debug(f"Dynamic slippage: qty={qty}, avg_vol={avg_volume}, ratio={volume_ratio:.2f}%, impact={impact_bps:.2f}bps")
        
        return slippage_result
    
    def calculate_total_friction(self, qty: int, avg_volume: float, price: float, side: str = 'buy') -> Dict:
        """
        Total execution cost = bid-ask spread + market impact
        """
        dynamic: Dict = self.calculate_dynamic_slippage(qty, avg_volume, price)
        spread_cost: float = (price * self.bid_ask_spread_bps) / 10000
        total_cost_bps: float = dynamic['impact_bps'] + self.bid_ask_spread_bps
        
        if side == 'buy':
            exec_price: float = price * (1 + (total_cost_bps / 10000))
        else:
            exec_price = price * (1 - (total_cost_bps / 10000))
        
        friction_result: Dict = {
            'bid_ask_spread_bps': self.bid_ask_spread_bps,
            'market_impact_bps': dynamic['impact_bps'],
            'total_friction_bps': total_cost_bps,
            'total_friction_dollars': (price * total_cost_bps) / 10000,
            'execution_price': exec_price,
            'volume_ratio': dynamic['volume_ratio']
        }
        
        logger.info(f"Total friction calculated: {side} {qty} @ ${price:.2f} -> exec price ${exec_price:.2f} (friction: {total_cost_bps:.2f}bps)")
        
        return friction_result
    
    def get_max_position_size(self, avg_volume: float, account_equity: float, max_volume_pct: float = 0.05) -> int:
        """
        Institutional constraint: Never use > 5% of daily volume.
        """
        max_size: int = int(avg_volume * max_volume_pct)
        logger.debug(f"Max position size: {max_size} shares (avg_vol={avg_volume:.0f}, constraint={max_volume_pct*100}%)")
        return max_size
