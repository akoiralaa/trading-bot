import numpy as np


class MarketFrictionModel:
    """
    Advanced market friction modeling for institutional trading.
    Accounts for slippage, market impact, and liquidity constraints.
    """
    
    def __init__(self, market_impact_coeff=0.1, bid_ask_spread_bps=2.0):
        self.market_impact_coeff = market_impact_coeff
        self.bid_ask_spread_bps = bid_ask_spread_bps
    
    def calculate_dynamic_slippage(self, qty, avg_volume, price):
        """
        Calculate slippage based on order size relative to liquidity.
        Jane Street Logic: Walking the book degrades entry price.
        
        Slippage = (qty / avg_volume) * impact_coefficient
        """
        volume_ratio = (qty / avg_volume) * 100
        
        if volume_ratio < 1:
            impact_bps = 0.5
        elif volume_ratio < 5:
            impact_bps = volume_ratio * self.market_impact_coeff
        elif volume_ratio < 10:
            impact_bps = 5 + (volume_ratio - 5) * 0.2
        else:
            impact_bps = 6 + (volume_ratio - 10) * 0.5
        
        return {
            'volume_ratio': volume_ratio,
            'impact_bps': impact_bps,
            'slippage_dollars': (price * impact_bps) / 10000,
            'execution_price': price * (1 + (impact_bps / 10000))
        }
    
    def calculate_total_friction(self, qty, avg_volume, price, side='buy'):
        """
        Total execution cost = bid-ask spread + market impact
        """
        dynamic = self.calculate_dynamic_slippage(qty, avg_volume, price)
        spread_cost = (price * self.bid_ask_spread_bps) / 10000
        total_cost_bps = dynamic['impact_bps'] + self.bid_ask_spread_bps
        
        if side == 'buy':
            exec_price = price * (1 + (total_cost_bps / 10000))
        else:
            exec_price = price * (1 - (total_cost_bps / 10000))
        
        return {
            'bid_ask_spread_bps': self.bid_ask_spread_bps,
            'market_impact_bps': dynamic['impact_bps'],
            'total_friction_bps': total_cost_bps,
            'total_friction_dollars': (price * total_cost_bps) / 10000,
            'execution_price': exec_price,
            'volume_ratio': dynamic['volume_ratio']
        }
    
    def get_max_position_size(self, avg_volume, account_equity, max_volume_pct=0.05):
        """
        Institutional constraint: Never use > 5% of daily volume.
        This prevents excessive market impact.
        """
        return int(avg_volume * max_volume_pct)
