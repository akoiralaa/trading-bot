import numpy as np
import pandas as pd
from market_friction_model import MarketFrictionModel
from bayesian_kelly import BayesianKellyCriterion
from monte_carlo_stress_test import MonteCarloStressTest
from regime_detector import RegimeDetector


class QuantumFractalEngine:
    """
    Institutional-grade trading engine for Jane Street.
    
    Combines:
    1. Advanced market friction modeling
    2. Bayesian Kelly position sizing
    3. Monte Carlo stress testing
    4. Regime-aware signal detection
    """
    
    def __init__(self, api, account_equity, fractional_kelly=0.5):
        self.api = api
        self.account_equity = account_equity
        
        self.friction_model = MarketFrictionModel(
            market_impact_coeff=0.1,
            bid_ask_spread_bps=2.0
        )
        
        self.kelly = BayesianKellyCriterion(
            account_equity=account_equity,
            fractional_kelly=fractional_kelly,
            reward_risk_ratio=2.0
        )
        
        self.monte_carlo = MonteCarloStressTest(
            initial_equity=account_equity,
            simulations=10000
        )
        
        self.regime_detector = RegimeDetector(
            atr_multiplier=2.0,
            min_vector_strength=0.51
        )
    
    def execute_trading_cycle(self, symbol, prices, vector_prices, 
                             vector_strengths, atr_values, avg_volume):
        """
        Complete trading cycle with institutional guardrails.
        """
        current_price = prices[-1]
        current_vector = vector_prices[-1]
        current_strength = vector_strengths[-1]
        current_atr = atr_values[-1]
        
        regime = self.regime_detector.detect_regime(prices, lookback=30)
        
        breakout = self.regime_detector.is_breakout_signal(
            current_price, current_vector, current_atr,
            current_strength, regime['regime']
        )
        
        if not breakout['is_signal']:
            return {
                'trade': None,
                'reason': 'Signal rejected - insufficient confirmation',
                'regime': regime,
                'breakout': breakout
            }
        
        stop_info = self.regime_detector.calculate_dynamic_stop(
            current_price, current_vector, current_atr, side='long'
        )
        
        buying_power = float(self.api.get_account().buying_power)
        risk_per_share = stop_info['risk_distance']
        
        qty = self.kelly.calculate_position_size(
            current_strength, risk_per_share, buying_power
        )
        
        if qty == 0:
            return {
                'trade': None,
                'reason': 'Kelly sizing rejected - insufficient confidence',
                'vector_strength': current_strength,
                'regime': regime
            }
        
        friction = self.friction_model.calculate_total_friction(
            qty, avg_volume, current_price, side='buy'
        )
        
        max_size = self.friction_model.get_max_position_size(avg_volume)
        if qty > max_size:
            qty = max_size
        
        target_price = current_price * 1.02
        ev = self.kelly.expected_value(
            current_strength, current_price, stop_info['stop_price'], target_price
        )
        
        if not ev['favorable']:
            return {
                'trade': None,
                'reason': 'EV negative - trade rejected',
                'expected_value': ev,
                'regime': regime
            }
        
        trade = {
            'symbol': symbol,
            'qty': int(qty),
            'entry_price': current_price,
            'execution_price': friction['execution_price'],
            'stop_price': stop_info['stop_price'],
            'target_price': target_price,
            'risk_per_share': risk_per_share,
            'kelly_fraction': (qty * risk_per_share) / self.account_equity,
            'vector_strength': current_strength,
            'regime': regime['regime'],
            'expected_value': ev,
            'friction': friction,
            'timestamp': pd.Timestamp.now()
        }
        
        return {
            'trade': trade,
            'regime': regime,
            'breakout': breakout,
            'friction': friction,
            'kelly_info': {
                'kelly_fraction': (qty * risk_per_share) / self.account_equity,
                'position_size': qty,
                'buying_power_used': qty * current_price
            }
        }
    
    def stress_test_strategy(self, historical_returns):
        """
        Run complete Monte Carlo suite: probability cone + risk of ruin + crash tests.
        """
        results = {
            'probability_cone': self.monte_carlo.run_probability_cone(historical_returns),
            'risk_of_ruin': self.monte_carlo.calculate_risk_of_ruin(historical_returns),
            'crash_stress_test': self.monte_carlo.stress_test_crashes(historical_returns),
            'var_cvar': self.monte_carlo.calculate_var_cvar(historical_returns, confidence=0.95)
        }
        
        return results
    
    def get_institutional_report(self, historical_returns, current_positions=None):
        """
        Jane Street Interview Report: Everything they want to see.
        """
        stress_results = self.stress_test_strategy(historical_returns)
        
        report = {
            'risk_metrics': {
                'var_95': stress_results['var_cvar']['value_at_risk'],
                'cvar_95': stress_results['var_cvar']['conditional_var'],
                'risk_of_ruin_20pct': stress_results['risk_of_ruin']['risk_of_ruin_pct'],
                'worst_case_equity': stress_results['probability_cone']['worst_case'],
                'best_case_equity': stress_results['probability_cone']['best_case'],
                'median_equity': stress_results['probability_cone']['median']
            },
            'stress_tests': {
                'crash_survival_rate': stress_results['crash_stress_test']['survival_rate'],
                'worst_drawdown_in_crash': stress_results['crash_stress_test']['worst_drawdown'],
                'median_final_equity_with_crash': stress_results['crash_stress_test']['median_final_equity']
            },
            'probability_cone': {
                'p5': stress_results['probability_cone']['percentiles']['p5'][-1],
                'p25': stress_results['probability_cone']['percentiles']['p25'][-1],
                'p50': stress_results['probability_cone']['percentiles']['p50'][-1],
                'p75': stress_results['probability_cone']['percentiles']['p75'][-1],
                'p95': stress_results['probability_cone']['percentiles']['p95'][-1]
            },
            'kelly_usage': {
                'fractional_kelly': self.kelly.fractional_kelly,
                'reward_risk_ratio': self.kelly.reward_risk_ratio,
                'min_vector_strength': self.kelly.min_vector_strength
            }
        }
        
        return report
