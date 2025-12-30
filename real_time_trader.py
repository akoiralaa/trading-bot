import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.alpaca_trader import AlpacaTrader
from src.vector_calculator import VectorCalculator
from src.fractal_detector import FractalDetector
from src.regime_detector import RegimeDetector
from src.market_friction_model import MarketFrictionModel
from src.bayesian_kelly import BayesianKellyCriterion

# Institutional logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FractalOrchestrator")

class QuantumFractalSystem:
    """
    Main execution coordinator for the Fractal Alpha strategy.
    
    Integrates signal generation, market regime filtering, and 
    Bayesian position sizing into a unified production loop.
    """
    
    # Calibrated parameters per asset class
    STRATEGY_MAP = {
        'PLTR': {'lookback': 10, 'threshold': 0.20},
        'QQQ':  {'lookback': 20, 'threshold': 0.15},
        'PENN': {'lookback': 35, 'threshold': 0.15},
        'SPY':  {'lookback': 10, 'threshold': 0.05},
    }

    def __init__(self) -> None:
        self.trader = AlpacaTrader()
        self.regime_guard = RegimeDetector()
        self.friction_engine = MarketFrictionModel()
        self.allocator = None # Initialized after connection
        
        if not self.trader.connect():
            raise ConnectionError("Alpaca API authentication failed.")
            
        account = self.trader.get_account_info()
        self.allocator = BayesianKellyCriterion(account_equity=float(account['portfolio_value']))

    def fetch_market_data(self, ticker: str, horizon: int = 365) -> pd.DataFrame:
        """Fetches and cleans historical OHLCV data."""
        try:
            end = datetime.now()
            start = end - timedelta(days=horizon)
            
            bars = self.trader.api.get_bars(
                ticker, '1Day',
                start=start.strftime('%Y-%m-%dT00:00:00Z'),
                end=end.strftime('%Y-%m-%dT00:00:00Z')
            ).df
            
            if bars.empty:
                return pd.DataFrame()
            
            # Standardization of dataframe schema
            df = bars[['open', 'high', 'low', 'close', 'volume']].copy()
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
        except Exception as e:
            logger.error(f"DataFetchError | Ticker: {ticker} | Reason: {e}")
            return pd.DataFrame()

    def run_cycle(self) -> None:
        """Executes one full iteration across the watch-list."""
        logger.info("Starting production cycle...")
        
        for ticker, params in self.STRATEGY_MAP.items():
            df = self.fetch_market_data(ticker)
            if df.empty or len(df) < params['lookback']:
                continue

            # 1. Primary Vector & Signal Generation
            vector_calc = VectorCalculator(wave_period=7, lookback=params['lookback'])
            vector = vector_calc.calculate_vector(df)
            strength = vector_calc.get_vector_strength(df, vector)
            
            # 2. Market Regime & Noise Filtering
            state_metrics = self.regime_guard.detect_regime(df['close'].values)
            
            # 3. Liquidity & Implementation Shortfall Modeling
            avg_volume = df['volume'].tail(10).mean()
            
            # Validation via Regime Guard (Volatility-adjusted Dead Band)
            signal = self.regime_guard.validate_execution_signal(
                price=df['close'].iloc[-1],
                vector=vector[-1],
                atr=df['close'].rolling(14).std().iloc[-1], # Simplified ATR proxy
                strength=strength[-1],
                state=state_metrics['state']
            )

            if signal['is_confirmed']:
                self._execute_trade_pipeline(ticker, df, strength[-1], avg_volume)

    def _execute_trade_pipeline(self, ticker: str, df: pd.DataFrame, strength: float, avg_vol: float) -> None:
        """Handles the final allocation and order placement logic."""
        current_price = df['close'].iloc[-1]
        
        # Risk-based stop calculation
        stop_price = current_price * 0.985
        risk_per_share = current_price - stop_price
        
        # Bayesian Kelly Sizing
        qty = self.allocator.calculate_position_size(
            vector_strength=strength,
            risk_per_share=risk_per_share,
            buying_power=float(self.trader.get_account_info()['buying_power'])
        )
        
        # Liquidity constraint check
        max_qty = self.friction_engine.get_liquidity_constrained_size(avg_vol)
        final_qty = min(qty, max_qty)

        if final_qty > 0:
            logger.info(f"EXECUTION_SIGNAL | {ticker} | Qty: {final_qty} | Price: {current_price:.2f}")
            # trader.place_order(...) would be called here

if __name__ == "__main__":
    system = QuantumFractalSystem()
    while True:
        try:
            system.run_cycle()
            # Standard institutional heartbeat interval
            time.sleep(3600) 
        except KeyboardInterrupt:
            break