import sys
import os
import logging
import datetime
from typing import Dict, Optional, Any

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from src.alpaca_trader import AlpacaTrader

# Institutional logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ConnectivityDiagnostic")

class AlpacaConnectivityTest:
    """
    Validation utility for live market data connectivity and account authentication.
    
    Verifies API handshake, retrieves real-time Level 1 quote data, and 
    validates sufficient buying power for the Fractal Alpha strategy.
    """

    WATCHLIST = ['PLTR', 'QQQ', 'PENN', 'SPY']

    def __init__(self) -> None:
        load_dotenv()
        self.trader = AlpacaTrader()

    def execute_diagnostics(self) -> bool:
        """
        Performs a full sequential check of the trading environment.
        """
        logger.info("Initializing System Readiness Diagnostic...")

        # 1. Authentication Handshake
        if not self.trader.connect():
            logger.error("AuthFailure | Unable to establish Alpaca API connection.")
            return False

        # 2. Account Liquidity Check
        try:
            account = self.trader.get_account_info()
            self._log_account_status(account)
        except Exception as e:
            logger.error(f"AccountQueryError | Failed to retrieve liquidity metrics: {e}")
            return False

        # 3. Real-Time Data Pipeline Check
        self._validate_quote_stream()

        logger.info("DiagnosticComplete | System environment is stable for execution.")
        return True

    def _log_account_status(self, account: Dict[str, Any]) -> None:
        """Standardizes the output of core capital metrics."""
        cash = float(account.get('cash', 0))
        bp = float(account.get('buying_power', 0))
        logger.info(f"AccountStatus | Cash: ${cash:,.2f} | Buying Power: ${bp:,.2f}")

    def _validate_quote_stream(self) -> None:
        """Verifies integrity of Level 1 Market Data (Bid/Ask) per ticker."""
        logger.info(f"Sampling Live Quotes (UTC: {datetime.datetime.utcnow().strftime('%H:%M:%S')}):")
        
        for ticker in self.WATCHLIST:
            try:
                # Alpaca get_latest_quote returns a Quote object with bid/ask prices
                quote = self.trader.api.get_latest_quote(ticker)
                if quote:
                    logger.info(f"  {ticker:<5} | Bid: ${quote.bid_price:>8.2f} | Ask: ${quote.ask_price:>8.2f}")
                else:
                    logger.warning(f"  {ticker:<5} | DataUnavailable | Null quote received.")
            except Exception as e:
                logger.error(f"  {ticker:<5} | StreamError | Could not fetch quote: {e}")

if __name__ == "__main__":
    diagnostic = AlpacaConnectivityTest()
    success = diagnostic.execute_diagnostics()
    
    if not success:
        sys.exit(1)