import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

# Ensure local imports are resolved
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from src.alpaca_trader import AlpacaTrader

# Configure institutional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MonitorUtility")

class PortfolioMonitor:
    """
    Real-time attribution and exposure monitoring for the Quantum Fractal system.
    
    Provides high-level summaries of buying power, unrealized PnL, and 
    execution state, maintaining audit-ready trade history logs.
    """

    def __init__(self) -> None:
        self.trader = AlpacaTrader()
        self.log_path = 'trade_log.json'

    def run_attribution_scan(self) -> None:
        """
        Performs a full scan of account status, active exposure, and order flow.
        """
        if not self.trader.connect():
            logger.error("ConnectionFailed | Alpaca API authentication error.")
            return

        self._display_header()
        
        # 1. Capital & Liquidity State
        account = self.trader.get_account_info()
        self._log_account_metrics(account)

        # 2. Market Exposure (Positions)
        positions = self.trader.get_positions()
        self._log_active_exposure(positions)

        # 3. Pending Execution (Orders)
        orders = self.trader.get_orders()
        self._log_order_flow(orders)

        # 4. Persistence Check (Trade Log)
        self._log_historical_persistence()

    def _display_header(self) -> None:
        print("\n" + "="*80)
        print(f" QUANTUM FRACTAL SYSTEM | STATUS REPORT | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

    def _log_account_metrics(self, account: Dict[str, Any]) -> None:
        """Outputs core liquidity metrics."""
        equity = float(account.get('portfolio_value', 0))
        cash = float(account.get('cash', 0))
        bp = float(account.get('buying_power', 0))
        
        print(f"\n[LIQUIDITY STATE]")
        print(f"  Total Equity:    ${equity:,.2f}")
        print(f"  Available Cash:  ${cash:,.2f}")
        print(f"  Buying Power:    ${bp:,.2f}")

    def _log_active_exposure(self, positions: List[Any]) -> None:
        """Summarizes current market-risk positions."""
        print(f"\n[ACTIVE EXPOSURE | Count: {len(positions)}]")
        if not positions:
            print("  NO_MARKET_EXPOSURE")
            return

        for pos in positions:
            pnl_pct = float(pos.unrealized_plpc) * 100
            print(f"  {pos.symbol:<6} | Qty: {pos.qty:>5} | Entry: ${float(pos.avg_fill_price):>8.2f} "
                  f"| Last: ${float(pos.current_price):>8.2f} | PnL: {pnl_pct:>+6.2f}%")

    def _log_order_flow(self, orders: List[Any]) -> None:
        """Tracks pending limit and market orders."""
        print(f"\n[PENDING EXECUTION | Count: {len(orders)}]")
        if not orders:
            print("  ZERO_PENDING_ORDERS")
            return

        for order in orders:
            price_type = f"${float(order.limit_price):.2f}" if order.limit_price else "MKT"
            print(f"  ID: {order.id[:8]} | {order.symbol:<6} | {order.side.upper():<4} "
                  f"| Qty: {order.qty:>5} | Target: {price_type:<8} | State: {order.status}")

    def _log_historical_persistence(self) -> None:
        """Reads recent audit logs from JSON store."""
        if not os.path.exists(self.log_path):
            return

        try:
            with open(self.log_path, 'r') as f:
                history = json.load(f)
            
            print(f"\n[AUDIT PERSISTENCE | Recent Events]")
            for event in history[-5:]:
                print(f"  {event['timestamp']} | {event['ticker']:<6} | {event['side'].upper():<4} "
                      f"| {event['quantity']:>5} @ ${float(event['price']):.2f}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"LogReadError | Could not parse audit log: {e}")

if __name__ == "__main__":
    load_dotenv()
    monitor = PortfolioMonitor()
    monitor.run_attribution_scan()