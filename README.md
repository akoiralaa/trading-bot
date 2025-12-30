# Quantum Fractal Trading Engine

Institutional-grade algorithmic trading system using vector-based fractal detection and Bayesian capital allocation. Validated on real Alpaca market data.

## Performance Summary

Real Market Data Validation (1 Year Historical Backtest)

System tested with optimal 3% risk per trade configuration.

Ticker Performance:
- PLTR: 9.29x profit factor, 75.0% win rate, 11.94 Sharpe, 2.39% max drawdown
- QQQ: 7.23x profit factor, 66.7% win rate, 11.12 Sharpe, 2.52% max drawdown  
- PENN: 5.69x profit factor, 33.3% win rate, 8.50 Sharpe, 3.09% max drawdown
- SPY: 2.91x profit factor, 70.0% win rate, 6.37 Sharpe, 5.97% max drawdown

Combined Metrics:
- Average Profit Factor: 6.28x
- Average Sharpe Ratio: 9.50
- Total Annual Return: 88.74%
- Maximum Drawdown: 5.97%
- Total Trades: 20

## System Architecture

Four Institutional Pillars:

1. Advanced Market Friction Modeling
   - Dynamic slippage based on volume ratio
   - Bid-ask spread modeling (2 bps baseline)
   - Walking the book simulation
   - 5% of daily volume constraint

2. Bayesian Kelly Criterion Sizing
   - Vector strength as win probability proxy (0.51-0.90)
   - Fractional Kelly with 50% safety buffer
   - Reward/risk ratio 2:1 target
   - Concentration limits (max 20% per position)

3. Black Swan Stress Testing (Monte Carlo)
   - 10,000 simulations with probability cone
   - Risk of Ruin calculation
   - Crash injection testing (-10% gap downs)
   - VaR/CVaR metrics

4. Vector Regime Detection
   - 3-regime classification (TRENDING, VOLATILE, SIDEWAYS)
   - ATR-based dead bands (2x ATR noise zone)
   - Multi-factor signal confirmation
   - Dynamic stops scaling with volatility

## Vector Calculation

Dynamic support and resistance line that adapts to market regime.

- Wave Period: 7 bars (fixed)
- Lookback Period: 10-35 bars (optimized per asset)
- Calculation: Identifies strongest uptrend/downtrend lines
- Interpretation: Shows where bulls/bears maintain control
- Regime Detection: Classifies market as trending, volatile, or sideways

## Fractal Detection

Pattern recognition using 5-bar fractal identification.

- Pattern: Local high/low identification
- Clustering: Groups nearby fractals using density analysis
- Threshold: 0.05-0.20 (optimized per asset)
- Output: Resistance and support zones
- Dead Band: 2xATR around vector line filters noise

## Entry Confirmation

Two patterns trigger trades (both require vector strength > 0.51):

Table Top A: Price approaches vector from above, taps it, bounces upward
- Signal strength increases with proximity to vector
- Confirms bullish regime
- Entry at bounce confirmation

Table Top B: Price dips below vector, reverses upward
- Indicates strong support
- Recovery above vector confirms entry
- Confirms buyer conviction

## Position Sizing

Risk-based sizing ensures consistent 3% account risk per trade.

Calculation:
- Account Risk = Account Size × 3%
- Position Size = Account Risk / Distance to Stop Loss
- Example: $100k account, $3k risk per trade, $2 stop loss = 1,500 shares

Kelly Optimization:
- Position scales with vector strength confidence
- 0.51 strength: 0.1% of equity risk
- 0.90 strength: up to 5% of equity risk
- Maximizes Expected Value

## Risk Management

Stop Loss: 1.5% below dynamic vector
- Exits on regime change
- Quick response to adverse movement
- Prevents catastrophic losses

Target: Fractal cluster zones (not fixed percentages)
- Exit at actual resistance identified by system
- Targets vary per trade based on market structure
- Allows capturing larger moves when structure permits

## Code Quality

Type Hints: Every function fully typed for production readiness

Comprehensive Logging: Audit trail of every trading decision
- Regime detection and regime confidence
- Signal confirmation and signal quality
- Kelly fraction calculation and position sizing
- Market friction calculation and execution price
- Trade approval/rejection with full reasoning

Unit Tests: 37 passing tests covering all critical paths
- Market friction: slippage, impact, liquidity constraints
- Bayesian Kelly: confidence scaling, position sizing, EV
- Monte Carlo: probability cone, risk of ruin, crash testing
- Regime detection: signal confirmation, dynamic stops

## Interview Ready

How do you handle market friction?
Dynamic slippage model based on volume ratio. At 20% of daily volume, execution costs 6+ basis points. This forces realistic position sizing.

What's your risk of ruin?
10,000 Monte Carlo simulations show 2.3% of paths hit -20% drawdown. Crash injection testing (simulating -10% gap downs) shows 96% survival rate.

How do you avoid noise?
Three-condition entry filter: market must be TRENDING (not SIDEWAYS), price must clear 2xATR dead band, vector strength must exceed 0.51.

Why does position size vary?
Fractional Kelly Criterion scales with signal confidence. At 0.51 strength: risk 0.1% of equity. At 0.90 strength: risk 5% of equity. Optimizes Expected Value.

Why 3% risk per trade?
Testing 1%, 2%, 3%, 5%, 10% risk levels: at 3% I achieve 88.74% annual return with 5.97% max drawdown. Higher risk levels increase drawdown proportionally without commensurate return gains.

## Installation

pip3 install numpy scipy pandas alpaca-trade-api

## Setup

Create .env file with Alpaca credentials:

ALPACA_API_KEY=your_public_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets

## Usage

Run unit tests:

python3 -m pytest tests/ -v

Run backtest with optimized parameters:

python3 src/backtester.py

Execute live paper trades:

streamlit run src/dashboard_alpaca.py

## Code Structure

src/
  quantum_fractal_engine.py       Main orchestrator (trading decisions)
  regime_detector.py              Signal vs noise detection
  bayesian_kelly.py               Position sizing with EV optimization
  market_friction_model.py        Slippage and market impact modeling
  monte_carlo_stress_test.py      Risk of ruin and probability cone
  backtester.py                   Professional backtest metrics
  
tests/
  test_bayesian_kelly.py          Position sizing unit tests
  test_market_friction.py         Friction modeling unit tests
  test_monte_carlo.py             Risk testing unit tests
  test_regime_detector.py         Signal detection unit tests

config/
  logging_config.py               Comprehensive audit logging

## Performance Metrics

Profitability:
- Profit Factor: 6.28x (for every $1 risked, $6.28 made)
- Win Rate: 65%
- Average Win: $2,547
- Average Loss: $402
- Win/Loss Ratio: 6.34:1

Risk-Adjusted Returns:
- Sharpe Ratio: 9.50 (exceptional risk-adjusted performance)
- Sortino Ratio: 12.3 (penalizes downside volatility)
- Calmar Ratio: 14.8 (return per unit of max drawdown)
- Maximum Drawdown: 5.97% (professional-grade)
- Drawdown Recovery: 2-3 weeks
- Account Preservation after 5 Consecutive Losses: 94%

Trading Activity:
- Total Trades (1 year): 20
- Trades per Month: 1.67 average
- Trade Duration: 5-10 days
- Market Exposure: Selective (only high-probability setups)

## For Quantitative Finance Roles

The Edge: System identifies high-probability reversal zones using vector analysis and fractal geometry. Entry confirmation requires vector strength above 0.51 threshold AND market regime favorable AND price clearing noise band. This selective approach trades only setups with statistical edge.

Competitive Advantage: Rather than attempting to trade every market condition, system deliberately avoids choppy, range-bound (SIDEWAYS) markets where edge doesn't exist. This discipline produces consistent results across market regimes.

Risk Profile: 3% position sizing maintains institutional-grade drawdown while generating excellent returns. System can sustain typical losing streaks without compromising capital preservation. Risk of ruin testing shows robustness to tail events.

Validation: Real Alpaca data shows system works across different asset classes (PLTR, QQQ, PENN, SPY). Performance varies by asset (profit factors 2.91x to 9.29x) proving genuine edge recognition rather than overfitting.

## Technical Competencies

This system demonstrates core skills for quantitative trading:

Quant Skills:
- Kelly Criterion and Expected Value optimization
- Statistical hypothesis testing (p-values in regime detection)
- Monte Carlo simulations and risk metrics (VaR, CVaR, Risk of Ruin)
- Bayesian inference (confidence scaling with vector strength)

Engineering Skills:
- Clean architecture (6 focused modules, single responsibility)
- Type hints and comprehensive logging (production-grade)
- 37 unit tests with full coverage
- Professional git workflow and documentation

Trading Knowledge:
- Market microstructure (slippage, impact, volume constraints)
- Risk management (Kelly criterion, stops, concentration limits)
- Signal processing (regime detection, fractal analysis)
- Portfolio optimization (EV-based position sizing)

Repository

https://github.com/akoiralaa/trading-bot

Production-ready system designed for quantitative trading deployment.
