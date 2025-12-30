"""
Risk-to-Reward Position Sizing Calculator
Integrates with confidence multiplier and account balance

Calculates exact position size based on:
- Account balance
- Entry price
- Stop loss (risk per trade)
- Take profit (potential reward)
- Confidence multiplier (p-value based)
- Risk percentage (Kelly adjusted)

This is the FINAL piece: from signal to exact share count.
"""

import streamlit as st
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PositionSizingResult:
    """Container for position sizing calculations"""
    # Account
    account_balance: float
    risk_per_trade_percent: float
    risk_per_trade_dollars: float
    
    # Entry
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    
    # Risk/Reward
    risk_per_share: float
    reward_per_share: float
    risk_reward_ratio: float
    
    # Position Sizing
    max_shares_by_risk: float
    kelly_fraction: float
    confidence_multiplier: float
    adjusted_kelly_fraction: float
    final_shares: int
    final_position_size_dollars: float
    
    # Analysis
    max_loss_at_sl: float
    potential_profit_at_tp: float
    breakeven_price: float
    
    # Metadata
    signal: str
    p_value: float
    asset_symbol: str


class PositionSizer:
    """
    Professional-grade position sizing calculator
    
    Implements:
    - Kelly Criterion sizing
    - Risk-to-reward analysis
    - Confidence adjustment
    - Fractional Kelly for safety
    """
    
    def __init__(self, account_balance: float, kelly_fraction: float = 0.03):
        """
        Args:
            account_balance: Total account capital
            kelly_fraction: Base Kelly % (typically 2-5%)
        """
        self.account_balance = account_balance
        self.kelly_fraction = kelly_fraction
    
    def calculate_position_size(self,
                               entry_price: float,
                               stop_loss_price: float,
                               take_profit_price: float,
                               confidence_multiplier: float = 1.0,
                               risk_percent: float = None,
                               min_risk_reward: float = 1.0) -> PositionSizingResult:
        """
        Calculate exact position size based on risk/reward
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss level
            take_profit_price: Take profit target
            confidence_multiplier: From p-value (0.0 to 1.0)
            risk_percent: Max risk % of account (default: Kelly fraction)
            min_risk_reward: Minimum R:R ratio required (default: 1.0)
        
        Returns:
            PositionSizingResult with all calculations
        """
        
        # Determine signal direction
        if entry_price > stop_loss_price:
            signal = "LONG"
            risk_per_share = entry_price - stop_loss_price
            reward_per_share = take_profit_price - entry_price
        else:
            signal = "SHORT"
            risk_per_share = stop_loss_price - entry_price
            reward_per_share = entry_price - take_profit_price
        
        # Risk/Reward ratio
        if reward_per_share > 0:
            risk_reward_ratio = reward_per_share / risk_per_share
        else:
            risk_reward_ratio = 0
        
        # Position sizing by risk
        if risk_percent is None:
            risk_percent = self.kelly_fraction
        
        # Risk in dollars
        risk_dollars = self.account_balance * risk_percent
        
        # Max shares by risk
        if risk_per_share > 0:
            max_shares = risk_dollars / risk_per_share
        else:
            max_shares = 0
        
        # Adjust by confidence multiplier (scale down with lower confidence)
        adjusted_kelly = self.kelly_fraction * confidence_multiplier
        adjusted_risk_dollars = self.account_balance * adjusted_kelly
        
        if risk_per_share > 0:
            final_shares_decimal = adjusted_risk_dollars / risk_per_share
        else:
            final_shares_decimal = 0
        
        # Round to whole shares (or fractional for crypto)
        final_shares = int(np.floor(final_shares_decimal))
        
        # Actual position size in dollars
        final_position_dollars = final_shares * entry_price
        
        # Max loss and potential profit
        max_loss = final_shares * risk_per_share
        potential_profit = final_shares * reward_per_share
        
        # Breakeven price
        breakeven = entry_price
        
        return PositionSizingResult(
            account_balance=self.account_balance,
            risk_per_trade_percent=risk_percent,
            risk_per_trade_dollars=risk_dollars,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            risk_per_share=risk_per_share,
            reward_per_share=reward_per_share,
            risk_reward_ratio=risk_reward_ratio,
            max_shares_by_risk=max_shares,
            kelly_fraction=self.kelly_fraction,
            confidence_multiplier=confidence_multiplier,
            adjusted_kelly_fraction=adjusted_kelly,
            final_shares=final_shares,
            final_position_size_dollars=final_position_dollars,
            max_loss_at_sl=max_loss,
            potential_profit_at_tp=potential_profit,
            breakeven_price=breakeven,
            signal=signal,
            p_value=None,
            asset_symbol=None
        )


def render_risk_reward_calculator():
    """
    Renders the complete risk/reward position sizing calculator in Streamlit sidebar
    
    This is meant to be called in your Streamlit app to add the calculator to the sidebar.
    """
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 Position Sizing Calculator")
    
    # Account section
    st.sidebar.markdown("**Account**")
    account_balance = st.sidebar.number_input(
        "Account Balance ($)",
        min_value=100,
        value=100000,
        step=1000,
        help="Total trading capital"
    )
    
    kelly_fraction = st.sidebar.slider(
        "Kelly Fraction (%)",
        min_value=1,
        max_value=10,
        value=3,
        help="Risk per trade as % of account (typically 2-5%)"
    ) / 100
    
    # Entry/Exit section
    st.sidebar.markdown("**Trade Setup**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        entry_price = st.number_input(
            "Entry Price ($)",
            min_value=0.01,
            value=100.0,
            step=0.01,
            help="Price you plan to enter"
        )
    
    with col2:
        signal_type = st.radio("Signal", ["LONG", "SHORT"], horizontal=True)
    
    # Risk/Reward
    st.sidebar.markdown("**Risk/Reward Levels**")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if signal_type == "LONG":
            stop_loss = st.number_input(
                "Stop Loss ($)",
                min_value=0.01,
                value=entry_price * 0.95,
                step=0.01,
                help="Exit if price goes down to this level"
            )
        else:
            stop_loss = st.number_input(
                "Stop Loss ($)",
                min_value=entry_price * 1.01,
                value=entry_price * 1.05,
                step=0.01,
                help="Exit if price goes up to this level"
            )
    
    with col2:
        if signal_type == "LONG":
            take_profit = st.number_input(
                "Take Profit ($)",
                min_value=entry_price * 1.01,
                value=entry_price * 1.10,
                step=0.01,
                help="Exit if price goes up to this level"
            )
        else:
            take_profit = st.number_input(
                "Take Profit ($)",
                min_value=0.01,
                value=entry_price * 0.90,
                step=0.01,
                help="Exit if price goes down to this level"
            )
    
    # Confidence multiplier
    st.sidebar.markdown("**Confidence Adjustment**")
    
    current_p_value = st.sidebar.number_input(
        "Current P-Value",
        min_value=0.001,
        max_value=1.0,
        value=0.025,
        step=0.001,
        help="P-value from vector slope significance test"
    )
    
    # Calculate confidence multiplier
    if current_p_value > 0.05:
        confidence_multiplier = 0.0
        confidence_pct = 0
    else:
        confidence_multiplier = 1.0 - (current_p_value / 0.05)
        confidence_pct = confidence_multiplier * 100
    
    st.sidebar.metric("Confidence Multiplier", f"{confidence_pct:.0f}%")
    
    # Calculate position size
    sizer = PositionSizer(account_balance, kelly_fraction)
    result = sizer.calculate_position_size(
        entry_price=entry_price,
        stop_loss_price=stop_loss,
        take_profit_price=take_profit,
        confidence_multiplier=confidence_multiplier,
        risk_percent=kelly_fraction
    )
    result.signal = signal_type
    result.p_value = current_p_value
    
    # =========================================================================
    # DISPLAY RESULTS
    # =========================================================================
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Position Size Results")
    
    # Key metrics
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric(
            "Position Size",
            f"{result.final_shares:,} shares",
            f"${result.final_position_size_dollars:,.0f}"
        )
    
    with col2:
        st.metric(
            "Risk/Reward",
            f"1:{result.risk_reward_ratio:.2f}",
            "Good ✓" if result.risk_reward_ratio > 1.0 else "Bad ✗"
        )
    
    # Breakdown
    st.sidebar.markdown("**Risk Breakdown**")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric(
            "Max Loss",
            f"${result.max_loss_at_sl:,.2f}",
            f"{(result.max_loss_at_sl/account_balance)*100:.1f}% of account"
        )
    
    with col2:
        st.metric(
            "Profit Target",
            f"${result.potential_profit_at_tp:,.2f}",
            f"{(result.potential_profit_at_tp/account_balance)*100:.1f}% of account"
        )
    
    # Detailed breakdown
    with st.sidebar.expander("📈 Detailed Analysis"):
        st.markdown(f"""
        **Trade Setup**
        - Signal: {result.signal}
        - Entry: ${result.entry_price:.2f}
        - Stop Loss: ${result.stop_loss_price:.2f}
        - Take Profit: ${result.take_profit_price:.2f}
        
        **Risk Analysis**
        - Risk per share: ${result.risk_per_share:.4f}
        - Reward per share: ${result.reward_per_share:.4f}
        - Risk/Reward Ratio: 1:{result.risk_reward_ratio:.2f}
        
        **Position Sizing**
        - Account Balance: ${result.account_balance:,.0f}
        - Base Kelly: {result.kelly_fraction*100:.1f}%
        - P-Value: {result.p_value:.4f}
        - Confidence: {result.confidence_multiplier*100:.0f}%
        - Adjusted Kelly: {result.adjusted_kelly_fraction*100:.1f}%
        - Risk Dollars: ${result.risk_per_trade_dollars:,.2f}
        
        **Final Position**
        - Shares: {result.final_shares:,}
        - Position Value: ${result.final_position_size_dollars:,.2f}
        - Max Loss: ${result.max_loss_at_sl:,.2f}
        - Profit Target: ${result.potential_profit_at_tp:,.2f}
        
        **Risk Management**
        - Risk % of Account: {(result.max_loss_at_sl/account_balance)*100:.1f}%
        - Reward % of Account: {(result.potential_profit_at_tp/account_balance)*100:.1f}%
        - Breakeven Price: ${result.breakeven_price:.2f}
        """)
    
    # Trading checklist
    with st.sidebar.expander("✅ Pre-Trade Checklist"):
        st.markdown(f"""
        - [ ] P-Value < 0.05? ({result.p_value:.4f} < 0.05) {'✓' if result.p_value < 0.05 else '✗'}
        - [ ] Risk/Reward > 1:1? (1:{result.risk_reward_ratio:.2f}) {'✓' if result.risk_reward_ratio > 1.0 else '✗'}
        - [ ] Position size reasonable? ({result.final_shares:,} shares) {'✓' if result.final_shares > 0 else '✗'}
        - [ ] Risk < 2% of account? ({(result.max_loss_at_sl/account_balance)*100:.1f}%) {'✓' if (result.max_loss_at_sl/account_balance) < 0.02 else '✗'}
        - [ ] Confidence high enough? ({result.confidence_multiplier*100:.0f}%) {'✓' if result.confidence_multiplier > 0.5 else '✗'}
        """)
    
    return result


def display_position_sizing_summary(result: PositionSizingResult):
    """
    Display a full-page summary of position sizing
    Called in main area (not sidebar)
    """
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Signal",
            result.signal,
            f"P-Value: {result.p_value:.4f}"
        )
    
    with col2:
        st.metric(
            "Position Size",
            f"{result.final_shares:,} shares",
            f"${result.final_position_size_dollars:,.0f}"
        )
    
    with col3:
        st.metric(
            "Risk/Reward",
            f"1:{result.risk_reward_ratio:.2f}",
            "Acceptable ✓" if result.risk_reward_ratio > 1.0 else "Poor ✗"
        )
    
    # Visual representation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Trade Analysis")
        st.markdown(f"""
        **Entry/Exit Setup**
        - Entry Price: **${result.entry_price:.2f}**
        - Stop Loss: **${result.stop_loss_price:.2f}**
        - Take Profit: **${result.take_profit_price:.2f}**
        
        **Risk Metrics**
        - Risk per Share: **${result.risk_per_share:.4f}**
        - Reward per Share: **${result.reward_per_share:.4f}**
        - Risk/Reward: **1:{result.risk_reward_ratio:.2f}**
        """)
    
    with col2:
        st.markdown("### 💰 Position Sizing")
        st.markdown(f"""
        **Account**
        - Balance: **${result.account_balance:,.0f}**
        - Base Kelly: **{result.kelly_fraction*100:.1f}%**
        - Risk Dollars: **${result.risk_per_trade_dollars:,.2f}**
        
        **Adjustments**
        - P-Value: **{result.p_value:.4f}**
        - Confidence: **{result.confidence_multiplier*100:.0f}%**
        - Adjusted Kelly: **{result.adjusted_kelly_fraction*100:.1f}%**
        """)
    
    # Risk summary
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Max Loss (at SL)",
            f"${result.max_loss_at_sl:,.2f}",
            f"{(result.max_loss_at_sl/result.account_balance)*100:.2f}% of account"
        )
    
    with col2:
        st.metric(
            "Potential Profit (at TP)",
            f"${result.potential_profit_at_tp:,.2f}",
            f"{(result.potential_profit_at_tp/result.account_balance)*100:.2f}% of account"
        )
    
    with col3:
        st.metric(
            "Confidence Adjustment",
            f"{result.confidence_multiplier*100:.0f}%",
            f"Kelly scaled by p-value"
        )
    
    # Trade decision
    st.markdown("---")
    
    decision_box = ""
    
    if result.p_value > 0.05:
        decision_box = "**DO NOT TRADE** - P-value > 0.05 (not significant)"
    elif result.risk_reward_ratio < 1.0:
        decision_box = "**CAUTION** - Risk/Reward < 1.0 (unfavorable odds)"
    elif result.final_shares == 0:
        decision_box = "**CAUTION** - Position size rounds to 0 shares"
    elif (result.max_loss_at_sl / result.account_balance) > 0.05:
        decision_box = "**CAUTION** - Risk > 5% of account"
    else:
        decision_box = f"**READY TO TRADE** - {result.final_shares:,} shares @ ${result.entry_price:.2f}"
    
    st.info(decision_box)


# ============================================================================
# TEST/EXAMPLE
# ============================================================================

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Risk/Reward Position Sizing Calculator")
    
    # Use the calculator
    result = render_risk_reward_calculator()
    
    # Display results in main area
    st.markdown("---")
    display_position_sizing_summary(result)