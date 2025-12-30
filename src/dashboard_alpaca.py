import streamlit as st
from alpaca_trade_api import REST
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Trading Dashboard")

# Professional Header
st.markdown("### Live Alpaca Trading Dashboard")

# API Connection
api_key = st.secrets["ALPACA_API_KEY"]
secret_key = st.secrets["ALPACA_SECRET_KEY"]
base_url = st.secrets["ALPACA_BASE_URL"]
api = REST(api_key, secret_key, base_url)

# Account Metrics
account = api.get_account()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Account Status", account.status)
with col2:
    st.metric("Equity", f"${float(account.equity):,.0f}")
with col3:
    st.metric("Cash", f"${float(account.cash):,.0f}")

st.markdown("---")

# Position Table
st.markdown("### Open Positions")
positions = api.list_positions()
if positions:
    pos_data = []
    for p in positions:
        pos_data.append({
            "Symbol": p.symbol,
            "Qty": int(p.qty),
            "Avg Entry": f"${float(p.avg_entry_price):,.2f}",
            "Current Price": f"${float(p.current_price):,.2f}",
            "Total PnL": f"${float(p.unrealized_pl):,.2f}",
            "PnL %": f"{float(p.unrealized_plpc) * 100:.2f}%"
        })
    st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
else:
    st.info("No open positions. Ready for deployment.")

st.markdown("---")

# Asset Selection
assets = ["AAPL", "SPY", "TSLA", "GOOGL", "MSFT"]
selected = st.selectbox("Select Asset:", assets)

# Data Fetching - Logic fix to ensure graphs change per asset
bars = api.get_bars(selected, "5min", limit=100).df.reset_index()
bars.columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
current_price = float(bars['close'].iloc[-1])

st.write(f"Latest price ({selected}): ${current_price:.2f}")

# Technical Chart
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
fig.add_trace(go.Scatter(x=bars.index, y=bars['close'], name="Price", line=dict(color='white', width=2)), row=1, col=1)
fig.add_trace(go.Bar(x=bars.index, y=bars['volume'], name="Volume", marker=dict(color='cyan', opacity=0.3)), row=2, col=1)
fig.update_layout(height=500, template="plotly_dark", hovermode="x unified", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Risk Management UI
st.markdown("### Institutional Risk Controller")
risk_of_equity = st.sidebar.slider("Risk per Trade (%)", 0.1, 2.0, 1.0) / 100

c1, c2, c3 = st.columns(3)
entry = c1.number_input("Entry Price:", value=current_price, format="%.2f")
stop = c2.number_input("Stop Loss:", value=current_price - 2.0, format="%.2f")
target = c3.number_input("Take Profit:", value=current_price + 4.0, format="%.2f")

# Sizing Logic
risk_per_share = entry - stop
if risk_per_share > 0:
    risk_dollars = float(account.equity) * risk_of_equity
    final_shares = int(min(risk_dollars / risk_per_share, float(account.buying_power) / entry))
    
    avg_vol = bars['volume'].tail(10).mean()
    impact_pct = (final_shares / avg_vol) * 100
    
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Shares", final_shares)
    mc2.metric("Market Impact", f"{impact_pct:.2f}%")
    mc3.metric("R:R Ratio", f"{(target-entry)/risk_per_share:.2f}")

    if st.button("EXECUTE LIMIT ORDER", type="primary"):
        api.submit_order(symbol=selected, qty=final_shares, side='buy', type='limit', 
                         time_in_force='day', limit_price=entry * 1.0005)
        st.success(f"Order submitted for {selected}")
else:
    st.error("Stop loss must be below entry price")

# Order History
with st.expander("Order History"):
    orders = api.list_orders(status='all', limit=5)
    if orders:
        st.dataframe(pd.DataFrame([{"Symbol": o.symbol, "Qty": o.qty, "Status": o.status} for o in orders]))
