"""
Complete Real-Time Trading Dashboard with Alpaca API + Position Sizing

Three-panel layout:
1. Price action with vector zones (4-panel chart)
2. Position sizing calculator (sidebar)
3. Trade decision summary (main area)
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy import stats
import warnings
import os
from datetime import datetime, timedelta
from position_sizing_calculator import PositionSizer, render_risk_reward_calculator, display_position_sizing_summary

warnings.filterwarnings('ignore')

# ============================================================================
# ALPACA API (Same as before)
# ============================================================================

def get_alpaca_credentials():
    """Get Alpaca API credentials"""
    try:
        api_key = st.secrets.get("ALPACA_API_KEY", os.getenv("ALPACA_API_KEY"))
        secret_key = st.secrets.get("ALPACA_SECRET_KEY", os.getenv("ALPACA_SECRET_KEY"))
        base_url = st.secrets.get("ALPACA_BASE_URL", "https://paper-trading-api.alpaca.markets")
        
        if not api_key or not secret_key:
            st.error("❌ Alpaca credentials not configured")
            st.stop()
        
        return api_key, secret_key, base_url
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        st.stop()


def fetch_alpaca_data(symbol: str, timeframe: str = "5min", days: int = 5):
    """Fetch real market data from Alpaca API"""
    try:
        from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        api_key, secret_key, base_url = get_alpaca_credentials()
        
        is_crypto = "/" in symbol or symbol in ["BTC", "ETH", "DOGE"]
        
        tf_map = {
            "1min": TimeFrame.Minute,
            "5min": TimeFrame.FiveMin,
            "15min": TimeFrame.FifteenMin,
            "1h": TimeFrame.Hour,
            "1d": TimeFrame.Day
        }
        tf = tf_map.get(timeframe, TimeFrame.FiveMin)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        if is_crypto:
            client = CryptoHistoricalDataClient(api_key, secret_key)
            if "/" not in symbol:
                symbol = f"{symbol}/USD"
            request = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start_date, end=end_date)
            bars = client.get_crypto_bars(request)
        else:
            client = StockHistoricalDataClient(api_key, secret_key)
            request = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start_date, end=end_date)
            bars = client.get_stock_bars(request)
        
        df = bars.df
        df.columns = [col.lower() for col in df.columns]
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.reset_index()
            df = df[df['symbol'] == symbol].reset_index(drop=True)
        
        df = df[['close', 'high', 'low', 'volume']].copy()
        df = df.reset_index(drop=True)
        
        st.success(f"✓ Loaded {len(df)} candles for {symbol}")
        return df
    
    except ImportError:
        st.error("❌ pip install alpaca-trade-api")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()


def fetch_yfinance_data(symbol: str, interval: str = "5m", period: str = "5d"):
    """Fallback: yfinance"""
    try:
        import yfinance as yf
        df = yf.download(symbol, interval=interval, period=period, progress=False)
        df.columns = [col.lower() for col in df.columns]
        df = df.reset_index()
        st.success(f"✓ Loaded {len(df)} candles for {symbol}")
        return df
    except ImportError:
        st.error("❌ pip install yfinance")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()


# ============================================================================
# STATISTICS (Same as before)
# ============================================================================

def calculate_vector_stats(data, window=20):
    """Calculate regression slope, t-statistic, p-value"""
    slopes, t_stats, p_values, r2_values, confidence_mults = [], [], [], [], []
    
    for i in range(len(data)):
        if i < window:
            slopes.append(0)
            t_stats.append(0)
            p_values.append(1.0)
            r2_values.append(0)
            confidence_mults.append(0)
            continue
        
        y = data['close'].iloc[i-window:i].values
        x = np.arange(window)
        slope, intercept, r_val, p_val, std_err = stats.linregress(x, y)
        
        t_stat = slope / std_err if std_err > 0 else 0
        
        slopes.append(slope)
        t_stats.append(t_stat)
        p_values.append(p_val)
        r2_values.append(r_val ** 2)
        
        confidence = 0.0 if p_val > 0.05 else 1.0 - (p_val / 0.05)
        confidence_mults.append(confidence)
    
    return slopes, t_stats, p_values, r2_values, confidence_mults


def get_vector_zones(data, window=20, atr_period=14):
    """Calculate vector zones"""
    slopes, t_stats, p_values, r2s, confs = calculate_vector_stats(data, window)
    
    high_low = data['high'] - data['low']
    high_close = abs(data['high'] - data['close'].shift())
    low_close = abs(data['low'] - data['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    
    data['slope'] = slopes
    data['t_stat'] = t_stats
    data['p_value'] = p_values
    data['r2'] = r2s
    data['confidence'] = confs
    data['atr'] = atr.fillna(atr.mean())
    
    data['vector'] = data['slope'] * np.arange(len(data))
    if len(data) > window:
        data['vector'] = data['vector'] - data['vector'].iloc[window] + data['close'].iloc[window]
    
    data['zone_width'] = data['atr'] * (0.5 + data['confidence'] * 0.5)
    data['upper_band'] = data['vector'] + data['zone_width']
    data['lower_band'] = data['vector'] - data['zone_width']
    
    return data


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(layout="wide", page_title="Trader Ready Dashboard")

st.markdown("""
<style>
    .metric-box { background-color: rgba(0, 255, 200, 0.1); padding: 15px; border-radius: 10px; }
    .header-title { color: cyan; font-size: 2.5em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown("<div class='header-title'>🎯 Real-Time Trading Dashboard</div>", unsafe_allow_html=True)
st.markdown("### Alpaca API + Statistical Filtering + Position Sizing")

# ============================================================================
# DATA SOURCE (Top of Sidebar)
# ============================================================================

st.sidebar.markdown("### 🔌 Data Source")

data_source = st.sidebar.radio("Choose:", ["Real-Time (Alpaca)", "Historical (yfinance)"])

if data_source == "Real-Time (Alpaca)":
    symbol = st.sidebar.text_input("Asset Symbol", "AAPL")
    timeframe = st.sidebar.selectbox("Timeframe", ["1min", "5min", "15min", "1h", "1d"], index=1)
    days = st.sidebar.slider("Days", 1, 30, 5)
    
    try:
        df = fetch_alpaca_data(symbol, timeframe=timeframe, days=days)
    except:
        st.sidebar.warning("Alpaca failed, using yfinance...")
        df = fetch_yfinance_data(symbol)
else:
    symbol = st.sidebar.text_input("Ticker", "AAPL")
    interval = st.sidebar.selectbox("Interval", ["5m", "15m", "1h", "1d"], index=0)
    period = st.sidebar.selectbox("Period", ["5d", "1mo", "3mo", "1y"], index=0)
    df = fetch_yfinance_data(symbol, interval=interval, period=period)

# ============================================================================
# VECTOR PARAMETERS
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Vector Parameters")

lookback = st.sidebar.slider("Lookback", 10, 50, 20)
alpha = st.sidebar.slider("Alpha (α)", 0.01, 0.10, 0.05)

show_zones = st.sidebar.checkbox("Show Zones", value=True)
show_confidence = st.sidebar.checkbox("Show Confidence", value=True)

# ============================================================================
# POSITION SIZING CALCULATOR
# ============================================================================

result = render_risk_reward_calculator()

# ============================================================================
# CALCULATE STATISTICS
# ============================================================================

df = get_vector_zones(df, window=lookback)

current_price = df['close'].iloc[-1]
current_p_value = df['p_value'].iloc[-1]
current_confidence = df['confidence'].iloc[-1]

# ============================================================================
# MAIN VISUALIZATION
# ============================================================================

st.markdown(f"### 📈 {symbol} - Live Analysis")

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.4, 0.2, 0.2, 0.2],
    subplot_titles=(f"Price & Vector Zone", "P-Value", "Confidence", "Slope")
)

# Row 1: Price
fig.add_trace(
    go.Scatter(x=df.index, y=df['close'], name="Price",
               line=dict(color='white', width=2)),
    row=1, col=1
)

if show_zones:
    fig.add_trace(go.Scatter(x=df.index, y=df['upper_band'], name="Upper", 
                  line=dict(color='cyan', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['lower_band'], name="Lower",
                  line=dict(color='cyan', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['vector'], name="Vector",
                  line=dict(color='lime', width=2)), row=1, col=1)

sig_idx = df[df['p_value'] < alpha].index
if len(sig_idx) > 0:
    fig.add_trace(go.Scatter(x=sig_idx, y=df.loc[sig_idx, 'close'],
                  mode='markers', name="Sig", marker=dict(color='cyan', size=6)),
                  row=1, col=1)

# Row 2: P-Value
fig.add_trace(go.Scatter(x=df.index, y=df['p_value'], name="P-Value",
              line=dict(color='red', width=2), fill='tozeroy',
              fillcolor='rgba(255,0,0,0.1)'), row=2, col=1)
fig.add_hline(y=alpha, line_dash="dash", line_color="white", row=2, col=1)
fig.update_yaxes(type="log", row=2, col=1)

# Row 3: Confidence
if show_confidence:
    colors = ['green' if c > 0.5 else 'yellow' if c > 0 else 'red' for c in df['confidence']]
    fig.add_trace(go.Bar(x=df.index, y=df['confidence'], name="Confidence",
                  marker=dict(color=colors)), row=3, col=1)

# Row 4: Slope
fig.add_trace(go.Bar(x=df.index, y=df['slope'], name="Slope",
              marker=dict(color=['green' if s > 0 else 'red' for s in df['slope']])),
              row=4, col=1)

fig.update_layout(height=900, template="plotly_dark", hovermode="x unified")
fig.update_xaxes(title_text="Bar", row=4, col=1)

st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# LIVE METRICS
# ============================================================================

st.markdown("---")
st.markdown("### 📊 Live Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(f"Price ({symbol})", f"${current_price:.2f}")

with col2:
    st.metric("P-Value", f"{current_p_value:.4f}",
              "Sig ✓" if current_p_value < alpha else f"Noise ✗")

with col3:
    st.metric("Confidence", f"{current_confidence*100:.0f}%",
              "High" if current_confidence > 0.7 else "Med" if current_confidence > 0 else "Low")

with col4:
    st.metric("Slope", f"{df['slope'].iloc[-1]:.6f}",
              "↗" if df['slope'].iloc[-1] > 0 else "↘")

# ============================================================================
# POSITION SIZING SUMMARY
# ============================================================================

st.markdown("---")
st.markdown("### 💰 Trade Execution Summary")

display_position_sizing_summary(result)

# ============================================================================
# INTERPRETATION
# ============================================================================

st.markdown("---")
st.markdown("### 🎯 Trading Guidance")

if current_p_value < 0.01:
    signal_strength = "🟢 STRONG"
    action = "Trade full position size"
elif current_p_value < 0.05:
    signal_strength = "🟡 MODERATE"
    action = f"Trade {current_confidence*100:.0f}% of position"
else:
    signal_strength = "🔴 WEAK"
    action = "Skip or minimal position"

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    **Signal Strength**
    {signal_strength}
    
    P-Value: {current_p_value:.4f}
    """)

with col2:
    st.markdown(f"""
    **Position Sizing**
    {result.final_shares:,} shares
    ${result.final_position_size_dollars:,.0f}
    
    Risk: ${result.max_loss_at_sl:,.0f}
    """)

with col3:
    st.markdown(f"""
    **Trade Action**
    {action}
    
    R:R Ratio: 1:{result.risk_reward_ratio:.2f}
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
### 🚀 Complete Trading System

This dashboard integrates:
1. **Real-time market data** (Alpaca API)
2. **Statistical significance filtering** (p-values)
3. **Confidence scaling** (Kelly × p-value)
4. **Position sizing calculator** (exact share count)
5. **Risk/reward analysis** (entry to exit)

**Ready to trade:**
- Set stop loss and take profit
- See exact position size
- Know max loss before entering
- Execute with confidence

**Interview impact:**
"From signal to execution. Real data. Statistical rigor. Exact position sizes."
""")