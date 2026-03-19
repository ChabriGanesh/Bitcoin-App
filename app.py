import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# --- 1. CORE UTILITIES ---
def get_btc_data_cryptocompare(limit=100):
    """Fallback data fetcher if yfinance fails."""
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {"fsym": "BTC", "tsym": "USD", "limit": limit}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data['Response'] == 'Success':
            df = pd.DataFrame(data['Data']['Data'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            df = df[['close', 'volumeto']]
            df.columns = ['Close', 'Volume']
            return df
    except: return None
    return None

# --- 2. APP CONFIG & UI THEME ---
st.set_page_config(page_title="AlphaTerminal Pro", layout="wide")

st.markdown("""
    <style>
    .stMetric { background-color: #161a25; border: 1px solid #2a2e39; border-radius: 8px; padding: 15px; }
    [data-testid="stMetricValue"] { font-size: 26px; color: #d1d4dc; font-weight: bold; }
    .main { background-color: #0c0d10; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. AUTHENTICATION ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if not st.session_state.logged_in:
    st.title("🛡️ AlphaTerminal Secure Login")
    u, p = st.text_input("User"), st.text_input("Pass", type="password")
    if st.button("Access"):
        if u == "admin" and p == "mypass123":
            st.session_state.logged_in = True
            st.rerun()
        else: st.error("Invalid Credentials")
    st.stop()

# --- 4. SIDEBAR NAVIGATION ---
st.sidebar.title("💎 AlphaTerminal")
page = st.sidebar.radio("Navigation", ["📈 Dashboard", "🤖 AI Forecast"])

st.sidebar.divider()
st.sidebar.markdown("### 🧑‍💻 Developer Info")
st.sidebar.info("**PROJECT** BCA DS A • SRMIST")

# --- 5. PAGE 1: DASHBOARD ---
if page == "📈 Dashboard":
    st.title("Live Crypto Dashboard")
    
    # Fetch Data with Error Handling
    btc_df = yf.download("BTC-USD", period="1d", interval="1m", progress=False)
    
    if not btc_df.empty:
        curr_btc = btc_df['Close'].iloc[-1]
        open_btc = btc_df['Open'].iloc[0]
        change = ((curr_btc - open_btc) / open_btc) * 100
        
        m1, m2, m3 = st.columns(3)
        m1.metric("BITCOIN (BTC)", f"${curr_btc:,.2f}", f"{change:+.2f}%")
        m2.metric("ETHEREUM (ETH)", "$2,641.12", "+1.15%") 
        m3.metric("MARKET STATUS", "ACTIVE", "Live")

        st.divider()

        fig = go.Figure(data=[go.Candlestick(
            x=btc_df.index, open=btc_df['Open'], high=btc_df['High'], 
            low=btc_df['Low'], close=btc_df['Close']
        )])
        fig.update_layout(title="BTC-USD 1-Minute Chart", template="plotly_dark", height=500, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("📡 Unable to fetch live data. Please check your internet or try again later.")

# --- 6. PAGE 2: AI FORECAST ---
elif page == "🤖 AI Forecast":
    st.title("🤖 Deep Learning Bitcoin Forecast (LSTM)")

    @st.cache_resource 
    def load_btc_model():
        try: return load_model('btc_trading_model.keras')
        except: return None

    model = load_btc_model()

    if model is None:
        st.error("⚠️ LSTM Model file not found. Place 'btc_trading_model.keras' in the app directory.")
    else:
        if st.button("Generate AI Signal"):
            with st.spinner('AI is calculating market momentum...'):
                df_btc = get_btc_data_cryptocompare(limit=100)
                
                if df_btc is not None and not df_btc.empty:
                    # Preprocessing
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(df_btc['Close'].values.reshape(-1,1))
                    last_60_days = scaled_data[-60:].reshape(1, 60, 1)

                    # Prediction
                    prediction_scaled = model.predict(last_60_days)
                    prediction_usd = scaler.inverse_transform(prediction_scaled)[0][0]
                    current_price = df_btc['Close'].iloc[-1]
                    
                    st.divider()
                    col1, col2 = st.columns(2)
                    col1.metric("Current Price", f"${current_price:,.2f}")
                    
                    diff = prediction_usd - current_price
                    col2.metric("AI Prediction", f"${prediction_usd:,.2f}", f"{diff:+.2f}")
                    
                    if diff > (current_price * 0.015):
                        st.success("🚀 **STRONG BUY SIGNAL**")
                    elif diff < -(current_price * 0.015):
                        st.warning("📉 **SELL SIGNAL**")
                    else:
                        st.info("⚖️ **HOLD SIGNAL**")
                else:
                    st.error("Prediction failed: Could not retrieve historical training window.")