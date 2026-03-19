import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- APP CONFIG ---
st.set_page_config(page_title="AlphaTerminal Pro", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .stMetric { background-color: #161a25; border: 1px solid #2a2e39; border-radius: 8px; padding: 15px; }
    [data-testid="stMetricValue"] { font-size: 24px; color: #d1d4dc; }
    .main { background-color: #0c0d10; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINES ---
def clean_df(df):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def get_crypto_data(symbol="BTC", limit=100):
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {"fsym": symbol, "tsym": "USD", "limit": limit}
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if data['Response'] == 'Success':
            df = pd.DataFrame(data['Data']['Data'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df[['close', 'volumeto']].rename(columns={'close': 'Close', 'volumeto': 'Volume'})
    except Exception as e:
        st.error(f"Crypto Data Error: {e}")
    return None

# --- AUTHENTICATION ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if not st.session_state.logged_in:
    st.title("🛡️ AlphaTerminal Secure Login")
    u, p = st.text_input("User"), st.text_input("Pass", type="password")
    if st.button("Access"):
        if u == "admin" and p == "mypass123":
            st.session_state.logged_in = True
            st.rerun()
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("💎 AlphaTerminal")
page = st.sidebar.radio("Navigation", ["📈 Dashboard", "🤖 AI Forecast (LSTM)", "📊 Market News"])

# --- PAGE 1: DASHBOARD ---
if page == "📈 Dashboard":
    st.title("Live Market Dashboard")
    
    # Indices Summary
    idx_cols = st.columns(3)
    indices = {"NIFTY 50": "^NSEI", "SENSEX": "^BSESN", "BTC/USD": "BTC-USD"}
    for i, (name, sym) in enumerate(indices.items()):
        try:
            d = yf.download(sym, period="1d", interval="1m", progress=False)
            curr = d['Close'].iloc[-1]
            prev = d['Open'].iloc[0]
            pct = ((curr - prev) / prev) * 100
            idx_cols[i].metric(name, f"${curr:,.2f}" if "BTC" in name else f"₹{curr:,.2f}", f"{pct:+.2f}%")
        except: pass

    st.divider()
    
    ticker = st.selectbox("Symbol", ["RELIANCE.NS", "TCS.NS", "BTC-USD", "ETH-USD"])
    hist = yf.download(ticker, period="1mo", interval="1h", progress=False)
    hist = clean_df(hist)
    
    if hist is not None:
        fig = go.Figure(data=[go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'])])
        fig.update_layout(height=500, template="plotly_dark", title=f"{ticker} Hourly Chart")
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: AI FORECAST (LSTM) ---
elif page == "🤖 AI Forecast (LSTM)":
    st.title("🤖 Deep Learning Price Prediction")
    st.info("Using LSTM (Long Short-Term Memory) to analyze time-series patterns.")

    @st.cache_resource 
    def load_ai_model():
        try:
            return load_model('btc_trading_model.keras')
        except:
            return None

    model = load_ai_model()
    
    if model is None:
        st.error("⚠️ Model file 'btc_trading_model.keras' not found in directory.")
    else:
        target_crypto = st.selectbox("Select Asset", ["BTC", "ETH"])
        
        if st.button("Generate AI Signal"):
            df = get_crypto_data(target_crypto, limit=100)
            if df is not None:
                # LSTM Preprocessing (Looking back 60 days)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))
                input_data = scaled_data[-60:].reshape(1, 60, 1)

                with st.spinner('Neural Network is processing...'):
                    prediction_scaled = model.predict(input_data)
                    prediction_usd = scaler.inverse_transform(prediction_scaled)[0][0]
                    current_price = df['Close'].iloc[-1]
                    diff = prediction_usd - current_price

                    st.divider()
                    c1, c2 = st.columns(2)
                    c1.metric("Current Price", f"${current_price:,.2f}")
                    c2.metric("AI Forecast (24h)", f"${prediction_usd:,.2f}", f"{diff:+.2f}")

                    # Signal Logic
                    if diff > (current_price * 0.015):
                        st.success("🚀 **STRONG BUY SIGNAL**")
                    elif diff < -(current_price * 0.015):
                        st.warning("📉 **SELL SIGNAL**")
                    else:
                        st.info("⚖️ **HOLD / NEUTRAL**")

# --- PAGE 3: NEWS ---
elif page == "📊 Market News":
    st.title("📰 Market Intel")
    feed_url = "https://news.google.com/rss/search?q=crypto+market+india&hl=en-IN&gl=IN&ceid=IN:en"
    rss = feedparser.parse(feed_url)
    for entry in rss.entries[:10]:
        st.markdown(f"**{entry.title}**")
        st.caption(f"Source: {entry.source.title} | {entry.published}")
        st.write(f"[Read Article]({entry.link})")
        st.divider()