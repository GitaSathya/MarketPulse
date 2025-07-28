import streamlit as st
import pandas as pd
import requests, datetime
import plotly.express as px
from llama_cpp import Llama
from crypto_core import fetch_crypto_news            # stays
from moving_average_view import show_moving_average  # page 2
from technical_dashboard_view import render_technical_dashboard  # page 3

# ─────────────────────────────────────────────────────────────
#  1️⃣  CONFIG & HELPERS
# ─────────────────────────────────────────────────────────────
COIN_SYMBOLS = {
    "bitcoin":  "BTCUSDT",
    "ethereum": "ETHUSDT",
    "dogecoin": "DOGEUSDT",
    "solana":   "SOLUSDT",
    "cardano":  "ADAUSDT",
    "ripple":   "XRPUSDT",
    "litecoin": "LTCUSDT"
}

llm = Llama(
    model_path="/Users/gita/Documents/Code/VS_Code/LLM/capybarahermes-2.5-mistral-7b.Q5_K_S.gguf",
    n_ctx=2048, n_threads=4, verbose=False
)

@st.cache_data(ttl=300)
def fetch_binance_price(symbol):
    r = requests.get("https://api.binance.com/api/v3/ticker/price",
                     params={"symbol": symbol})
    return float(r.json().get("price", 0))

@st.cache_data(ttl=300)
def fetch_binance_price_change(symbol):
    r = requests.get("https://api.binance.com/api/v3/ticker/24hr",
                     params={"symbol": symbol}).json()
    return {
        "price":          float(r.get("lastPrice", 0)),
        "change":         float(r.get("priceChange", 0)),
        "percent_change": float(r.get("priceChangePercent", 0))
    }

@st.cache_data(ttl=300)
def fetch_binance_price_trend(symbol, days=7):
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": symbol, "interval": "1d", "limit": days}).json()
    df = pd.DataFrame(r, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"])
    df["date"]  = pd.to_datetime(df["open_time"], unit="ms")
    df["price"] = df["close"].astype(float)
    return df[["date", "price"]]

def plot_price_trend_binance(symbol, days=7):
    df = fetch_binance_price_trend(symbol, days)
    fig = px.line(df, x="date", y="price",
                  title=f"{symbol.replace('USDT','')} – last {days} d",
                  labels={"price":"Price (USD)"}, template="plotly_dark")
    fig.update_traces(line=dict(color="cyan", width=2))
    fig.update_layout(margin=dict(l=40,r=40,t=60,b=40), height=450)
    return fig

# ─────────────────────────────────────────────────────────────
#  2️⃣  DASHBOARD PAGE (was crypto_dashboard.py)
# ─────────────────────────────────────────────────────────────
def show_dashboard():
    st.title("💹 Market Pulse – Crypto LLM Assistant")

    coin   = st.selectbox("Choose a cryptocurrency", list(COIN_SYMBOLS))
    symbol = COIN_SYMBOLS[coin]

    # Current price & change ------------------------------------------------
    price_info = fetch_binance_price_change(symbol)
    price, change, pct = price_info.values()
    arrow = "↓" if change < 0 else "↑"
    color = "#ff4b4b" if change < 0 else "#4CAF50"
    ts    = datetime.datetime.utcnow().strftime("%b %d, %H:%M UTC")
    st.markdown(f"""
    <div>
      <span style="font-size:42px;font-weight:bold;">${price:,.6f}</span>
      <div style="color:{color};font-size:22px;">{arrow}&nbsp;{abs(change):,.6f} ({pct:.2f}%)</div>
      <div style="font-size:14px;color:#aaa;">{ts}</div>
    </div>
    """, unsafe_allow_html=True)

    # News ------------------------------------------------------------------
    news = fetch_crypto_news(coin)
    st.subheader("📰 Latest News")
    if news:
        for n in news:
            title = n.get("title","")
            url   = n.get("url","#")
            st.markdown(f"- [{title}]({url})")
    else:
        st.write("No news available.")

    # LLM Q&A ---------------------------------------------------------------
    st.subheader("🤖 Ask a question about the market")
    q = st.text_area("Your question", "Why is the price fluctuating today?")
    if st.button("Generate insight") and q:
        news_ctx = " | ".join(i["title"] for i in news)
        prompt   = f"""User: {q}\nPrice: {price}\nNews: {news_ctx}\nAnswer:"""
        ans      = llm(prompt, max_tokens=300)["choices"][0]["text"]
        st.success(ans.strip())

    # Price trend -----------------------------------------------------------
    rng_map = {"7 Days":7,"1 Month":30,"3 Months":90,"6 Months":180,"1 Year":365}
    label   = st.selectbox("Trend range", rng_map.keys())
    fig     = plot_price_trend_binance(symbol, rng_map[label])
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────
#  3️⃣  PAGE CONTROLLER
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Market Pulse", layout="wide")
st.sidebar.title("📊 Navigation")
page_choice = st.sidebar.selectbox(
    "Select Analysis View",
    ["Dashboard", "Moving Average", "Technical Indicators", "Forecasting"]
)

if page_choice == "Dashboard":
    show_dashboard()
elif page_choice == "Moving Average":
    show_moving_average()
elif page_choice == "Technical Indicators":
    render_technical_dashboard()
elif page_choice == "Forecasting":
    st.title("📈 Forecasting Module")
    st.info("This section is under development. Coming soon!")
