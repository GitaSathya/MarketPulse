import streamlit as st
import pandas as pd
import requests
from llama_cpp import Llama
import datetime
import plotly.express as px
from streamlit.components.v1 import html
from crypto_core import fetch_crypto_news

# -------- Model Configuration --------

# ------- Symbol Mapping -------
COIN_SYMBOLS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "dogecoin": "DOGEUSDT",
    "solana": "SOLUSDT",
    "cardano": "ADAUSDT",
    "ripple": "XRPUSDT",
    "litecoin": "LTCUSDT"
}

# ------- LLM Setup (adjust path as needed) -------
model_path = "/Users/gita/Documents/Code/VS_Code/LLM/capybarahermes-2.5-mistral-7b.Q5_K_S.gguf"
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

# ------- Fetch Crypto Price (Binance) -------
def fetch_binance_price(symbol="BTCUSDT"):
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    return float(response.json().get("price", 0))

# ------- Fetch 24h Price Change (Binance) -------
def fetch_binance_price_change(symbol="BTCUSDT"):
    url = "https://api.binance.com/api/v3/ticker/24hr"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    data = response.json()
    return {
        "price": float(data.get("lastPrice", 0)),
        "change": float(data.get("priceChange", 0)),
        "percent_change": float(data.get("priceChangePercent", 0))
    }

# ------- Fetch Historical Price Trend (Binance) -------
def fetch_binance_price_trend(symbol="BTCUSDT", days=7):
    url = "https://api.binance.com/api/v3/klines"
    interval = "1d"
    limit = days if isinstance(days, int) else 1000  # Binance max is 1000
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    klines = response.json()
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df["price"] = df["close"].astype(float)
    return df[["date", "price"]]

def plot_price_trend_binance(symbol, days=7):
    df = fetch_binance_price_trend(symbol, days)
    fig = px.line(
        df,
        x="date",
        y="price",
        title=f"{symbol.replace('USDT','')} Price Trend - Last {days} Days",
        labels={"price": "Price (USD)", "date": "Date"},
        template="plotly_dark"
    )
    fig.update_traces(line=dict(color="cyan", width=2))
    fig.update_layout(
        title_font_size=20,
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    return fig

# -------- Streamlit UI --------
st.set_page_config(page_title="Crypto LLM Assistant", layout="wide")

st.title("💹 Market Pulse - Crypto LLM Chat Assistant")
st.write("Get real-time cryptocurrency updates and insights powered by LLM.")

# Select coin
coin = st.selectbox("Choose a cryptocurrency:", list(COIN_SYMBOLS.keys()))
symbol = COIN_SYMBOLS[coin]

# Show live price
price = fetch_binance_price(symbol)
st.metric(label=f"Current Price of {coin.capitalize()}", value=f"${price:,.6f}")

# --- Enhanced Price Change Widget ---
price_data = fetch_binance_price_change(symbol)
current_price = price_data["price"]
change = price_data["change"]
percent_change = price_data["percent_change"]
currency = "USD"
timestamp = datetime.datetime.utcnow().strftime("%b %d, %H:%M UTC")

color = "#ff4b4b" if change < 0 else "#4CAF50"
arrow = "↓" if change < 0 else "↑"

price_display = f"{current_price:,.6f}" if isinstance(current_price, (int, float)) else "N/A"
summary_html = f"""
<div style="background-color:#111; padding: 25px; border-radius: 12px; color: white; width: 100%; font-family: 'Arial', sans-serif;">
    <div style="font-size: 16px; color: #ccc;">Market Summary &gt; <span style="font-weight: bold; color: white;">{coin.capitalize()}</span></div>
    <div style="font-size: 36px; font-weight: bold; margin-top: 5px;">{price_display} <span style="font-size: 20px;">{currency}</span></div>
    <div style="color: {color}; font-size: 16px; margin-top: 5px;">
        {arrow} {abs(change):,.6f} ({percent_change:.2f}%) today
    </div>
    <div style="font-size: 12px; color: #aaa; margin-top: 10px;">{timestamp}</div>
</div>
"""
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    html(summary_html, height=160)

# --- News Section (keep using your existing fetch_crypto_news) ---
news_list = fetch_crypto_news(coin)
def display_news(news_list):
    st.markdown("📰 **Latest Crypto News**")
    if not news_list:
        st.write("No news articles available right now.")
        return
    for news in news_list:
        try:
            title = news.get("title", "No title")
            url = news.get("url", "#")
            source = news.get("source", {}).get("title", "Unknown source")
            published_at = news.get("published_at", "")
            if published_at:
                try:
                    published_at = datetime.datetime.fromisoformat(published_at.replace("Z", "+00:00")).strftime("%b %d, %Y %H:%M")
                except Exception:
                    pass
            st.markdown(f"""
                <div style='margin-bottom: 20px;'>
                    <a href="{url}" target="_blank" style='font-size: 18px; font-weight: bold; color: #1f77b4;'>{title}</a><br>
                    <span style='font-size: 14px; color: gray;'>Source: {source} | Published: {published_at}</span>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error displaying an article: {e}")
display_news(news_list)

# --- LLM Q&A Section ---
st.subheader("🤖 Ask a question about the market")
user_prompt = st.text_area("Your Question", "Why is the price fluctuating today?")
run = st.button("Generate Insight")
if run and user_prompt:
    news_context = "\n".join([item.get("title", "") + ": " + item.get("summary", "") for item in news_list])
    llm_input = f"""
User question: {user_prompt}

Market Price: ${price}
Recent News:
{news_context}

Answer the user with a concise financial insight.
"""
    response = llm(llm_input, max_tokens=1000)
    output = response["choices"][0]["text"]
    st.success("LLM Insight")
    st.text(output)

# --- Price Trend Section ---
time_range_label = "Choose time range for trend:"
time_range_options = {
    "7 Days": 7,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "Max": 1000
}
time_range_display = st.selectbox(time_range_label, list(time_range_options.keys()))
selected_days = time_range_options[time_range_display]

st.subheader(f"📊 {time_range_display} Price Trend")
fig = plot_price_trend_binance(symbol, days=selected_days)
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by FicMe")
st.markdown("---")