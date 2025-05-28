import streamlit as st
import pandas as pd
import requests
from llama_cpp import Llama
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.express as px

# -------- Model Configuration --------


# ------- Symbol Mapping -------
COIN_TICKERS = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "dogecoin": "DOGE",
    "solana": "SOL",
    "cardano": "ADA",
    "ripple": "XRP",
    "litecoin": "LTC"
}

# ------- LLM Setup (adjust path as needed) -------
model_path = "/Users/gita/Documents/Code/VS_Code/LLM/capybarahermes-2.5-mistral-7b.Q5_K_S.gguf"
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

# ------- Fetch Crypto Price -------
def fetch_crypto_price(coin="bitcoin"):
    coin_id = coin.lower()
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin_id, "vs_currencies": "usd"}
    response = requests.get(url, params=params)
    return response.json().get(coin_id, {}).get("usd", "N/A")

# ------- Fetch Crypto News -------
def fetch_crypto_news(coin="bitcoin"):
    try:
        ticker = COIN_TICKERS.get(coin.lower(), "BTC")
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": "c6629503d252f8dd082c35de7bba474ec0cd0874",  # Use environment variable in real setup
            "currencies": ticker,
            "kind": "news"
        }
        response = requests.get(url, params=params)
        data = response.json()
        return data.get("results", [])
    except Exception as e:
        return [{"title": "Error fetching news", "summary": str(e)}]

# ------- Generate Summary Using LLM -------
def generate_summary(news_data):
    text_block = " ".join([item.get("title", "") + ": " + item.get("summary", "") for item in news_data])
    prompt = f"Summarize the following crypto news in a few sentences:\n{text_block}"
    output = llm(prompt=prompt, max_tokens=200)
    return output["choices"][0]["text"].strip()

# ------- Plot Price Trend -------

def plot_price_trend(coin="bitcoin", days=7):
    coin_id = coin.lower()
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    response = requests.get(url, params=params)
    prices = response.json().get("prices", [])

    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")

    fig = px.line(
        df,
        x="date",
        y="price",
        title=f"{coin.title()} Price Trend - Last {days} Days",
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

st.title("💹 Crypto LLM Chat Assistant")
st.write("Get real-time cryptocurrency updates and insights powered by LLM.")

# Select coin
coin = st.selectbox("Choose a cryptocurrency:", ["bitcoin", "ethereum", "dogecoin", "solana"])

# Show live price
price = fetch_crypto_price(coin)
st.metric(label=f"Current Price of {coin.capitalize()}", value=f"${price}")

# Show news

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
            # Format the datetime string
            if published_at:
                try:
                    published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00")).strftime("%b %d, %Y %H:%M")
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
# Fetch and display news
news_list = fetch_crypto_news(coin)
display_news(news_list)

# User prompt
st.subheader("🤖 Ask a question about the market")
user_prompt = st.text_area("Your Question", "Why is the price fluctuating today?")
run = st.button("Generate Insight")

# Response
if run and user_prompt:
    # Compose context
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
# Display summary of news

# Show price trend plot
st.subheader("📊 7-Day Price Trend")
fig = plot_price_trend(coin)
st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by FicMe")
st.markdown("---")