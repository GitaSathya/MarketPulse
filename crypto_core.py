import requests
import pandas as pd
from datetime import datetime
from llama_cpp import Llama
import os
from dotenv import load_dotenv
import streamlit as st
import matplotlib.pyplot as plt
import io

# Load environment variables from .env file
env_path = '/Users/gita/Documents/Code/VS_Code/LLM/.env'
load_dotenv(dotenv_path=env_path)

# Load the LLM model (adjust the path as needed)
llm = Llama(
    model_path="/Users/gita/Documents/Code/VS_Code/LLM/capybarahermes-2.5-mistral-7b.Q5_K_S.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

# Symbol mapping for Binance
SYMBOL_MAP = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "dogecoin": "DOGEUSDT",
    "solana": "SOLUSDT",
    "cardano": "ADAUSDT",
    "ripple": "XRPUSDT",
    "litecoin": "LTCUSDT"
}

@st.cache_data(ttl=300)
def fetch_crypto_price(crypto_id):
    symbol = SYMBOL_MAP.get(crypto_id.lower(), "BTCUSDT")
    url = "https://api.binance.com/api/v3/ticker/price"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return "N/A"
    try:
        return float(response.json().get("price", 0))
    except Exception:
        return "N/A"

@st.cache_data(ttl=300)
def fetch_crypto_price_change(crypto_id):
    symbol = SYMBOL_MAP.get(crypto_id.lower(), "BTCUSDT")
    url = "https://api.binance.com/api/v3/ticker/24hr"
    params = {"symbol": symbol}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return {"price": "N/A", "change": 0.0, "percent_change": 0.0}
    data = response.json()
    try:
        return {
            "price": float(data.get("lastPrice", 0)),
            "change": float(data.get("priceChange", 0)),
            "percent_change": float(data.get("priceChangePercent", 0))
        }
    except Exception:
        return {"price": "N/A", "change": 0.0, "percent_change": 0.0}

@st.cache_data(ttl=300)
def fetch_crypto_news(crypto_id):
    # Use CryptoCompare for news, mapped by symbol
    symbol_map = {
        "bitcoin": "BTC",
        "ethereum": "ETH",
        "dogecoin": "DOGE",
        "solana": "SOL",
        "cardano": "ADA",
        "ripple": "XRP",
        "litecoin": "LTC"
    }
    symbol = symbol_map.get(crypto_id.lower(), "BTC")
    url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={symbol}&lang=EN"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    articles = response.json().get("Data", [])
    return [
        {
            "title": a.get("title", ""),
            "url": a.get("url", "#"),
            "source": {"title": a.get("source", "CryptoCompare")},
            "published_at": datetime.utcfromtimestamp(a.get("published_on", 0)).strftime("%b %d, %Y %H:%M") if a.get("published_on") else ""
        }
        for a in articles[:5]
    ]

def generate_summary(prompt):
    formatted_prompt = f"Q: {prompt}\nA:"
    response = llm(formatted_prompt, max_tokens=150)
    return response["choices"][0]["text"].strip()

@st.cache_data(ttl=300)
def fetch_binance_price_trend(crypto_id, days=7):
    symbol = SYMBOL_MAP.get(crypto_id.lower(), "BTCUSDT")
    url = "https://api.binance.com/api/v3/klines"
    interval = "1d"
    limit = days if isinstance(days, int) else 1000
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return pd.DataFrame(columns=["date", "price"])
    klines = response.json()
    if not isinstance(klines, list) or len(klines) == 0:
        return pd.DataFrame(columns=["date", "price"])
    df = pd.DataFrame(klines, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms")
    df["price"] = df["close"].astype(float)
    return df[["date", "price"]]

def plot_price_trend(price_df, crypto_id):
    plt.figure(figsize=(10, 4))
    plt.plot(price_df['date'], price_df['price'], label=crypto_id.capitalize())
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"{crypto_id.capitalize()} Price Trend")
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def plot_moving_average(price_df, crypto_id, window=5):
    df = price_df.copy()
    df["SMA"] = df["price"].rolling(window=window).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['price'], label=f'{crypto_id.capitalize()} Price', alpha=0.6)
    plt.plot(df['date'], df['SMA'], label=f'{window}-Day SMA', linestyle='--', color='orange')
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"{crypto_id.capitalize()} {window}-Day Moving Average")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

