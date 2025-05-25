import requests
import time
import pandas as pd
from datetime import datetime, timedelta
from llama_cpp import Llama
import os
from dotenv import load_dotenv
import streamlit as st
import matplotlib.pyplot as plt
import io
# Load environment variables from .env file

env_path = '/Users/gita/Documents/Code/VS_Code/LLM/.env'  # ✅ Update this path
load_dotenv(dotenv_path=env_path)

# Load the LLM model (adjust the path as needed)
llm = Llama(
    model_path="/Users/gita/Documents/Code/VS_Code/LLM/capybarahermes-2.5-mistral-7b.Q5_K_S.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

# Fetch real-time crypto price

@st.cache_data(ttl=3600)  # 1-hour cache
def fetch_crypto_price(crypto_id):
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {'ids': crypto_id, 'vs_currencies': 'usd'}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise ValueError(f"❌ Failed to fetch data. HTTP Status: {response.status_code}")
    data = response.json()
    if crypto_id not in data:
        raise KeyError(f"❌ Crypto ID '{crypto_id}' not found in API response.")
    return data[crypto_id]['usd']


# Fetch news (replace with your CryptoPanic API Key if needed)
def fetch_crypto_news(crypto_id):

    api_key = os.getenv('CRYPTO_PANIC_API_KEY')
    url = f'https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&currencies={crypto_id}'
    response = requests.get(url)
    articles = response.json().get("results", [])
    return [f"{a['title']} - {a['url']}" for a in articles[:5]]


# LLM summary
def generate_summary(prompt):
    formatted_prompt = f"Q: {prompt}\nA:"
    response = llm(formatted_prompt, max_tokens=150)
    return response["choices"][0]["text"].strip()

# 📈 Simulated price trend using just ONE API call
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

