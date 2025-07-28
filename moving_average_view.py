import streamlit as st
from PIL import Image
from crypto_core import fetch_binance_price_trend, plot_moving_average
import matplotlib.pyplot as plt
import pandas as pd

def show_moving_average():
    st.title("📈 Moving Average Viewer")

    # Select crypto
    crypto = st.selectbox("Select Cryptocurrency", [
        "bitcoin", "ethereum", "dogecoin", "solana", "cardano", "ripple", "litecoin"
    ])

    # Select MA window dynamically
    window_size = st.slider("Select Moving Average Window (in days)", min_value=1, max_value=50, value=7)

    # Number of days to load (at least as large as the largest MA window)
    days = max(60, window_size + 10)
    price_df = fetch_binance_price_trend(crypto, days=days)

    if price_df.empty:
        st.warning("No data available. Please try again.")
        return

    # Calculate Moving Average
    price_df['moving_avg'] = price_df['price'].rolling(window=window_size).mean()

    # Plot
    st.subheader(f"{crypto.capitalize()} - {window_size}-Day Moving Average")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(price_df['date'], price_df['price'], label='Price', color='blue')
    ax.plot(price_df['date'], price_df['moving_avg'], label=f'{window_size}-Day MA', color='orange')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.set_title(f"{crypto.capitalize()} Price & {window_size}-Day Moving Average")
    ax.legend()
    st.pyplot(fig)