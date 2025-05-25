import streamlit as st
import pandas as pd
import requests
from llama_cpp import Llama
import datetime
import matplotlib.pyplot as plt
from io import BytesIO

# -------- Model Configuration --------

model_path = "/Users/gita/Documents/Code/VS_Code/LLM/capybarahermes-2.5-mistral-7b.Q5_K_S.gguf"
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=4,
    verbose=False
)

# -------- Helper Functions --------
def fetch_crypto_price(coin="bitcoin"):
    url = f"https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin, "vs_currencies": "usd"}
    response = requests.get(url, params=params)
    return response.json().get(coin, {}).get("usd", "N/A")

def fetch_crypto_news(coin="bitcoin"):
    try:
        url = "https://cryptopanic.com/api/v1/posts/"
        params = {
            "auth_token": "c6629503d252f8dd082c35de7bba474ec0cd0874",  # Replace this
            "currencies": coin,
            "kind": "news"
        }
        response = requests.get(url, params=params)
        news_items = response.json().get("results", [])
        return [f"{item['published_at']}: {item['title']}" for item in news_items[:5]]
    except:
        return ["No news available."]

# LLM summary
def generate_summary(prompt):
    formatted_prompt = f"Q: {prompt}\nA:"
    response = llm(formatted_prompt, max_tokens=150)
    return response["choices"][0]["text"].strip()

# Dummy plotting function (replace with real data from API if available)


def plot_price_trend(crypto_id):
    dates = pd.date_range(end=datetime.datetime.now(), periods=7)
    base_price = fetch_crypto_price(crypto_id)

    # Make sure it's a float (or set a fallback)
    try:
        base_price = float(base_price)
    except ValueError:
        base_price = 100.0  # or whatever default you want

    prices = [base_price * (1 + i*0.01) for i in range(7)]

    df = pd.DataFrame({'Date': dates, 'Price': prices})
    df.set_index('Date', inplace=True)

    # Plot
    fig, ax = plt.subplots()
    df['Price'].plot(ax=ax, marker='o', title=f"{crypto_id.capitalize()} Price Trend (Past Week)")
    ax.set_ylabel("USD")
    ax.set_xlabel("Date")
    ax.grid(True)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf



# -------- Streamlit UI --------
#st.set_page_config(page_title="Crypto LLM Assistant", layout="wide")

st.title("💹 Crypto LLM Chat Assistant")
st.write("Get real-time cryptocurrency updates and insights powered by LLM.")

# Select coin
coin = st.selectbox("Choose a cryptocurrency:", ["bitcoin", "ethereum", "dogecoin", "solana"])

# Show live price
price = fetch_crypto_price(coin)
st.metric(label=f"Current Price of {coin.capitalize()}", value=f"${price}")

# Show news
st.subheader("📰 Latest News")
news = fetch_crypto_news(coin)
for item in news:
    st.markdown(f"- {item}")

# User prompt
st.subheader("🤖 Ask a question about the market")
user_prompt = st.text_area("Your Question", "Why is the price fluctuating today?")
run = st.button("Generate Insight")

# Response
if run and user_prompt:
    # Compose context
    news_context = "\n".join(news)
    llm_input = f"""
User question: {user_prompt}

Market Price: ${price}
Recent News:
{news_context}

Answer the user with a concise financial insight.
"""
    response = llm(llm_input, max_tokens=300)
    output = response["choices"][0]["text"]
    st.success("LLM Insight")
    st.write(output)
# Plotting (dummy data)
# Show price trend plot
st.subheader("📊 7-Day Price Trend")
img_buf = plot_price_trend(coin)
st.image(img_buf)
# Footer
st.markdown("---")
st.markdown("Made with ❤️ by FicMe")