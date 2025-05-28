import streamlit as st
import pandas as pd
import requests
from llama_cpp import Llama
import datetime
import matplotlib.pyplot as plt
from io import BytesIO

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

    fig, ax = plt.subplots()
    ax.plot(df["date"], df["price"])
    ax.set_title(f"{coin.title()} Price Trend - Last {days} Days")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    fig.autofmt_xdate()

    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

# # -------- Helper Functions --------
# def fetch_crypto_price(coin="bitcoin"):
#     url = f"https://api.coingecko.com/api/v3/simple/price"
#     params = {"ids": coin, "vs_currencies": "usd"}
#     response = requests.get(url, params=params)
#     return response.json().get(coin, {}).get("usd", "N/A")

# def fetch_crypto_news(coin="bitcoin"):
#     try:
#         url = "https://cryptopanic.com/api/v1/posts/"
#         params = {
#             "auth_token": "",  # Replace this
#             "currencies": coin,
#             "kind": "news"
#         }
#         response = requests.get(url, params=params)
#         news_items = response.json().get("results", [])
#         return [f"{item['published_at']}: {item['title']}" for item in news_items[:5]]
#     except:
#         return ["No news available."]

# # LLM summary
# def generate_summary(prompt):
#     formatted_prompt = f"Q: {prompt}\nA:"
#     response = llm(formatted_prompt, max_tokens=150)
#     return response["choices"][0]["text"].strip()

# # Dummy plotting function (replace with real data from API if available)


# def plot_price_trend(crypto_id):
#     dates = pd.date_range(end=datetime.datetime.now(), periods=7)
#     base_price = fetch_crypto_price(crypto_id)

#     # Make sure it's a float (or set a fallback)
#     try:
#         base_price = float(base_price)
#     except ValueError:
#         base_price = 100.0  # or whatever default you want

#     prices = [base_price * (1 + i*0.01) for i in range(7)]

#     df = pd.DataFrame({'Date': dates, 'Price': prices})
#     df.set_index('Date', inplace=True)

#     # Plot
#     fig, ax = plt.subplots()
#     df['Price'].plot(ax=ax, marker='o', title=f"{crypto_id.capitalize()} Price Trend (Past Week)")
#     ax.set_ylabel("USD")
#     ax.set_xlabel("Date")
#     ax.grid(True)

#     buf = BytesIO()
#     plt.savefig(buf, format="png")
#     buf.seek(0)
#     return buf



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




# if news_list:
#     for news in news_list:
#         title = news.get("title", "No title")
#         summary = news.get("summary", "No summary")
#         link = news.get("url", "#")
#         published_at = news.get("published_at", "Unknown time")

#         st.markdown(f"### [{title}]({link})")
#         st.markdown(f"*{summary}*")
#         st.markdown(f"`Published at: {published_at}`")
#         st.markdown("---")
# else:
#     st.write("No news found.")

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
st.markdown("---")