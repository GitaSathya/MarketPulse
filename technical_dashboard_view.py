import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import ta  # Technical Analysis Library
import plotly.express as px
import yfinance as yf

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
COIN_SYMBOLS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "dogecoin": "DOGEUSDT",
    "solana": "SOLUSDT"
}

@st.cache_data(ttl=300)
def fetch_binance_ohlcv(symbol, days=90):
    r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol": symbol, "interval": "1d", "limit": days})
    df = pd.DataFrame(r.json(), columns=[
        "time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["date"] = pd.to_datetime(df["time"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

def apply_indicators(df):
    df = df.copy()
    df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["rsi_14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    return df

def plot_candlestick_with_indicators(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["date"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Price",
        increasing_line_color="green",
        decreasing_line_color="red"
    ))
    fig.add_trace(go.Scatter(x=df["date"], y=df["sma_20"], line=dict(color="orange", width=2), name="SMA (20)"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["bb_upper"], line=dict(color="lightblue", dash="dot"), name="Upper BB"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["bb_lower"], line=dict(color="lightblue", dash="dot"), name="Lower BB",
                             fill="tonexty", fillcolor="rgba(173,216,230,0.2)"))
    fig.update_layout(
        title=f"{symbol.replace('USDT','')} Price with Technical Indicators",
        xaxis_title="Date", yaxis_title="Price (USD)",
        template="plotly_dark", height=600,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_secondary_indicators(df):
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=df["date"], y=df["rsi_14"],
        line=dict(color="magenta"), name="RSI (14)"
    ))
    fig_rsi.add_hline(y=70, line_dash="dot", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dot", line_color="green")
    fig_rsi.update_layout(
        title="Momentum – RSI (14)",
        template="plotly_dark", height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )

    fig_obv = go.Figure()
    fig_obv.add_trace(go.Scatter(
        x=df["date"], y=df["obv"],
        line=dict(color="cyan"), name="OBV"
    ))
    fig_obv.update_layout(
        title="Volume – On-Balance Volume (OBV)",
        template="plotly_dark", height=300,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig_rsi, fig_obv

# ─────────────────────────────────────────────────────────────
#  🔹 THIS IS THE FUNCTION YOU IMPORT IN app.py
# ─────────────────────────────────────────────────────────────
def render_technical_dashboard():
    st.title("📈 Technical Analysis Dashboard")

    st.header("🔍 Stock-Based Technical Analysis")

    symbol_options = [
    "NASDAQ:COIN", "NYSE:CRCL", "NASDAQ:HOOD", "NASDAQ:BTBT", "NASDAQ:MARA", "NASDAQ:RIOT",
    "NASDAQ:MSTR", "EBR:BNB", "STO:XMR", "SWX:AVAX", "AMS:AAVE", "BIT:DOT", "NYSEARCA:ETH",
    "NYSE:ENJ", "NYSEARCA:BTC", "NASDAQ:BTCS", "NASDAQ:HIVE", "NASDAQ:CLSK", "OTCKTS:REGRF",
    "OTCMKTS:BBKCF", "OTCMKTS:NPPTF", "NYSE:BTCM", "OTCMKTS:BTCWF", "OTCMKTS:DMGGF",
    "NASDAQ:BITF", "NASDAQ:HUT", "OTCMKTS:LUXFF", "TSE:ETC", "OTCMKTS:CSTXF"
    ]

    ticker = st.selectbox("Select Stock/Crypto Asset:", options=symbol_options, index=symbol_options.index("NASDAQ:COIN"))

    start_date = st.date_input("Start date", pd.to_datetime("2022-01-01"))
    end_date = st.date_input("End date", pd.to_datetime("today"))

    if st.button("Generate Technical Charts"):
        # Fix for BTC input to match yfinance format
        if ticker in ["BTC", "ETH", "LTC", "DOGE", "SOL"]:
            ticker += "-USD"

        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            st.warning("No data found. Please check the symbol and date range.")
            return

        # Handle MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        st.subheader(f"📉 Closing Price for {ticker}")
        st.line_chart(data["Close"])

        # Compute and plot MAs
        data["MA20"] = data["Close"].rolling(window=20).mean()
        data["MA50"] = data["Close"].rolling(window=50).mean()
        st.subheader("📊 Moving Averages (MA20 & MA50)")
        fig = px.line(data, x=data.index, y=["Close", "MA20", "MA50"],
                      labels={"value": "Price", "variable": "Indicator"}, title="MA20 & MA50")
        st.plotly_chart(fig)

        # Compute RSI manually
        delta = data["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        data["RSI"] = 100 - (100 / (1 + rs))

        st.subheader("📈 Relative Strength Index (RSI)")
        fig_rsi = px.line(data, x=data.index, y="RSI", title="RSI (14)")
        st.plotly_chart(fig_rsi)

    st.markdown("---")
    st.header("💹 Cryptocurrency Technical Indicators")
    coin = st.selectbox("Select Cryptocurrency", list(COIN_SYMBOLS.keys()))
    symbol = COIN_SYMBOLS[coin]
    days = st.slider("Select History Range (in Days)", min_value=30, max_value=365, value=90)

    df = fetch_binance_ohlcv(symbol, days)
    df = apply_indicators(df)

    st.subheader(f"{coin.capitalize()} – Price Chart with SMA and Bollinger Bands")
    fig = plot_candlestick_with_indicators(df, symbol)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 RSI and OBV Indicators")
    fig_rsi, fig_obv = plot_secondary_indicators(df)
    st.plotly_chart(fig_rsi, use_container_width=True)
    st.plotly_chart(fig_obv, use_container_width=True)
    st.markdown("---")
