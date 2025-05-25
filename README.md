# MarketPulse - Crypto Insights Streamlit App

## Overview
This project is a Streamlit web application that provides real-time cryptocurrency insights, price trends, and news, enhanced by a local Large Language Model (LLM). The app allows users to select a cryptocurrency, view its current price, see recent news, visualize price trends, and ask market-related questions to an AI assistant.

## Key Features

### Cryptocurrency Selection
- Users can choose from Bitcoin, Ethereum, Dogecoin, or Solana via a dropdown menu.

### Live Price Display
- The app fetches the current price of the selected cryptocurrency using the CoinGecko API.
- The price is displayed prominently with a metric widget.

### Latest News
- The app retrieves the latest news headlines related to the selected cryptocurrency using the CryptoPanic API.
- News items are listed for quick reading.

### 7-Day Price Trend Visualization
- The app generates a 7-day price trend plot for the selected cryptocurrency.
- Currently, this uses simulated data based on the current price (for demonstration), but can be extended to use real historical data.

### AI-Powered Market Insights
- Users can enter a question about the market.
- The app uses a local LLM (CapybaraHermes 2.5 Mistral 7B) to generate a concise, context-aware answer, incorporating the latest price and news.

### User Interface
- Built with Streamlit for an interactive, modern web experience.
- Includes clear sections for price, news, AI insights, and trend visualization.

## Technical Stack
- **Frontend/UI:** Streamlit
- **Data Fetching:** `requests` library for API calls (CoinGecko, CryptoPanic)
- **Data Processing:** `pandas` for data manipulation
- **Plotting:** `matplotlib` for price trend visualization
- **AI/LLM:** `llama.cpp` via the `llama_cpp` Python package, running a local GGUF model
- **Other:** `datetime` for date handling, `io.BytesIO` for image streaming

## How It Works
1. User selects a cryptocurrency.
2. App fetches and displays:
   - Current price (from CoinGecko)
   - Latest news (from CryptoPanic)
   - 7-day price trend (simulated)
3. User can ask a question about the market.
4. The app sends the question, current price, and news context to the local LLM.
5. The LLM generates and displays a concise financial insight.

## Customization & Extensibility
- **Historical Data:** The price trend plot currently uses simulated data. You can enhance it by fetching real historical prices from CoinGecko’s `/coins/{id}/market_chart` endpoint.
- **News Filtering:** News can be filtered or expanded to include more sources or types.
- **Model:** You can swap the LLM model for another GGUF-compatible model as needed.
- **UI:** Streamlit components can be extended for richer interactivity.

## Typical Use Cases
- Crypto enthusiasts seeking quick insights and news.
- Traders wanting to ask AI-driven questions about market trends.
- Developers exploring LLM integration with financial data.
