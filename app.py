import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.title("Explainable AI Stock Trend Predictor")

stock_symbol = st.text_input(
    "Enter Stock Symbol",
    "TSLA"
)

if st.button("Predict"):

    try:

        df = yf.download(
            stock_symbol,
            start="2018-01-01",
            end="2025-01-01",
            auto_adjust=False
        )

        df.columns = df.columns.get_level_values(0)

        # SMA
        df['SMA_20'] = df['Close'].rolling(20).mean()

        # EMA
        df['EMA_20'] = df['Close'].ewm(
            span=20,
            adjust=False
        ).mean()

        # RSI
        delta = df['Close'].diff()

        gain = delta.clip(lower=0)

        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()

        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss

        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['Close'].ewm(
            span=12,
            adjust=False
        ).mean()

        ema26 = df['Close'].ewm(
            span=26,
            adjust=False
        ).mean()

        df['MACD'] = ema12 - ema26

        # Bollinger Bands
        middle_band = df['Close'].rolling(20).mean()

        std = df['Close'].rolling(20).std()

        df['BB_Upper'] = middle_band + 2 * std

        df['BB_Lower'] = middle_band - 2 * std

        df.dropna(inplace=True)

        latest = df.iloc[-1]

        bullish_score = 0

        st.success(
            f"Stock Symbol : {stock_symbol}"
        )

        st.subheader("Explanation")

        # RSI
        if latest['RSI'] > 60:
            bullish_score += 1
            st.write(
                f"✓ RSI = {round(latest['RSI'],2)} → Positive momentum (Bullish)"
            )

        elif latest['RSI'] < 40:
            st.write(
                f"✓ RSI = {round(latest['RSI'],2)} → Weak momentum (Bearish)"
            )

        else:
            st.write(
                f"✓ RSI = {round(latest['RSI'],2)} → Neutral"
            )

        # MACD
        if latest['MACD'] > 0:
            bullish_score += 1
            st.write(
                "✓ MACD > 0 → Bullish signal"
            )

        else:
            st.write(
                "✓ MACD < 0 → Bearish signal"
            )

        # SMA
        if latest['Close'] > latest['SMA_20']:
            bullish_score += 1
            st.write(
                "✓ Price above SMA20 → Uptrend"
            )

        else:
            st.write(
                "✓ Price below SMA20 → Downtrend"
            )

        # EMA
        if latest['Close'] > latest['EMA_20']:
            bullish_score += 1
            st.write(
                "✓ Price above EMA20 → Strong momentum"
            )

        else:
            st.write(
                "✓ Price below EMA20 → Weak momentum"
            )

        # Bollinger Bands
        if latest['Close'] > latest['BB_Upper']:
            bullish_score += 1
            st.write(
                "✓ Price above Upper Bollinger Band → Strong Bullish"
            )

        elif latest['Close'] < latest['BB_Lower']:
            st.write(
                "✓ Price below Lower Bollinger Band → Strong Bearish"
            )

        else:
            st.write(
                "✓ Price inside Bollinger Bands → Normal volatility"
            )

        st.subheader("Technical Score")

        st.write(
            f"Bullish Score : {bullish_score}/5"
        )

        if bullish_score == 5:

            recommendation = "Strong Bullish"

        elif bullish_score >= 3:

            recommendation = "Bullish"

        elif bullish_score == 2:

            recommendation = "Neutral"

        else:

            recommendation = "Bearish"

        st.subheader("Final Recommendation")

        st.success(
            recommendation
        )

        st.line_chart(
            df['Close']
        )

    except Exception as e:

        st.error(
            f"Error : {e}"
        )
