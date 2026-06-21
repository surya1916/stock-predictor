import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load model
model = load_model("stock_model.h5")

st.title("AI Stock Trend Predictor")

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

        features = [
            'Close',
            'Volume',
            'SMA_20',
            'EMA_20',
            'RSI',
            'MACD',
            'BB_Upper',
            'BB_Lower'
        ]

        scaler = MinMaxScaler()

        scaled_data = scaler.fit_transform(
            df[features]
        )

        X = []

        for i in range(60, len(scaled_data)):
            X.append(
                scaled_data[i-60:i]
            )

        X = np.array(X)

        prediction = model.predict(
            X[-1].reshape(1, 60, 8),
            verbose=0
        )

        confidence = prediction[0][0]

        if confidence > 0.5:

            trend = "Bullish"

            confidence_score = confidence

        else:

            trend = "Bearish"

            confidence_score = 1 - confidence

        st.success(
            f"Stock Symbol : {stock_symbol}"
        )

        st.success(
            f"Predicted Trend : {trend}"
        )

        st.success(
            f"Confidence Score : {round(confidence_score*100,2)} %"
        )
        st.subheader("Explanation")

        latest = df.iloc[-1]
        
        # RSI explanation
        if latest['RSI'] > 60:
            st.write("✓ RSI =", round(latest['RSI'],2),
                     "→ Positive momentum (Bullish)")
        elif latest['RSI'] < 40:
            st.write("✓ RSI =", round(latest['RSI'],2),
                     "→ Weak momentum (Bearish)")
        else:
            st.write("✓ RSI =", round(latest['RSI'],2),
                     "→ Neutral")
        
        # MACD explanation
        if latest['MACD'] > 0:
            st.write("✓ MACD > 0 → Bullish signal")
        else:
            st.write("✓ MACD < 0 → Bearish signal")
        
        # SMA explanation
        if latest['Close'] > latest['SMA_20']:
            st.write("✓ Price above SMA20 → Uptrend")
        else:
            st.write("✓ Price below SMA20 → Downtrend")
        
        # EMA explanation
        if latest['Close'] > latest['EMA_20']:
            st.write("✓ Price above EMA20 → Strong momentum")
        else:
            st.write("✓ Price below EMA20 → Weak momentum")
        
        # Bollinger Bands explanation
        if latest['Close'] > latest['BB_Upper']:
            st.write("✓ Price above Upper Bollinger Band → Strong Bullish")
        elif latest['Close'] < latest['BB_Lower']:
            st.write("✓ Price below Lower Bollinger Band → Strong Bearish")
        else:
            st.write("✓ Price inside Bollinger Bands → Normal volatility")

        st.line_chart(
            df['Close']
        )

    except Exception as e:

        st.error(
            f"Error : {e}"
        )
