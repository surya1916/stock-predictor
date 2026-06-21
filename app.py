import streamlit as st
from tensorflow.keras.models import load_model

# Load model
model = load_model("stock_model.h5")

st.title("AI Stock Trend Predictor")

stock_symbol = st.text_input(
    "Enter Stock Symbol",
    "TSLA"
)

if st.button("Predict"):

    prediction = model.predict(
        [[[0]*8]*60],
        verbose=0
    )

    confidence = prediction[0][0]

    if confidence > 0.5:
        trend = "Bullish"
        confidence_score = confidence
    else:
        trend = "Bearish"
        confidence_score = 1-confidence

    st.success(f"Stock Symbol : {stock_symbol}")
    st.success(f"Predicted Trend : {trend}")
    st.success(
        f"Confidence Score : {round(confidence_score*100,2)}%"
    )
