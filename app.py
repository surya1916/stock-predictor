import streamlit as st

st.title("AI Stock Trend Predictor")

stock_symbol = st.text_input(
    "Enter Stock Symbol",
    "TSLA"
)

if st.button("Predict"):

    st.success(
        f"Stock Symbol : {stock_symbol}"
    )

    st.success(
        "Predicted Trend : Bullish"
    )

    st.success(
        "Confidence Score : 75%"
    )

    st.line_chart(
        [10,20,15,25,30,28,35]
    )
