import streamlit as st
import pandas as pd
import pickle

# Page setup
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="wide")

# Background and custom CSS
page_style = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #f2fff0, #d4f1f4);
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
h1 {
    color: #2d6a4f;
    text-align: center;
    font-family: 'Poppins', sans-serif;
}
label, input, textarea {
    font-family: 'Poppins', sans-serif !important;
}
div.stButton > button {
    background-color: #40916c;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 1em;
    font-weight: bold;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #1b4332;
}
.result-box {
    background-color: #e9f5db;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    margin-top: 20px;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# Model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.markdown("<h1>🌾 Crop Recommendation System 🌾</h1>", unsafe_allow_html=True)
st.write("### Enter your land and environmental details below to get the best crop suggestion.")

# Input section
col1, col2 = st.columns(2)
with col1:
    N = st.number_input("Nitrogen (N)", 0, 200)
    P = st.number_input("Phosphorus (P)", 0, 200)
    K = st.number_input("Potassium (K)", 0, 200)
    ph = st.number_input("pH Value", 0.0, 14.0)
with col2:
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0)
    humidity = st.number_input("Humidity (%)", 0.0, 100.0)
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0)

# Predict button
if st.button("🌱 Recommend Crop"):
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    prediction = model.predict(input_data)[0]
    st.markdown(f"<div class='result-box'><h3>✅ Recommended Crop: <b>{prediction}</b></h3></div>", unsafe_allow_html=True)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Developed by Pranjalee Dahal 🌻</p>", unsafe_allow_html=True)
