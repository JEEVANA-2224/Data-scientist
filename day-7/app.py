import streamlit as st
import numpy as np
import pickle

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="KNN Purchase Predictor",
    layout="centered"
)

st.title("🛒 Purchase Prediction using KNN")
st.write("Predict whether a customer will purchase based on **Age** and **Estimated Salary**")

# ---------------------------
# Load Model and Scaler
# ---------------------------
@st.cache_resource
def load_model_and_scaler():
    with open("knn_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model_and_scaler()

# ---------------------------
# User Inputs
# ---------------------------
age = st.number_input("Enter Age", min_value=10, max_value=100, value=30)
salary = st.number_input("Enter Estimated Salary", min_value=1000, max_value=300000, value=50000)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict"):
    input_data = np.array([[age, salary]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Customer is likely to PURCHASE")
    else:
        st.error("❌ Customer is NOT likely to purchase")