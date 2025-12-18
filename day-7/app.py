import streamlit as st
import numpy as np
import pickle
import os

st.set_page_config(page_title="KNN Purchase Prediction", layout="centered")
st.title("🛒 KNN Purchase Predictor")

@st.cache_resource
def load_model():
    if not os.path.exists("knn_model_file.pkl"):
        st.error("❌ knn_model.pkl not found")
        st.stop()

    if not os.path.exists("scaler.pkl"):
        st.error("❌ scaler.pkl not found")
        st.stop()

    with open("knn_model_file.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model()

age = st.number_input("Enter Age", 18, 100, 30)
salary = st.number_input("Enter Estimated Salary", 1000, 200000, 50000, step=1000)

if st.button("Predict"):
    input_data = np.array([[age, salary]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.success("✅ Customer is likely to PURCHASE")
    else:
        st.error("❌ Customer is NOT likely to purchase")
