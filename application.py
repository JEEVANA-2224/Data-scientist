import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- Page config ---
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("❤️ Heart Disease Prediction App")

# --- Load model & scaler ---
@st.cache_resource
def load_model():
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# --- Sidebar input for features ---
st.sidebar.header("Input Features")

def user_input_features():
    age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.sidebar.selectbox("Chest Pain Type", [1,2,3,4])
    bp = st.sidebar.number_input("BP", value=120)
    chol = st.sidebar.number_input("Cholesterol", value=200)
    fbs = st.sidebar.selectbox("FBS over 120", [0,1])
    ekg = st.sidebar.selectbox("EKG results", [0,1,2])
    max_hr = st.sidebar.number_input("Max HR", value=150)
    exercise_angina = st.sidebar.selectbox("Exercise Angina", ["Yes","No"])
    st_depression = st.sidebar.number_input("ST depression", value=1.0)
    slope = st.sidebar.selectbox("Slope of ST", [0,1,2])
    num_vessels = st.sidebar.selectbox("Number of vessels fluro", [0,1,2,3])
    thallium = st.sidebar.selectbox("Thallium", [0,1,2,3])
    
    # Convert categorical to numerical
    sex = 1 if sex=="Male" else 0
    exercise_angina = 1 if exercise_angina=="Yes" else 0
    
    features = [age, sex, chest_pain, bp, chol, fbs, ekg, max_hr, exercise_angina,
                st_depression, slope, num_vessels, thallium]
    return np.array([features])

input_data = user_input_features()

# --- Predict button ---
if st.button("Predict"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    result = "Presence" if prediction==1 else "Absence"
    st.success(f"Predicted Heart Disease: {result}")

# --- Optional: show stripplot for demonstration ---
if st.checkbox("Show prediction plot"):
    # Example: fake y_test & y_pred for demo purpose
    y_test = np.random.randint(0,2,50)
    y_pred_demo = np.random.randint(0,2,50)
    
    sns.stripplot(x=y_test, y=y_pred_demo, jitter=True)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted Heart Disease")
    st.pyplot(plt)
