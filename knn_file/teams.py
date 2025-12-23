import streamlit as st
import pickle
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# App title
st.title("🛒 Shopping App")
st.write("Predict if a person is likely to purchase based on their profile")

# Load model and scaler
model = pickle.load(open(os.path.join(BASE_DIR, "student_final_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "student_scaler.pkl"), "rb"))

# ---------------- User Inputs ----------------
age = st.number_input("Enter Age", min_value=1, max_value=100, value=30)
salary = st.number_input("Enter Estimated Salary", min_value=1000, max_value=200000, value=50000)
gender = st.selectbox("Select Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)

# Optional: convert categorical inputs to numeric
gender_num = 1 if gender == "Male" else 0
marital_num = 1 if marital_status == "Married" else 0

# Prepare input data
new_data = [[age, salary, gender_num, marital_num, experience]]
new_data_scaled = scaler.transform(new_data)

# Prediction
prediction = model.predict(new_data_scaled)

# Display result
if prediction[0] == 1:
    st.success("✅ Person is likely to PURCHASE")
else:
    st.warning("❌ Person is NOT likely to purchase")
