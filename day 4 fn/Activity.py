import streamlit as st
import numpy as np
import pickle

# --- Page Config ---
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# --- Title ---
st.title("🎓 Student Performance Predictor")
st.write("Predict a student's **Final Score** using study habits, attendance, and lifestyle data.")

# --- Load Model & Scaler ---
@st.cache_resource
def load_model():
    # Update paths to your actual .pkl files
    model_path = r"C:\Users\jeeva\Documents\Data-scientist\day 4 fn\student_final_model.pkl"
    scaler_path = r"C:\Users\jeeva\Documents\Data-scientist\day 4 fn\scaler.pkl"
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# --- Sidebar Inputs ---
st.sidebar.header("📊 Student Details")

study_hours = st.sidebar.slider("Study Hours per Week", 0.0, 80.0, 10.0)
attendance = st.sidebar.slider("Attendance Percentage", 0.0, 100.0, 75.0)
previous_score = st.sidebar.slider("Previous Semester Score", 0.0, 100.0, 60.0)
sleep_hours = st.sidebar.slider("Sleep Hours per Day", 0.0, 12.0, 7.0)
travel_time = st.sidebar.slider("Travel Time (hours/day)", 0.0, 5.0, 1.0)
library_usage = st.sidebar.slider("Library Usage per Week", 0.0, 40.0, 5.0)

# --- Prepare Input Array ---
input_data = np.array([[
    study_hours,
    attendance,
    previous_score,
    sleep_hours,
    travel_time,
    library_usage
]])

# --- Scale Input ---
input_scaled = scaler.transform(input_data)

# --- Prediction ---
if st.button("🚀 Predict Final Score"):
    prediction = model.predict(input_scaled)[0]
    
    st.success(f"📈 Predicted Final Score: **{prediction:.2f}**")
    
    if prediction >= 75:
        st.balloons()
        st.write("🔥 Excellent performance expected!")
    elif prediction >= 50:
        st.write("👍 Average to good performance.")
    else:
        st.warning("⚠ Needs improvement. More consistency required.")

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ using Linear Regression & Streamlit")
