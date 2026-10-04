import streamlit as st
import pickle
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Random Forest Prediction",
    page_icon="🌳",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "random(1).pkl")

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("❌ Model file 'random(1).pkl' not found.")
    st.info("Make sure random(1).pkl is in the same folder as app.py")
    st.stop()

# -----------------------------
# Title
# -----------------------------
st.title("🌳 Random Forest Prediction App")
st.write("Enter the required values below to get a prediction.")

# -----------------------------
# Prediction
# -----------------------------
st.subheader("📋 Input Details")

# Add your model's actual input columns here
# Example:
col1, col2 = st.columns(2)

with col1:
    feature1 = st.number_input("Feature 1", value=0.0)

with col2:
    feature2 = st.number_input("Feature 2", value=0.0)

if st.button("🔮 Predict"):
    try:
        prediction = model.predict([[feature1, feature2]])

        st.success(f"✅ Prediction: {prediction[0]}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
    
        
