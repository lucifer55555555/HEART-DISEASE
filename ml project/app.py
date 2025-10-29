# ============================================
# 🫀 Heart Disease Prediction Streamlit App (Final Fixed Version)
# ============================================

import streamlit as st
import numpy as np
import joblib

# --------------------------------------------
# Load trained model and scaler
# --------------------------------------------
import os
model = joblib.load(os.path.join(os.path.dirname(__file__), "best_heart_model_all_features.pkl"))

scaler = joblib.load("scaler.pkl")

# --------------------------------------------
# Streamlit Page Setup
# --------------------------------------------
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")
st.title("🫀 Heart Disease Prediction App")
st.write("Predict the likelihood of heart disease based on medical attributes.")

st.sidebar.header("User Input Parameters")
st.sidebar.info("Adjust the sliders to simulate patient data and predict heart disease risk.")

# --------------------------------------------
# Input Fields (Match Training Feature Order)
# --------------------------------------------
age = st.sidebar.slider("Age", 20, 80, 45)
sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.sidebar.slider("Chest Pain Type (0-3)", 0, 3, 1)
trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [0, 1])
restecg = st.sidebar.slider("Resting ECG (0-2)", 0, 2, 1)
thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 170)
exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 0.0)
slope = st.sidebar.slider("Slope of ST Segment (0-2)", 0, 2, 2)
ca = st.sidebar.slider("Number of Major Vessels (0-4)", 0, 4, 0)
thal = st.sidebar.slider("Thal (0-3)", 0, 3, 2)

# --------------------------------------------
# Derived Features (same as in training)
# --------------------------------------------
chol_bp_ratio = chol / trestbps
age_oldpeak = age * oldpeak
bmi_age_ratio = age / (thalach + 1)
slope_cp = slope * cp
combined_stress = oldpeak * exang

# Combine all inputs (must match training feature order)
# Combine only the selected 10 features
features = np.array([[
    age, sex, cp, thalach, exang, oldpeak, slope, ca, thal, chol
]])

# --------------------------------------------
# Prediction
# --------------------------------------------
if st.button("🔍 Predict"):
    try:
        # Scale input same as training
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_scaled)[0][1] * 100
        else:
            proba = None

        # Display result
        if prediction == 1:
            st.error(f"⚠️ The model predicts a **high risk of heart disease** ({proba:.2f}% probability).")
        else:
            st.success(f"✅ The model predicts **no heart disease** ({100 - proba:.2f}% probability).")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# --------------------------------------------
# Footer
# --------------------------------------------
st.markdown("---")
st.caption("Developed by [PRATHAM] — CSE (AIML) | ML Mini Project 2025")

