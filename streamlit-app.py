import streamlit as st
import numpy as np
import pandas as pd
import joblib  
import tensorflow as tf

# Load the saved Keras model
try:
    model = tf.keras.models.load_model("carbon_footprint_model.h5")
    st.write("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Now you can use `model.predict()` with user inputs


# Streamlit UI
st.title("Carbon Footprint Calculator")
st.write("Enter your details to estimate your carbon footprint.")

# Collect user inputs
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
mode_of_transport = st.selectbox("Mode of Transport", ["Car", "Bike", "Bus", "Train", "Walking", "Other"])
daily_mileage = st.number_input("Daily Mileage (km)", min_value=0.0, value=10.0)
electricity_usage = st.number_input("Monthly Electricity Usage (kWh)", min_value=0.0, value=100.0)
water_usage = st.number_input("Daily Water Usage (liters)", min_value=0.0, value=100.0)
meat_consumption = st.number_input("Meat Consumption (kg/week)", min_value=0.0, value=2.0)
public_event_hours = st.number_input("Public Event Hours (per month)", min_value=0.0, value=5.0)

# Convert categorical inputs
gender_map = {"Male": 0, "Female": 1, "Other": 2}
transport_map = {"Car": 0, "Bike": 1, "Bus": 2, "Train": 3, "Walking": 4, "Other": 5}

input_data = np.array([
    age,
    gender_map[gender],
    transport_map[mode_of_transport],
    daily_mileage,
    electricity_usage,
    water_usage,
    meat_consumption,
    public_event_hours
]).reshape(1, -1)

# Predict carbon footprint
if st.button("Submit"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Carbon Footprint: {prediction:.2f} kgCO2/month")
