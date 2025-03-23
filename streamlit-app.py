import streamlit as st
import numpy as np
import pandas as pd
import joblib  
import tensorflow as tf

st.markdown(
    """
    <style>
    * {
        font-family: 'Consolas', monospace !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the saved Keras model
try:
    model = tf.keras.models.load_model("carbon_footprint_model.h5")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Now you can use `model.predict()` with user inputs


# Streamlit UI
st.title("Carbon Footprint Calculator")
st.write("Enter your details to estimate your carbon footprint.")

# Collect user inputs
Age = st.number_input("Age", min_value=0, max_value=120, value=30)
Gender = st.selectbox("Gender", ["Male", "Female", "Other"])
Mode_of_transport = st.selectbox("Mode of Transport", ["Car", "Bike", "Public Transport", "Walking", "EV", "Bicycle"])
Work_Hours = st.number_input("Work hours", min_value=0, max_value=10, value=0)
Shopping_Hours = st.number_input("Shopping hours", min_value=0, max_value=8, value=0)
Entertainment_Hours = st.number_input("Entertainment hours", min_value=0, max_value=24, value=0)
Home_Energy_Consumption_kWh = st.number_input("Home Energy Consumption (kWh)", min_value=0.0, max_value=12.0, value=0.0)
Charging_Station_Usage = st.number_input("Charging Station Usage", min_value=0, max_value=1, value=0)
Steps_Walked = st.number_input("Steps walked", min_value=0, max_value=20000, value=0)
Calories_Burned = st.number_input("Calories burned", min_value=300, max_value=15000, value=300)
Sleep_Hours = st.number_input("Sleep hours", min_value=0.0, max_value=24.0, value=6.0)
Social_Media_Hours = st.number_input("Social Media hours", min_value=0.0, max_value=24.0, value=1.0)
Public_Events_Hours = st.number_input("Public Event Hours", min_value=0.0, max_value=3.0, value=0.0)

# Convert categorical inputs
gender_map = {"Male": 0, "Female": 1, "Other": 2}
transport_map = {"Car": 0, "Bike": 1, "Public Transport": 2, "Walking": 3, "EV": 4, "Bicycle": 5}

input_data = np.array([
    Age,
    gender_map[Gender],
    transport_map[Mode_of_transport],
    Work_Hours,
    Shopping_Hours,
    Entertainment_Hours,
    Home_Energy_Consumption_kWh,
    Charging_Station_Usage,
    Steps_Walked,
    Calories_Burned,
    Sleep_Hours,
    Social_Media_Hours,
    Public_Events_Hours
]).reshape(1, -1)

# Predict carbon footprint
if st.button("Submit"):
    prediction = model.predict(input_data)  # Get the model's prediction

    # ✅ Ensure prediction is a float before displaying
    if isinstance(prediction, np.ndarray):  # If it's a NumPy array, extract first element
        prediction = float(prediction[0])
    elif isinstance(prediction, list):  # If it's a list, extract first element
        prediction = float(prediction[0])
    elif isinstance(prediction, tf.Tensor):  # If it's a TensorFlow tensor, convert to float
        prediction = float(prediction.numpy().item())

    st.success(f"Estimated Carbon Footprint: {prediction:.2f} kgCO2/month")  # Display result
