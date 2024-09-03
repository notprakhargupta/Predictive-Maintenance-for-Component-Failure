# app/app.py
import streamlit as st
import pandas as pd
import joblib

# Load models
rf_model = joblib.load('../models/random_forest_model.pkl')
xgb_model = joblib.load('../models/xgboost_model.pkl')

# Streamlit UI
st.title('Predictive Maintenance for Component Failure')

# User input for machine telemetry data
volt = st.number_input('Voltage', min_value=0.0)
rotate = st.number_input('Rotation', min_value=0.0)
pressure = st.number_input('Pressure', min_value=0.0)
vibration = st.number_input('Vibration', min_value=0.0)

# Model prediction
if st.button('Predict'):
    input_data = pd.DataFrame([[volt, rotate, pressure, vibration]], columns=['volt', 'rotate', 'pressure', 'vibration'])
    prediction = xgb_model.predict(input_data)[0]
    
    if prediction == 1:
        st.write("The machine is likely to fail within the next 7 days.")
    else:
        st.write("The machine is not likely to fail within the next 7 days.")
