import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model
with open('models/ridge.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Set the title of the app
st.title('Ridge Regression Model Prediction')

# Create input fields for the user to enter data
st.header('Input Features')
Temprature = st.number_input('Temperature (°C)', min_value=-30.0, max_value=80.0, value=20.0)
RH = st.number_input('Relative Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)
Ws = st.number_input('Wind Speed (m/s)', min_value=0.0, max_value=30.0, value=5.0)
Rain = st.number_input('Rainfall (mm)', min_value=0.0, max_value=500.0, value=0.0)
FFMC = st.number_input('Fine Fuel Moisture Code', min_value=0.0, max_value=100.0, value=50.0)
DMC = st.number_input('Duff Moisture Code', min_value=0.0, max_value=100.0, value=50.0)
ISI = st.number_input('Initial Spread Index', min_value=0.0, max_value=100.0, value=50.0)
classes = st.selectbox('Fire Class', ['0', '1'])
region = st.selectbox('Region', ['0', '1'])

# Predict button
if st.button('Predict'):
    # Prepare the input data
    input_data = np.array([[Temprature, RH, Ws, Rain, FFMC, DMC, ISI, classes, region]])
    
    # Scale the input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    # Display the prediction result
    st.subheader('Prediction Result')
    st.write(f'Predicted value: {prediction[0]:.2f}')