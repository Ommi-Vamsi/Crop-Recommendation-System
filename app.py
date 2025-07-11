import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the pre-trained model
model = joblib.load('model.pkl')

# Function to preprocess input data
def preprocess_input(data):
    # Convert input data to DataFrame
    df = pd.DataFrame(data, index=[0])
    # Perform any preprocessing steps (scaling, encoding, etc.) if necessary
    return df

# Function to predict crop
def predict_crop(input_data):
    # Preprocess input data
    input_df = preprocess_input(input_data)
    # Perform prediction
    prediction = model.predict(input_df)
    return prediction

# Main function to run the app
def main():
    # Set title and description
    st.title('Crop Prediction App')
    st.write('This app predicts the crop based on input features.')
    
    # Add input fields for features
    st.sidebar.title('Input Features')
    N = st.sidebar.number_input('N', value=0.0)
    P = st.sidebar.number_input('P', value=0.0)
    K = st.sidebar.number_input('K', value=0.0)
    temperature = st.sidebar.number_input('Temperature', value=0.0)
    humidity = st.sidebar.number_input('Humidity', value=0.0)
    ph = st.sidebar.number_input('pH', value=0.0)
    rainfall = st.sidebar.number_input('Rainfall', value=0.0)
    
    # Predict button
    if st.sidebar.button('Predict'):
        # Create input data dictionary
        input_data = {
            'N': N,
            'P': P,
            'K': K,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        }
        # Get prediction
        prediction = predict_crop(input_data)
        # Display prediction
        st.write('Predicted Crop:', prediction)
        
    # Table of sowing information
    st.subheader('Sowing Information')
    sowing_info = {
            'Crop': ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

    }
    sowing_df = pd.DataFrame(sowing_info)
    st.table(sowing_df)

# Run the app
if __name__ == '__main__':
    main()
