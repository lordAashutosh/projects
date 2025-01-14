import pandas as pd
import streamlit as st
from joblib import load

# Load the trained model
model = load('car_price_predictor.joblib')

# Load the dataset
file_path = 'used_cars.csv'  # Replace with your actual dataset path
data = pd.read_csv(file_path)
car_models = data['model'].unique()

# Streamlit app
st.title("Car Price Prediction App")
st.title(itz me again, anchor=None, *, help=None)

# Input fields
car_model = st.selectbox("Select Car Model", car_models)
car_age = st.number_input("Enter Car Age", min_value=0, max_value=50, value=5)

# Encode selected car model
encoded_model = pd.factorize([car_model])[0][0]  # Convert to numerical label

# Predict button
if st.button("Predict Price"):
    predicted_price = model.predict([[encoded_model, car_age]])[0]
    st.success(f"Predicted Price: ${predicted_price:,.2f}")

