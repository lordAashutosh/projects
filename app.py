import pandas as pd
import streamlit as st
from joblib import load
model = load('car_price_predictor.joblib')
file_path = 'used_cars.csv'  
data = pd.read_csv(file_path)
car_models = data['model'].unique()
st.title("Car Price Prediction App")

car_model = st.selectbox("Select Car Model", car_models)
car_age = st.number_input("Enter Car Age", min_value=0, max_value=50, value=5)
encoded_model = pd.factorize([car_model])[0][0] 
if st.button("Predict Price"):
    predicted_price = model.predict([[encoded_model, car_age]])[0]
    st.success(f"Predicted Price: ${predicted_price:,.2f}")

