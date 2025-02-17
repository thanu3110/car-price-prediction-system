import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st

# Load Model
model = pkl.load(open('linear_regression_model.pkl', 'rb'))

# Custom Background Style
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://img.freepik.com/free-photo/weathered-concrete-surface-wallpaper-backdrop_53876-124398.jpg?semt=ais_hybrid");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.header('Car Price Prediction System')

# Load and preprocess data
car_data = pd.read_csv('car data.csv')

# Extract car brands from car names
car_data['Car_Name'] = car_data['Car_Name'].apply(lambda x: x.split(' ')[0])

# List of known bike brands to remove
bike_brands = ['Royal', 'UM', 'KTM', 'Bajaj', 'Hyosung', 'Mahindra', 'Honda', 'Yamaha', 'TVS', 'Hero', 'Activa', 'Suzuki']

# Filter out bike brands
car_data = car_data[~car_data['Car_Name'].isin(bike_brands)]

# User Inputs
car_name = st.selectbox('Select Car Brand', car_data['Car_Name'].unique())
year = st.number_input('Manufactured Year of Car', min_value=2010, max_value=2024, step=1)
kms_driven = st.number_input('No. of Kms Driven', min_value=500, max_value=100000, step=500)
fuel_type = st.selectbox('Select Fuel type', car_data['Fuel_Type'].unique())
transmission = st.selectbox('Select Transmission', car_data['Transmission'].unique())
owner = st.selectbox('Select Owner', car_data['Owner'].unique())
present_price = st.number_input('Present Price of Car', min_value=300000, max_value=6000000, step=50000)

# Encode categorical inputs
fuel_dict = {'Petrol': 2, 'Diesel': 1, 'CNG': 0}
transmission_dict = {'Manual': 1, 'Automatic': 0}

fuel_type_encoded = fuel_dict[fuel_type]
transmission_encoded = transmission_dict[transmission]

# Ensure Car Name Encoding
car_names = {name: idx for idx, name in enumerate(car_data['Car_Name'].unique())}
car_name_encoded = car_names.get(car_name, -1)  # Default to -1 if brand not found

if st.button("Predict"):
    # Default seller type encoding (0 = 'Dealer', 1 = 'Individual')
    seller_type_encoded = 0  # Assuming "Dealer" as default

    # Prepare test data
    x_test = pd.DataFrame(
        [[present_price * 1e-5, kms_driven, owner, year, car_name_encoded, fuel_type_encoded, seller_type_encoded, transmission_encoded]],
        columns=['Present_Price', 'Kms_Driven', 'Owner', 'Year', 'Car_Name_encoded', 'Fuel_Type_encoded', 'Seller_Type_encoded', 'Transmission_encoded']
    )

    # Predict Selling Price
    selling_price_predicted = model.predict(x_test)
    predicted_price = round(selling_price_predicted[0][0] * 1e5, 2)

    st.markdown(f'<p style="font-size: 20px;">Predicted Selling Car Price is around</p> <p style="color: red ;font-size: 40px;"> ₹ {predicted_price}</p>', unsafe_allow_html=True)
