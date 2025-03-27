import numpy as np
import pandas as pd
import pickle as pkl
import streamlit as st

# Load Model with Error Handling
try:
    with open('linear_regression_model.pkl', 'rb') as file:
        model = pkl.load(file)
except Exception as e:
    st.error(f"🚨 Error loading model: {e}")
    st.stop()

# Set up Page Navigation
st.sidebar.title("🚗 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Car Price Prediction", "🛒 Buy a Car", "📢 Sell Your Car"])

# Home Page
if page == "🏠 Home":
    st.title("🚗 Welcome to JANATA GARAGE")
    st.write("🔹 Use to predict car prices, buy cars, and sell your car easily!")
    
    st.image("C:/Users/HP/Car_Price_Prediction_System/car_home.jpg", use_column_width=True)

    
    st.markdown("""
    ### Features:
    - 🏎 **Car Price Prediction** – Estimate your car’s resale value
    - 🛒 **Buy a Car** – Browse cars for sale
    - 📢 **Sell Your Car** – List your car for sale
    """)

# Car Price Prediction Page
elif page == "📊 Car Price Prediction":
    st.title("📊 Car Price Prediction")
    
    # Load and preprocess data
    car_data = pd.read_csv('car data.csv')

    # Extract car brands
    car_data['Car_Name'] = car_data['Car_Name'].apply(lambda x: x.split(' ')[0])

    # Remove bike brands
    bike_brands = ['Royal', 'UM', 'KTM', 'Bajaj', 'Hyosung', 'Mahindra', 'Honda', 'Yamaha', 'TVS', 'Hero', 'Activa', 'Suzuki']
    car_data = car_data[~car_data['Car_Name'].isin(bike_brands)]

    # User Inputs
    car_name = st.selectbox('🚘 Select Car Brand', car_data['Car_Name'].unique())
    year = st.number_input('📅 Manufactured Year', min_value=2010, max_value=2024, step=1)
    kms_driven = st.number_input('🛣 KMs Driven', min_value=500, max_value=100000, step=500)
    fuel_type = st.selectbox('⛽ Fuel Type', car_data['Fuel_Type'].unique())
    transmission = st.selectbox('⚙️ Transmission', car_data['Transmission'].unique())
    owner = st.selectbox('👤 Owner Type', [1, 2, 3])
    present_price = st.number_input('💰 Present Price', min_value=300000, max_value=6000000, step=50000)

    # Encode categorical inputs
    fuel_dict = {'Petrol': 2, 'Diesel': 1, 'CNG': 0}
    transmission_dict = {'Manual': 1, 'Automatic': 0}
    fuel_type_encoded = fuel_dict[fuel_type]
    transmission_encoded = transmission_dict[transmission]

    # Encode Car Brand
    car_names = {name: idx for idx, name in enumerate(car_data['Car_Name'].unique())}
    car_name_encoded = car_names.get(car_name, len(car_names))

    if st.button("🚀 Predict Price"):
        with st.spinner('🔍 Predicting... Please wait...'):
            try:
                # Prepare test data
                x_test = pd.DataFrame(
                    [[present_price * 1e-5, kms_driven, owner, year, car_name_encoded, fuel_type_encoded, 0, transmission_encoded]],
                    columns=['Present_Price', 'Kms_Driven', 'Owner', 'Year', 'Car_Name_encoded', 'Fuel_Type_encoded', 'Seller_Type_encoded', 'Transmission_encoded']
                )

                # Predict Selling Price
                selling_price_predicted = model.predict(x_test)
                predicted_price = round(float(selling_price_predicted[0]) * 1e5, 2)

                st.success(f"💰 Predicted Selling Price: ₹ {predicted_price}")

            except Exception as e:
                st.error(f"⚠️ Prediction error: {e}")

# Car Buying Page
elif page == "🛒 Buy a Car":
    st.title("🛒 Cars for Sale")

    # Example car listings
    cars_for_sale = [
        # 🚗 Budget-Friendly Cars
   {"name": "Maruti Swift 2021", "price": "₹5,80,000", "image": "maruti_swift.jpeg"},
    {"name": "Renault Kwid 2021", "price": "₹4,80,000", "image": "Renault KWID.jpeg"},
   
    # 🚘 Mid-Range Cars
   {"name": "Hyundai Creta 2020", "price": "₹9,90,000", "image": "Hyundai Creta 2020.jpeg"},
        {"name": "Honda City 2018", "price": "₹7,40,000", "image": "Honda City 2018.jpeg"},
    {"name": "Tata Nexon 2020", "price": "₹8,00,000", "image": "tata_nexon.jpeg"},
    
    # 🚙 Premium Cars
    {"name": "Toyota Innova Crysta 2019", "price": "₹17,50,000", "image": "toyota_innova.jpeg"},
    {"name": "Ford Endeavour 2020", "price": "₹29,90,000", "image": "ford_endeavour.jpeg"},
    {"name": "Mahindra XUV700 2022", "price": "₹21,30,000", "image": "mahindra_xuv700.jpeg"},
    
    # 🏎️ Luxury Cars
    {"name": "Mercedes-Benz C-Class 2021", "price": "₹52,00,000", "image": "mercedes_c_class.jpeg"},
    {"name": "BMW X5 2022", "price": "₹84,00,000", "image": "bmw_x5.jpeg"},
    {"name": "Audi Q7 2021", "price": "₹79,50,000", "image": "audi_q7.jpeg"},
    {"name": "Jaguar F-Pace 2020", "price": "₹75,00,000", "image": "jaguar_f_pace.jpeg"}
    ]

    # Display car listings
    for car in cars_for_sale:
        st.image(car["image"], width=300)
        st.write(f"**{car['name']}**")
        st.write(f"💰 **Price:** {car['price']}")
        st.button(f"🔍 View Details - {car['name']}", key=car["name"])

# Car Selling Page
elif page == "📢 Sell Your Car":
    st.title("📢 Sell Your Car")

    st.write("🔹 Fill out the form below to list your car for sale.")

    # Input Fields
    seller_name = st.text_input("📝 Your Name")
    contact_number = st.text_input("📞 Contact Number")
    car_brand = st.selectbox("🚘 Car Brand", ["Maruti", "Hyundai", "Toyota", "Ford", "Honda", "Other"])
    model = st.text_input("🚗 Car Model")
    year = st.number_input("📅 Year of Manufacture", min_value=2000, max_value=2024, step=1)
    kms_driven = st.number_input("🛣 KMs Driven", min_value=500, max_value=200000, step=500)
    expected_price = st.number_input("💰 Expected Price", min_value=100000, max_value=10000000, step=50000)

    if st.button("📤 Submit Listing"):
        st.success(f"🎉 Your {car_brand} {model} has been listed successfully!")
