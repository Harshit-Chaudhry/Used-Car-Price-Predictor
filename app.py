import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import pickle
from PIL import Image

st.title("Used Car Price Prediction")

image = Image.open('image copy.png')
st.image(image, 'Wanna to buy a 2nd Hand Car ')

required_columns = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats', 
                    'model_3', 'model_5', 'model_6', 'model_7', 'model_A4', 'model_A6', 
                    'model_A8', 'model_Alto', 'model_Altroz', 'model_Alturas', 'model_Amaze', 
                    'model_Aspire', 'model_Aura', 'model_Baleno', 'model_Bolero', 'model_C', 
                    'model_C-Class', 'model_CLS', 'model_CR', 'model_CR-V', 'model_Camry', 
                    'model_Carnival', 'model_Cayenne', 'model_Celerio', 'model_Ciaz', 'model_City', 
                    'model_Civic', 'model_Compass', 'model_Continental', 'model_Cooper', 'model_Creta', 
                    'model_D-Max', 'model_Duster', 'model_Dzire LXI', 'model_Dzire VXI', 'model_Dzire ZXI', 
                    'model_E-Class', 'model_ES', 'model_Ecosport', 'model_Eeco', 'model_Elantra', 
                    'model_Endeavour', 'model_Ertiga', 'model_F-PACE', 'model_Figo', 'model_Fortuner', 
                    'model_Freestyle', 'model_GL-Class', 'model_GLS', 'model_GO', 'model_GTC4Lusso', 
                    'model_Ghibli', 'model_Ghost', 'model_Glanza', 'model_Grand', 'model_Gurkha', 
                    'model_Harrier', 'model_Hector', 'model_Hexa', 'model_Ignis', 'model_Innova', 
                    'model_Jazz', 'model_KUV', 'model_KUV100', 'model_KWID', 'model_Kicks', 'model_MUX', 
                    'model_Macan', 'model_Marazzo', 'model_NX', 'model_Nexon', 'model_Octavia', 
                    'model_Panamera', 'model_Polo', 'model_Q7', 'model_Quattroporte', 'model_RX', 
                    'model_Rapid', 'model_RediGO', 'model_Rover', 'model_S-Class', 'model_S-Presso', 
                    'model_S90', 'model_Safari', 'model_Santro', 'model_Scorpio', 'model_Seltos', 
                    'model_Superb', 'model_Swift', 'model_Swift Dzire', 'model_Thar', 'model_Tiago', 
                    'model_Tigor', 'model_Triber', 'model_Tucson', 'model_Vento', 'model_Venue', 
                    'model_Verna', 'model_Vitara', 'model_WR-V', 'model_Wagon R', 'model_Wrangler', 
                    'model_X-Trail', 'model_X1', 'model_X3', 'model_X4', 'model_X5', 'model_XC', 
                    'model_XC60', 'model_XC90', 'model_XE', 'model_XF', 'model_XL6', 'model_XUV300', 
                    'model_XUV500', 'model_Yaris', 'model_Z4', 'model_i10', 'model_i20', 'model_redi-GO', 
                    'seller_type_Dealer', 'seller_type_Individual', 'seller_type_Trustmark Dealer', 
                    'fuel_type_CNG', 'fuel_type_Diesel', 'fuel_type_Electric', 'fuel_type_LPG', 
                    'fuel_type_Petrol', 'transmission_type_Automatic', 'transmission_type_Manual']

brand_name = [
    'Maruti', 'Hyundai', 'Ford', 'Renault', 'Mini', 'Mercedes_Benz',
    'Toyota', 'Volkswagen', 'Honda', 'Mahindra', 'Datsun', 'Tata',
    'Kia', 'BMW', 'Audi', 'Land_Rover', 'Jaguar', 'MG', 'Isuzu',
    'Porsche', 'Skoda', 'Volvo', 'Lexus', 'Jeep', 'Maserati',
    'Bentley', 'Nissan', 'ISUZU', 'Ferrari', 'Mercedes_AMG',
    'Rolls_Royce', 'Force'
]
Maruti_brand = ['Alto', 'Wagon R', 'Swift', 'Ciaz', 'Baleno', 'Swift Dzire', 'Ignis', 'Vitara', 'Celerio', 'Ertiga', 'Eeco', 'Dzire VXI', 'XL6', 'S-Presso', 'Dzire LXI', 'Dzire ZXI']
Hyundai_brand = ['Grand', 'i20', 'i10', 'Venue', 'Verna', 'Creta', 'Santro', 'Elantra', 'Aura', 'Tucson']
Ford_brand = ['Ecosport', 'Aspire', 'Figo', 'Endeavour', 'Freestyle']
Renault_brand = ['Duster', 'KWID', 'Triber']
Mini_brand = ['Cooper']
Mercedes_Benz_brand = ['C-Class', 'E-Class', 'GL-Class', 'S-Class', 'CLS', 'GLS']
Toyota_brand = ['Innova', 'Fortuner', 'Camry', 'Yaris', 'Glanza']
Volkswagen_brand = ['Vento', 'Polo']
Honda_brand = ['City', 'Amaze', 'CR-V', 'Jazz', 'Civic', 'WR-V', 'CR']
Mahindra_brand = ['Bolero', 'XUV500', 'KUV100', 'Scorpio', 'Marazzo', 'KUV', 'Thar', 'XUV300', 'Alturas']
Datsun_brand = ['RediGO', 'GO', 'redi-GO']
Tata_brand = ['Tiago', 'Tigor', 'Safari', 'Hexa', 'Nexon', 'Harrier', 'Altroz']
Kia_brand = ['Seltos', 'Carnival']
BMW_brand = ['5', '3', 'Z4', '6', 'X5', 'X1', '7', 'X3', 'X4']
Audi_brand = ['A4', 'A6', 'Q7', 'A8']
Land_Rover_brand = ['Rover']
Jaguar_brand = ['XF', 'F-PACE', 'XE']
MG_brand = ['Hector']
Isuzu_brand = ['D-Max', 'MUX']
Porsche_brand = ['Cayenne', 'Macan', 'Panamera']
Skoda_brand = ['Rapid', 'Superb', 'Octavia']
Volvo_brand = ['S90', 'XC', 'XC90', 'XC60']
Lexus_brand = ['ES', 'NX', 'RX']
Jeep_brand = ['Wrangler', 'Compass']
Maserati_brand = ['Ghibli', 'Quattroporte']
Bentley_brand = ['Continental']
Nissan_brand = ['Kicks', 'X-Trail']
ISUZU_brand = ['MUX']
Ferrari_brand = ['GTC4Lusso']
Mercedes_AMG_brand = ['C']
Rolls_Royce_brand = ['Ghost']
Force_brand = ['Gurkha']


seller_name = ['Individual', 'Dealer', 'Trustmark Dealer']
fuel_name = ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric']
transmission_name = ['Manual', 'Automatic']

def model(b):
    return eval(b + "_brand")

with st.sidebar:
    car_name = st.text_input('Car Name')
    brand = st.selectbox('Brand', brand_name)
    model = st.selectbox('Model', model(brand))
    vehicle_age = st.slider('Vehicle Age', min_value=0, max_value=29, value=0)
    km_driven = st.slider('Kilometers Driven', min_value=100, max_value=3800000, value=100, step=1000)
    seller_type = st.selectbox('Seller Type', seller_name)
    fuel_type = st.selectbox('Fuel Type', fuel_name)
    transmission_type = st.selectbox('Transmission Type', transmission_name)
    mileage = st.slider('Mileage', min_value=4.0, max_value=33.54, value=4.0, step=0.1)
    engine = st.slider('Engine', min_value=793, max_value=6592, value=793, step=10)
    max_power = st.slider('Max Power', min_value=38.4, max_value=626.0, value=38.4, step=1.0)
    seats = st.slider('Seats', min_value=0, max_value=9, value=5)

user_data = pd.DataFrame({
    'model': [model],
    'vehicle_age': [vehicle_age],
    'km_driven': [km_driven],
    'seller_type': [seller_type],
    'fuel_type': [fuel_type],
    'transmission_type': [transmission_type],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'seats': [seats]
})

user_data = pd.get_dummies(user_data, columns=['model','seller_type','fuel_type','transmission_type'])

missing_cols = set(required_columns) - set(user_data.columns)
for col in missing_cols:
    user_data[col] = 0

user_df = user_data[required_columns]

st.write(user_data.head())

dtr_model = joblib.load('dtr_model.pkl')

pred=dtr_model.predict(user_df)[0]

def format_inr(amount):
    return f"₹{amount:,.2f}"

if st.button('Predict'):
    formatted_pred = format_inr(pred)
    st.subheader('Price of Used Car')
    st.subheader(formatted_pred)
