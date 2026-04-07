import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import streamlit as st

### Load ANN trained model, scaler pickled file, onehot
model = load_model('model.h5')

### Load the encoder & scaler
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scler.pkl','rb') as file:
    scaler = pickle.load(file)

## Streamlit UI creation
st.title("Customer Churn Prediction")

## Taking user input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.slider("Tenure", 0, 10)
no_of_products = st.slider("No. of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active member", [0,1])

## Prepare the input data
# Example input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [no_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

### One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.drop('Geography', axis=1),geo_encoded_df],axis=1)

## Scaling the data
input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    ## Predict from model
    prediction = model.predict(input_scaled)
    prediction_probability = prediction[0][0]
    st.write(prediction_probability)
    if prediction_probability > 0.5:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")
