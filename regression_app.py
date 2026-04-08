import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import streamlit as st

### Load ANN trained model, scaler pickled file, onehot
model = load_model('regression_model.h5')

### Load the encoder & scaler
with open('reg_label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('reg_onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('reg_scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## Streamlit UI creation
st.title("Customer Salary Prediction")

## Taking user input
geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
tenure = st.slider("Tenure", 0, 10)
no_of_products = st.slider("No. of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0,1])
is_active_member = st.selectbox("Is Active member", [0,1])
exited = st.selectbox("Is Exited", [0,1])

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
    'Exited': [exited]
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
    prediction_salary = prediction[0][0]
    st.markdown(f"**Estimated Salary : *{prediction_salary:.2f}***")
    

# https://ann-classification-churn-techiemini.streamlit.app/