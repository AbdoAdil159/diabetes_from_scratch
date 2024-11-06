import joblib
import pandas as pd
import streamlit as st
from utils import *

def input_prep(input_: pd.DataFrame, data: pd.DataFrame)->pd.DataFrame:
    input_['Outcome'] = 0
    data = pd.concat([data, input_], ignore_index=True)
    X_, y_ = diabetes_data_prep(data)
    result = X_.iloc[-1]
    return pd.DataFrame([result.values], columns=X_.columns)

def predict(value: pd.DataFrame):    
    model = joblib.load('voting_clf.pkl')
    data = pd.read_csv('diabetes.csv')
    return model.predict(input_prep(value, data))


# Streamlit

st.title('Diabetes Prediction App')
st.write('*Developed by: Abdallah Adil Awad BASHIR*')

# Input fields
pregnancies = st.number_input('**Pregnancies**', min_value=0, max_value=20, value=0)
glucose = st.number_input('**Glucose**', min_value=0, max_value=200, value=100)
blood_pressure = st.number_input('**BloodPressure**', min_value=0, max_value=200, value=70)
skin_thickness = st.number_input('**SkinThickness**', min_value=0, max_value=50, value=20)
insulin = st.number_input('**Insulin**', min_value=0, max_value=800, value=80)
bmi = st.number_input('**BMI**', min_value=10, max_value=60, value=25)
diabetes_pedigree_function = st.number_input('**DiabetesPedigreeFunction**', min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input('**Age**', min_value=18, max_value=120, value=30)

# Create a DataFrame from the inputs
input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]],
                          columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

st.markdown("""
    <style>
        hr {
            border: 2px solid black;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown('<hr>', unsafe_allow_html=True)

# Button to make the prediction
if st.button('Predict Diabetes'):
    probability = predict(input_data)
    if probability == 0:        
        st.write(f'This person has **diabetes** based on the prediction')
    else:
        st.write(f'This person **does not have diabetes** based on the prediction')
