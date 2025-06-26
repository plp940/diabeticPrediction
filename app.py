import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix,classification_report 
import joblib
import streamlit as st

# Load the dataset
model = keras.models.load_model("Diabetic_model.h5")
scaler = joblib.load('Scaler.pkl')

st.set_page_config(page_title="Diabetes Prediction App", page_icon=":hospital:", layout="wide") 
st.title("Diabetes Prediction App")
st.write("This app predicts whether a patient has diabetes based on medical attributes.")
st.markdown("enter the medical attributes below:")
# Input fields for user data  
#slider or input number st.slier,st.number_input
def user_input_features():
    pregnancies = st.slider("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.slider("Glucose", min_value=0, max_value=200, value=0)
    blood_pressure = st.slider("Blood Pressure", min_value=0, max_value=200, value=0)
    skin_thickness = st.slider("Skin Thickness", min_value=0, max_value=100, value=0)
    insulin = st.slider("Insulin", min_value=0, max_value=500, value=0)
    bmi = st.slider("BMI", min_value=0.0, max_value=50.0, value=0.0)
    diabetes_pedigree_function = st.slider("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.0)
    age = st.slider("Age", min_value=1, max_value=120, value=18)
    # Create a DataFrame with the input data
    if st.button("Predict"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])  
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0][0]
        result = "Diabetic" if prediction > 0.5 else "Not Diabetic"
        st.success(f"The model predicts: {result} (Probability: {prediction:.2f})")

user_input_features()
