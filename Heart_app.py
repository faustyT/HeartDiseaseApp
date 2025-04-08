#import streamlit library
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import requests
import matplotlib.pyplot as plt 

#load model
model = joblib.load('heart_disease_prediction.pkl')

# Sidebar
with st.sidebar:
    st.title("App Info")
    st.markdown("""
    **Heart Disease Predictor**

    - Model: Trained ML Classifier
    - Output: Probability of Heart Disease
    - Author: Tubo, Faustinah Nemieboka
    
    **Model Metrics**
    - Accuracy: 84.7%
    - F1 Score: 0.87
    - Threshold: 0.4
    """)

    st.markdown("---")
    st.info("Streamlit Web App")


# Title and animation
with st.container():
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.title("Heart Disease Prediction App")
        st.markdown("""
        Predict the likelihood of heart disease using your health metrics. 
        This tool uses a machine learning model trained on patient data to provide predictions.
        """)
    
# Input form
with st.form("input_form"):
    st.header("Enter Your Health Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age', min_value=1, max_value=120, value=50)
        sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0, 1])
        cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
        trestbps = st.number_input('Resting Blood Pressure', 80, 200, 120)

    with col2:
        chol = st.number_input('Cholesterol', 100, 600, 240)
        fbs = st.selectbox('Fasting Blood Sugar > 120? (1 = Yes, 0 = No)', [0, 1])
        restecg = st.selectbox('Resting ECG (0-2)', [0, 1, 2])
        thalach = st.number_input('Max Heart Rate Achieved', 60, 250, 150)

    with col3:
        exang = st.selectbox('Exercise Induced Angina (1 = Yes, 0 = No)', [0, 1])
        oldpeak = st.number_input('ST Depression', 0.0, 10.0, 1.0, step=0.1)
        slope = st.selectbox('Slope of ST (0-2)', [0, 1, 2])
        ca = st.selectbox('Number of Major Vessels (0-4)', [0, 1, 2, 3, 4])
        thal = st.selectbox('Thalassemia (1=normal, 2=fixed defect, 3=reversable defect)', [1, 2, 3])

    submitted = st.form_submit_button("Submit")


#predict button

if st.button("Predict"):
    user_input = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

    
    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"High risk of heart disease(Probability: {probability:.2f})")
    else:
        st.success(f"Low risk of heart disease(Probability: {probability:.2f})")

    # Pie chart of prediction probabilities
    labels = ['No Heart Disease', 'Heart Disease']
    sizes = [1 - probability, probability]
    colors = ['#66b3ff', '#ff9999']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

    
    # Bar chart of numeric inputs
    st.markdown("### Your Health Metrics Overview")
    health_data = pd.DataFrame({
        'Metric': ['Age', 'Resting BP', 'Cholesterol', 'Max HR', 'ST Depression'],
        'Value': [age, trestbps, chol, thalach, oldpeak]
    })
    st.bar_chart(data=health_data.set_index('Metric'))


    st.markdown("---")
    st.markdown("Thank you for using the app! Stay healthy")

