import streamlit as st
import pickle
import numpy as np

# Load model
with open("model/diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💉 DiaPredict: Diabetes Prediction App")
st.markdown("Use the form below to predict diabetes risk.")

pregnancies = st.number_input("Pregnancies", 0, 20, 0)
glucose = st.number_input("Glucose", 0, 200, 100)
blood_pressure = st.number_input("Blood Pressure", 0, 140, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 1, 100, 30)

if st.button("Predict"):
    data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    result = model.predict(data)[0]
    st.success("🟢 The person is Diabetic" if result == 1 else "🟢 The person is Not Diabetic")
