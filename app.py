import tensorflow
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

model = load_model("fuel.h5")

st.title("Fuel Efficiency Prediction")
cylinders = st.number_input("Cylinders")
displacement = st.number_input("Displacement")
horsepower = st.number_input("Horsepower")
weight = st.number_input("Weight")
acceleration = st.number_input("Acceleration")
model_year = st.number_input("Model Year")

if st.button("Predict"):
    test = np.array([[cylinders, displacement, horsepower, weight, acceleration, model_year]])
    res = model.predict(test)
    print(res)
    st.success("Predicted: " + str(res[0][0]))
