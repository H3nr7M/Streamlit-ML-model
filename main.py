# In this file we will create a web app using streamlit and our trained model from 'trained_model.sav'

import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def diabetes_prediction(input_data):

    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    return "No diabetic" if prediction == 0 else "diabetic"

def main():
    
    # giving a title
    st.title('Diabetes Prediction Web App')
       
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
   
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Glucose, BloodPressure,  Insulin, BMI])
        
    st.success(diagnosis)
  
    
if __name__ == '__main__':
    main()

# this file can be deployed on Streamlit https://streamlit.io/ 
# I jope this easy tutorial will help you to understand how to deploy a machine learning model on web app using streamlit