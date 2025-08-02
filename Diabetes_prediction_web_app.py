import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import streamlit as st

scaler = StandardScaler()

#loading the model
loaded_model = pickle.load(open('D:/Python/trained_model.sav', 'rb'))

def diabetes_prediction(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    std_data = scaler.fit_transform(input_data_reshaped)

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if(prediction[0] == 1):
      return "Person is diabetic"
    else:
      return "Person is non-diabetic"
  
def main():
    
    #giving title to the web page
    st.title("Diabetes Prediction Web App")
    
    #taking input from the user
    Pregnancies = st.text_input("No of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin Tevel")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")
    
    #code for prediction
    diagnosis = ''
    
    if st.button('Diabetes Test Result'):

        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
    st.success(diagnosis)
    
    gradient_css = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
      background: linear-gradient(
        135deg,
        #4f2991 10%,
        #7dc4ff 50%,
        #36cfcc 90%
      );
    }
    </style>
    """
    st.markdown(gradient_css, unsafe_allow_html=True)




        
        
if __name__ == '__main__':
    main()