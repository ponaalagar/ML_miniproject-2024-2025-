import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load encoders
Jobencoder = LabelEncoder()
Eduencoder = LabelEncoder()
GenderEn = LabelEncoder()

def main():
    st.title("Income Prediction App")
    st.write("This app predicts your income based on your details.")

    # Get user input
    age = st.number_input("Enter your age", min_value=0, max_value=100, step=1)
    gender = st.selectbox("Select your gender", ["Male", "Female", "Other"])
    education_level = st.selectbox("Select your education level", ["High School", "Bachelor's", "Master's", "PhD"])
    job_title = st.text_input("Enter your job title")
    years_experience = st.number_input("Enter your years of experience", min_value=0.0, step=0.1)

    # Prepare input for prediction
    if st.button("Predict Income"):
        # Encode input
        gender_encoded = GenderEn.fit_transform([gender])[0]
        education_encoded = Eduencoder.fit_transform([education_level])[0]
        job_encoded = Jobencoder.fit_transform([job_title])[0]

        # Create a DataFrame
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender_encoded],
            'Education Level': [education_encoded],
            'Job Title': [job_encoded],
            'Years of Experience': [years_experience]
        })

        # Make prediction
        try:
            salary_prediction = model.predict(input_data)[0]
            st.success(f"Predicted Income: ${salary_prediction:.2f}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

if __name__ == "__main__":
    main()
