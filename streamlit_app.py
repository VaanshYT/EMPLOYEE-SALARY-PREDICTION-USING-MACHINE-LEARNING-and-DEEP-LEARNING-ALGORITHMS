import streamlit as st
import pandas as pd
import joblib

# Load the regression model and encoders
model = joblib.load("salary_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# Load the feature scaler (if you enabled it in train_salary_model.py)
try:
    scaler = joblib.load("feature_scaler.pkl")
except FileNotFoundError:
    st.warning("Feature scaler not found. Ensure 'feature_scaler.pkl' is saved by train_salary_model.py if you intend to use scaling.")
    scaler = None # Set scaler to None if not found

# Title and form UI
st.sidebar.title("💼 Employee Info Form")
st.title("💰 Employee Annual Salary Predictor")
st.write("Enter employee details to predict their exact annual salary.")

# Input fields
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=1, value=100000)
education = st.sidebar.selectbox("Education", encoders["education"].classes_)
education_num = st.sidebar.number_input("Education-Num", min_value=1, max_value=20, value=10)
workclass = st.sidebar.selectbox("Workclass", encoders["workclass"].classes_)
marital_status = st.sidebar.selectbox("Marital Status", encoders["marital-status"].classes_)
occupation = st.sidebar.selectbox("Occupation", encoders["occupation"].classes_)
relationship = st.sidebar.selectbox("Relationship", encoders["relationship"].classes_)
race = st.sidebar.selectbox("Race", encoders["race"].classes_)
gender = st.sidebar.selectbox("Gender", encoders["gender"].classes_)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, value=0)
hours_per_week = st.sidebar.number_input("Hours per Week", min_value=1, max_value=100, value=40)
native_country = st.sidebar.selectbox("Native Country", encoders["native-country"].classes_)

# Predict button
if st.button("Predict Annual Salary"):
    # Create input dictionary with raw values
    input_dict = {
        'age': age,
        'workclass': encoders["workclass"].transform([workclass.strip()])[0],
        'fnlwgt': fnlwgt,
        'education': encoders["education"].transform([education.strip()])[0],
        'educational-num': education_num,
        'marital-status': encoders["marital-status"].transform([marital_status.strip()])[0],
        'occupation': encoders["occupation"].transform([occupation.strip()])[0],
        'relationship': encoders["relationship"].transform([relationship.strip()])[0],
        'race': encoders["race"].transform([race.strip()])[0],
        'gender': encoders["gender"].transform([gender.strip()])[0],
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': encoders["native-country"].transform([native_country.strip()])[0]
    }

    input_df = pd.DataFrame([input_dict])

    # --- Apply Feature Scaling to Numerical Columns ---
    # Ensure these column names match those used for scaling in train_salary_model.py
    numerical_columns = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    if scaler: # Only apply scaling if the scaler was loaded successfully
        # Create a copy to avoid SettingWithCopyWarning
        input_df_scaled = input_df.copy()
        input_df_scaled[numerical_columns] = scaler.transform(input_df_scaled[numerical_columns])
        # Use the scaled DataFrame for prediction
        prediction_input = input_df_scaled
    else:
        # If no scaler, use the original DataFrame (this might lead to incorrect predictions
        # if the model was trained with scaling but the scaler couldn't be loaded)
        prediction_input = input_df
    # --- End of Feature Scaling Application ---

    # Make prediction
    predicted_salary = model.predict(prediction_input)[0]

    # Display the predicted exact salary, formatted as currency
    st.success(f"💰 Predicted Annual Salary: **${predicted_salary:,.2f}**")
