import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load('financial_inclusion_model.pkl')

st.set_page_config(page_title="Financial Inclusion Prediction", layout="centered")

st.title("Financial Inclusion Predictor")
st.write("Predict if an individual is likely to have a **bank account** based on socio-economic details.")

st.markdown("----")

# Feature inputs
location = st.selectbox("Location Type", ["Rural", "Urban"])
cellphone = st.selectbox("Cellphone Access", ["No", "Yes"])
gender = st.selectbox("Gender", ["Female", "Male"])
age = st.slider("Age of Respondent", 18, 100, 30)
household_size = st.number_input("Household Size", min_value=1, max_value=30, value=5)

# Job Type Encoding (as in training with one-hot, drop_first=True)
st.markdown("### Job Type")
job_types = [
    "Farming and Fishing",
    "Formally employed Government",
    "Formally employed Private",
    "Informally employed",
    "Other Income",
    "Remittance Dependent",
    "Self employed"
]

selected_job = st.selectbox("Select your job type", job_types)

# One-hot encode job types (manually)
job_dict = {f"job_type_{job}": 0 for job in job_types if job != "Remittance Dependent"}  # dropped in training
if selected_job != "Remittance Dependent":
    job_dict[f"job_type_{selected_job}"] = 1

# Prepare data for prediction
if st.button("Predict"):
    # Map inputs to model features
    input_data = pd.DataFrame([{
        "location_type": 1 if location == "Urban" else 0,
        "cellphone_access": 1 if cellphone == "Yes" else 0,
        "age_of_respondent": age,
        "household_size": household_size,
        "gender_of_respondent": 1 if gender == "Male" else 0,
        **job_dict
    }])

    # Ensure all expected columns are in place
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0  # fill missing one-hot columns with 0

    # Align column order
    input_data = input_data[model_features]

    # Make prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ The individual is **likely** to have a bank account.")
    else:
        st.warning("❌ The individual is **unlikely** to have a bank account.")
