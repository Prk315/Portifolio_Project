"""
Student Dropout Prediction Dashboard
A Streamlit app for predicting student outcomes
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Student Dropout Predictor",
    page_icon="🎓",
    layout="wide"
)

# Load models and preprocessors
@st.cache_resource
def load_models():
    """Load all necessary artifacts for prediction"""
    try:
        model = joblib.load('data/best_model_final.joblib')
    except:
        model = joblib.load('data/rf_classifier.joblib')

    preprocessor_fe = joblib.load('data/preprocessor_fe.joblib')
    selector = joblib.load('data/selector.joblib')
    le = joblib.load('data/label_encoder.joblib')

    return model, preprocessor_fe, selector, le

model, preprocessor_fe, selector, le = load_models()

# App header
st.title("🎓 Student Dropout Prediction System")
st.markdown("""
This tool predicts whether a student will **Graduate**, **Dropout**, or remain **Enrolled**
based on their academic and demographic information.
""")

st.divider()

# Sidebar for input method selection
st.sidebar.header("Input Method")
input_method = st.sidebar.radio(
    "Choose how to input data:",
    ["Manual Entry", "Upload CSV"]
)

def engineer_features(df):
    """Apply the same feature engineering as training"""
    df_new = df.copy()

    # Academic performance aggregates
    df_new['total_approved'] = (
        df['Curricular units 1st sem (approved)'] +
        df['Curricular units 2nd sem (approved)']
    )

    df_new['total_enrolled'] = (
        df['Curricular units 1st sem (enrolled)'] +
        df['Curricular units 2nd sem (enrolled)']
    )

    # Approval rate
    df_new['approval_rate'] = np.where(
        df_new['total_enrolled'] > 0,
        df_new['total_approved'] / df_new['total_enrolled'],
        0
    )

    # Grade average
    df_new['avg_grade'] = (
        df['Curricular units 1st sem (grade)'] +
        df['Curricular units 2nd sem (grade)']
    ) / 2

    # Semester improvement
    df_new['grade_improvement'] = (
        df['Curricular units 2nd sem (grade)'] -
        df['Curricular units 1st sem (grade)']
    )

    df_new['approval_improvement'] = (
        df['Curricular units 2nd sem (approved)'] -
        df['Curricular units 1st sem (approved)']
    )

    # Credited units
    df_new['total_credited'] = (
        df['Curricular units 1st sem (credited)'] +
        df['Curricular units 2nd sem (credited)']
    )

    # Without evaluations
    df_new['total_without_eval'] = (
        df['Curricular units 1st sem (without evaluations)'] +
        df['Curricular units 2nd sem (without evaluations)']
    )

    # Age group
    df_new['age_group'] = pd.cut(
        df['Age at enrollment'],
        bins=[0, 20, 25, 30, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    return df_new

def predict_outcome(student_data):
    """Make prediction for a single student"""
    # Apply feature engineering
    student_fe = engineer_features(student_data)

    # Transform using preprocessor
    X_transformed = preprocessor_fe.transform(student_fe)

    # Apply feature selection
    X_selected = selector.transform(X_transformed)

    # Predict
    prediction = model.predict(X_selected)[0]
    probabilities = model.predict_proba(X_selected)[0]

    return le.classes_[prediction], probabilities

# Manual Entry Mode
if input_method == "Manual Entry":
    st.header("Enter Student Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        age = st.number_input("Age at enrollment", min_value=17, max_value=70, value=20)
        gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        marital_status = st.selectbox("Marital Status", list(range(1, 7)))
        nationality = st.number_input("Nationality Code", min_value=1, value=1)

    with col2:
        st.subheader("Academic Background")
        admission_grade = st.slider("Admission Grade", 0.0, 200.0, 120.0)
        prev_qual = st.selectbox("Previous Qualification", list(range(1, 18)))
        prev_grade = st.slider("Previous Qualification Grade", 0.0, 200.0, 120.0)
        application_mode = st.selectbox("Application Mode", list(range(1, 19)))
        application_order = st.selectbox("Application Order", list(range(0, 10)))
        course = st.selectbox("Course", list(range(1, 18)))

    with col3:
        st.subheader("Status & Support")
        scholarship = st.selectbox("Scholarship Holder", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        tuition_uptodate = st.selectbox("Tuition Fees Up to Date", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        debtor = st.selectbox("Debtor", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        displaced = st.selectbox("Displaced", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        international = st.selectbox("International Student", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        special_needs = st.selectbox("Educational Special Needs", [0, 1], format_func=lambda x: "No" if x==0 else "Yes")

    st.subheader("1st Semester Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sem1_credited = st.number_input("1st Sem Credited", min_value=0, value=0)
        sem1_enrolled = st.number_input("1st Sem Enrolled", min_value=0, value=6)
    with col2:
        sem1_evaluations = st.number_input("1st Sem Evaluations", min_value=0, value=6)
        sem1_approved = st.number_input("1st Sem Approved", min_value=0, value=5)
    with col3:
        sem1_grade = st.slider("1st Sem Grade", 0.0, 20.0, 12.0)
        sem1_without_eval = st.number_input("1st Sem Without Evaluations", min_value=0, value=0)

    st.subheader("2nd Semester Performance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sem2_credited = st.number_input("2nd Sem Credited", min_value=0, value=0)
        sem2_enrolled = st.number_input("2nd Sem Enrolled", min_value=0, value=6)
    with col2:
        sem2_evaluations = st.number_input("2nd Sem Evaluations", min_value=0, value=6)
        sem2_approved = st.number_input("2nd Sem Approved", min_value=0, value=5)
    with col3:
        sem2_grade = st.slider("2nd Sem Grade", 0.0, 20.0, 12.5)
        sem2_without_eval = st.number_input("2nd Sem Without Evaluations", min_value=0, value=0)

    st.subheader("Family Background")
    col1, col2 = st.columns(2)
    with col1:
        mothers_qual = st.selectbox("Mother's Qualification", list(range(1, 35)))
        mothers_occ = st.selectbox("Mother's Occupation", list(range(0, 40)))
    with col2:
        fathers_qual = st.selectbox("Father's Qualification", list(range(1, 35)))
        fathers_occ = st.selectbox("Father's Occupation", list(range(0, 50)))

    st.subheader("Economic Indicators")
    col1, col2, col3 = st.columns(3)
    with col1:
        unemployment_rate = st.slider("Unemployment Rate", 0.0, 30.0, 10.0)
    with col2:
        inflation_rate = st.slider("Inflation Rate", -5.0, 10.0, 1.5)
    with col3:
        gdp = st.slider("GDP", -10.0, 10.0, 1.0)

    daytime = st.selectbox("Daytime/Evening Attendance", [0, 1], format_func=lambda x: "Evening" if x==0 else "Daytime")

    if st.button("Predict Outcome", type="primary"):
        # Create dataframe with correct column order
        student_dict = {
            'Marital status': marital_status,
            'Application mode': application_mode,
            'Application order': application_order,
            'Course': course,
            'Daytime/evening attendance\t': daytime,
            'Previous qualification': prev_qual,
            'Previous qualification (grade)': prev_grade,
            'Nacionality': nationality,
            "Mother's qualification": mothers_qual,
            "Father's qualification": fathers_qual,
            "Mother's occupation": mothers_occ,
            "Father's occupation": fathers_occ,
            'Admission grade': admission_grade,
            'Displaced': displaced,
            'Educational special needs': special_needs,
            'Debtor': debtor,
            'Tuition fees up to date': tuition_uptodate,
            'Gender': gender,
            'Scholarship holder': scholarship,
            'Age at enrollment': age,
            'International': international,
            'Curricular units 1st sem (credited)': sem1_credited,
            'Curricular units 1st sem (enrolled)': sem1_enrolled,
            'Curricular units 1st sem (evaluations)': sem1_evaluations,
            'Curricular units 1st sem (approved)': sem1_approved,
            'Curricular units 1st sem (grade)': sem1_grade,
            'Curricular units 1st sem (without evaluations)': sem1_without_eval,
            'Curricular units 2nd sem (credited)': sem2_credited,
            'Curricular units 2nd sem (enrolled)': sem2_enrolled,
            'Curricular units 2nd sem (evaluations)': sem2_evaluations,
            'Curricular units 2nd sem (approved)': sem2_approved,
            'Curricular units 2nd sem (grade)': sem2_grade,
            'Curricular units 2nd sem (without evaluations)': sem2_without_eval,
            'Unemployment rate': unemployment_rate,
            'Inflation rate': inflation_rate,
            'GDP': gdp
        }

        student_df = pd.DataFrame([student_dict])

        # Make prediction
        outcome, probabilities = predict_outcome(student_df)

        # Display results
        st.divider()
        st.header("Prediction Results")

        col1, col2 = st.columns([1, 2])

        with col1:
            if outcome == "Graduate":
                st.success(f"### Predicted Outcome: {outcome} 🎓")
            elif outcome == "Dropout":
                st.error(f"### Predicted Outcome: {outcome} ⚠️")
            else:
                st.warning(f"### Predicted Outcome: {outcome} 📚")

        with col2:
            st.subheader("Confidence Levels")
            for i, class_name in enumerate(le.classes_):
                st.progress(probabilities[i], text=f"{class_name}: {probabilities[i]*100:.1f}%")

# CSV Upload Mode
else:
    st.header("Upload Student Data (CSV)")
    st.markdown("Upload a CSV file with student information. The file should have the same format as the training data.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';' if ';' in str(uploaded_file.read(100)) else ',')
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)

        st.write(f"Loaded {len(df)} students")
        st.dataframe(df.head())

        if st.button("Predict for All Students"):
            results = []
            for idx, row in df.iterrows():
                student_df = pd.DataFrame([row])
                outcome, probabilities = predict_outcome(student_df)
                results.append({
                    'Student_ID': idx,
                    'Predicted_Outcome': outcome,
                    'Dropout_Probability': probabilities[0],
                    'Enrolled_Probability': probabilities[1],
                    'Graduate_Probability': probabilities[2]
                })

            results_df = pd.DataFrame(results)

            st.success("Predictions complete!")
            st.dataframe(results_df)

            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
Built with Streamlit • Model: Random Forest / Gradient Boosting • Accuracy: ~77%
</div>
""", unsafe_allow_html=True)
