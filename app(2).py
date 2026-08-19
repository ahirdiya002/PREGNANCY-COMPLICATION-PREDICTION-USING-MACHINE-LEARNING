import streamlit as st
import numpy as np
import joblib

# Page Configuration
st.set_page_config(
    page_title="Maternal Health Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# Load the saved Scikit-Learn Logistic Regression model
@st.cache_resource
def load_model():
    # Make sure this matches the filename of your downloaded pickle file
    return joblib.load("pregnancy_complication_model  (1).pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model file: {e}")
    model_loaded = False

st.title("🩺 Pregnancy Complication Risk Predictor")
st.markdown("Enter patient parameters to evaluate complication risk using the trained Logistic Regression model.")

if model_loaded:
    st.subheader("Patient Clinical Data Input")
    
    # 3-Column Layout to organize all 16 exact model features
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics & Vitals**")
        age = st.number_input("Age", min_value=10, max_value=60, value=25)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=200, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=130, value=80)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=180, value=75)
        body_temp = st.number_input("Body Temp (°F)", min_value=95.0, max_value=105.0, value=98.6)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=23.5)

    with col2:
        st.markdown("**Obstetric & Lab Measurements**")
        gestational_weeks = st.number_input("Gestational Weeks", min_value=1, max_value=42, value=20)
        trimester = st.selectbox("Pregnancy Trimester", options=[1, 2, 3], index=1)
        prev_pregnancies = st.number_input("Previous Pregnancies", min_value=0, max_value=15, value=0)
        blood_sugar = st.number_input("Blood Sugar (mmol/L)", min_value=3.0, max_value=20.0, value=5.5)
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=12.0)

    with col3:
        st.markdown("**Risk Factors & Medical History**")
        prev_comp = st.selectbox("Previous Complication", options=["No", "Yes"])
        physical_act = st.selectbox("Physical Activity", options=["Low", "Moderate", "High"])
        fam_history = st.selectbox("Family History", options=["No", "Yes"])
        diabetes = st.selectbox("Gestational Diabetes", options=["No", "Yes"])
        proteinuria = st.selectbox("Proteinuria", options=["Absent", "Present"])

    # Features formatted in exact sequence required by model:
    # ['Age', 'Systolic_BP', 'Diastolic_BP', 'Blood_Sugar', 'BMI', 'Heart_Rate', 
    #  'Body_Temperature', 'Hemoglobin', 'Pregnancy_Trimester', 'Previous_Pregnancies', 
    #  'Previous_Complication', 'Gestational_Weeks', 'Physical_Activity', 
    #  'Family_History', 'Gestational_Diabetes', 'Proteinuria']
    
    input_features = np.array([[
        age,
        systolic_bp,
        diastolic_bp,
        blood_sugar,
        bmi,
        heart_rate,
        body_temp,
        hemoglobin,
        trimester,
        prev_pregnancies,
        1 if prev_comp == "Yes" else 0,
        gestational_weeks,
        {"Low": 0, "Moderate": 1, "High": 2}[physical_act],
        1 if fam_history == "Yes" else 0,
        1 if diabetes == "Yes" else 0,
        1 if proteinuria == "Present" else 0
    ]])

    st.divider()

    # Prediction execution
    if st.button("Calculate Risk", type="primary", use_container_width=True):
        prediction = model.predict(input_features)[0]
        probabilities = model.predict_proba(input_features)[0]

        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.metric("Predicted Result", f"Class {prediction}")

        with res_col2:
            st.markdown("**Prediction Probabilities**")
            for idx, cls in enumerate(model.classes_):
                prob = probabilities[idx]
                st.write(f"Class {cls}: {prob * 100:.2f}%")
                st.progress(float(prob))