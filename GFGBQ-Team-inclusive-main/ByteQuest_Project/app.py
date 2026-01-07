import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Page Configuration ---
st.set_page_config(page_title="ByteQuest Disease Predictor", page_icon="🏥", layout="wide")

# --- Load Model & Data ---
@st.cache_resource # This keeps the model in memory so it doesn't reload on every click
def load_assets():
    model = joblib.load('medical_rf_model.pkl')
    symptoms_list = joblib.load('symptom_features_list.pkl')
    # Load description data for the "About Disease" section
    description = pd.read_csv('data_sets/symptom_Description.csv')
    return model, symptoms_list, description

try:
    model, symptoms_list, description_df = load_assets()
except Exception as e:
    st.error(f"Error loading files. Ensure they are in the correct folder! {e}")
    st.stop()

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/387/387561.png", width=100)
st.sidebar.title("Diagnosis Panel")
st.sidebar.info("Select the symptoms you are experiencing to see the most likely diagnosis.")

# --- Main UI ---
st.title("🏥 Medical Diagnosis Support System")
st.write("A Machine Learning tool for predicting diseases based on symptoms.")

# 1. Input Section
selected_symptoms = st.multiselect(
    "Search and select symptoms:",
    options=sorted(symptoms_list),
    help="You can select multiple symptoms from the list."
)

if st.button("Generate Diagnosis"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # 2. Processing
        input_vector = np.zeros(len(symptoms_list))
        for s in selected_symptoms:
            idx = symptoms_list.index(s)
            input_vector[idx] = 1
        
        # 3. Prediction
        probs = model.predict_proba(input_vector.reshape(1, -1))[0]
        
        # Create a DataFrame for all results
        results_df = pd.DataFrame({
            'Disease': model.classes_,
            'Probability': probs * 100
        }).sort_values(by='Probability', ascending=False)

        # Get Top Result
        top_disease = results_df.iloc[0]['Disease']
        top_conf = results_df.iloc[0]['Probability']

        # 4. Display Results
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Results")
            if top_conf > 0:
                st.success(f"**Primary Diagnosis:** {top_disease}")
                st.metric(label="Confidence Level", value=f"{top_conf:.2f}%")
                
                # Show description from CSV
                desc = description_df[description_df['Disease'] == top_disease]['Description']
                if not desc.empty:
                    st.info(f"**About {top_disease}:** \n{desc.values[0]}")
            else:
                st.error("The model could not confidently identify a disease. Please add more symptoms.")

        with col2:
            st.subheader("Probability Distribution")
            # Show only top 5 for the chart
            chart_data = results_df.head(5).set_index('Disease')
            st.bar_chart(chart_data)

# --- Footer ---
st.divider()
st.caption("Disclaimer: This is a student project for educational purposes. Consult a doctor for medical advice.")