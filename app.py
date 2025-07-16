import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler

# Load ML model
try:
    model = joblib.load("models/productivity_model.pkl")
except:
    model = None

# Title
st.set_page_config(page_title="Workforce Productivity Analyzer", layout="wide")
st.title("🧠 AI-Based Workforce Productivity & Performance Analyzer")
st.markdown("Upload employee data and analyze productivity, performance, and burnout risks.")

# Sidebar
st.sidebar.header("⚙️ Upload & Settings")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Employee Data (.csv)", type=['csv'])

# Show sample format
if st.sidebar.checkbox("Show sample format"):
    sample_df = pd.DataFrame({
        'EmployeeID': ['E001', 'E002'],
        'WorkHours': [45, 38],
        'TasksCompleted': [20, 12],
        'BreaksTaken': [2, 5],
        'OvertimeHours': [5, 0],
        'LeavesTaken': [0, 2],
        'PerformanceRating': [4.2, 3.1]
    })
    st.sidebar.write(sample_df)

# Main logic
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(df)

    # Feature Selection
    features = ['WorkHours', 'TasksCompleted', 'BreaksTaken', 'OvertimeHours', 'LeavesTaken', 'PerformanceRating']
    if all(col in df.columns for col in features):
        X = df[features]

        # Preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Prediction
        if model:
            predictions = model.predict(X_scaled)
            df['ProductivityLevel'] = predictions
            df['BurnoutRisk'] = ['High' if x < 1 else 'Low' for x in predictions]

            st.subheader("🔍 Prediction Results")
            st.dataframe(df[['EmployeeID'] + features + ['ProductivityLevel', 'BurnoutRisk']])

            # Visualizations
            st.subheader("📈 Productivity Level Distribution")
            fig, ax = plt.subplots()
            sns.countplot(data=df, x='ProductivityLevel', palette='Set2', ax=ax)
            st.pyplot(fig)

            st.subheader("🔥 Burnout Risk Analysis")
            burnout_counts = df['BurnoutRisk'].value_counts()
            st.bar_chart(burnout_counts)

        else:
            st.error("⚠️ ML model not found. Please ensure `productivity_model.pkl` exists in the `models/` folder.")
    else:
        st.error("Missing required columns in uploaded file.")
else:
    st.info("👈 Please upload a CSV file to start analysis.")

# Footer
st.markdown("---")
st.markdown("📌 Developed by Raunak Kumar | LNCTU - BCA (AIDA)")
