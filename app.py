import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler

# Page Config
st.set_page_config(page_title="Burnout Risk Analyzer", layout="wide")
st.title("🧠 AI-Based Workforce Productivity & Burnout Analyzer")
st.markdown("Upload employee data and let AI detect potential burnout risks.")

# Sidebar
st.sidebar.header("⚙️ Upload Settings")
uploaded_file = st.sidebar.file_uploader("📁 Upload Employee Data (.csv)", type=["csv"])

# Show example data
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

# Load ML Model
try:
    model = joblib.load("models/burnout_model.pkl")
    model_loaded = True
except:
    st.error("❌ burnout_model.pkl not found in models folder.")
    model_loaded = False

# Optional: Load scaler (if used during training)
try:
    scaler = joblib.load("models/scaler.pkl")
except:
    scaler = StandardScaler()

# Main Logic
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Employee Data")
    st.dataframe(df)

    # Required features
    features = ['WorkHours', 'TasksCompleted', 'BreaksTaken', 'OvertimeHours', 'LeavesTaken', 'PerformanceRating']
    
    if all(col in df.columns for col in features):
        X = df[features]

        # Scale
        X_scaled = scaler.transform(X)

        if model_loaded:
            # Predict Burnout
            predictions = model.predict(X_scaled)
            df['BurnoutRisk'] = ['High' if p == 1 else 'Low' for p in predictions]

            st.subheader("🔍 Prediction Results")
            st.dataframe(df[['EmployeeID'] + features + ['BurnoutRisk']])

            # Chart 1: Burnout Distribution
            st.subheader("🔥 Burnout Risk Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(x='BurnoutRisk', data=df, palette='coolwarm', ax=ax1)
            ax1.set_title("Burnout Risk Count")
            st.pyplot(fig1)

            # Chart 2: Correlation Heatmap
            st.subheader("📈 Feature Correlation Heatmap")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.heatmap(df[features].corr(), annot=True, cmap="YlGnBu", ax=ax2)
            st.pyplot(fig2)

        else:
            st.error("⚠️ Model not loaded. Check the `models/` folder for burnout_model.pkl.")
    else:
        st.error(f"❌ Missing required columns: {features}")
else:
    st.info("👈 Upload a CSV file to begin analysis.")

# Footer
st.markdown("---")
st.markdown("📌 Developed by **Raunak Kumar** | LNCTU - BCA (AIDA)")
