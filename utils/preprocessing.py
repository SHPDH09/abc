import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Required feature columns
REQUIRED_COLUMNS = ['WorkHours', 'TasksCompleted', 'BreaksTaken', 'OvertimeHours', 'LeavesTaken', 'PerformanceRating']

def check_columns(df):
    """Check if all required columns exist in DataFrame."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return missing

def scale_data(df, scaler=None):
    """Scale input DataFrame using a provided or new scaler."""
    X = df[REQUIRED_COLUMNS]
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler

def load_scaler(path="models/scaler.pkl"):
    """Load a saved scaler object if exists."""
    try:
        return joblib.load(path)
    except:
        return None
