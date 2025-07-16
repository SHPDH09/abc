import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Load data
df = pd.read_csv("employee_data.csv")
X = df[['WorkHours', 'TasksCompleted', 'BreaksTaken', 'OvertimeHours', 'LeavesTaken', 'PerformanceRating']]
y = ((df['WorkHours'] > 45) & (df['BreaksTaken'] < 2)).astype(int)

# Train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_scaled, y)

# Save
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/burnout_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
