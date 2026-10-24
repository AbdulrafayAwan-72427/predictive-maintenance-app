# Author identifier: 72427_abdulrafay
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Data (Ensure 'ai4i2020.csv' is in your 'data/' folder)
df = pd.read_csv("data/ai4i2020.csv")

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
X = df[features]
y = df["Machine failure"]

# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Apply SMOTE to training split
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled, y_train
)

# 5. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 6. Extract Probabilities & Optimize Threshold via Precision-Recall Curve
y_scores = model.predict_proba(X_test_scaled)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
optimal_threshold = thresholds[np.argmax(f1_scores)]

print(f"Optimal Threshold: {optimal_threshold:.4f}")

# 7. Evaluate
custom_preds = (y_scores >= optimal_threshold).astype(int)
print(classification_report(y_test, custom_preds))

# 8. Save Artifacts
joblib.dump(model, "maintenance_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(optimal_threshold, "optimal_threshold.pkl")
print("Artifacts saved successfully.")