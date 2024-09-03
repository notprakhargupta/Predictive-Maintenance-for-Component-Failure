# scripts/model_evaluation.py
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Load processed telemetry data
data = pd.read_csv('data/processed_telemetry.csv')

# Split data into features and target
X = data.drop('failure', axis=1)
y = data['failure']

# Load models
rf_model = joblib.load('models/random_forest_model.pkl')
xgb_model = joblib.load('models/xgboost_model.pkl')
svc_model = joblib.load('models/svc_model.pkl')

# Evaluate models
for model in [rf_model, xgb_model, svc_model]:
    y_pred = model.predict(X)
    print(classification_report(y, y_pred))
    print(confusion_matrix(y, y_pred))
