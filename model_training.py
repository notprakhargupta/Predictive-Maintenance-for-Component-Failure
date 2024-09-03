# scripts/model_training.py
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Load processed telemetry data
data = pd.read_csv('data/processed_telemetry.csv')

# Split data into features and target
X = data.drop('failure', axis=1)
y = data['failure']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
# 1. Random Forest Classifier
rf = RandomForestClassifier()
rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 20]}
rf_search = RandomizedSearchCV(rf, rf_params, cv=5, scoring='recall')
rf_search.fit(X_train, y_train)

# 2. XGBoost Classifier
xgb = XGBClassifier()
xgb_params = {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1]}
xgb_search = RandomizedSearchCV(xgb, xgb_params, cv=5, scoring='recall')
xgb_search.fit(X_train, y_train)

# 3. Linear SVC
svc = SVC(kernel='linear', probability=True)
svc.fit(X_train, y_train)

# Save the trained models
import joblib
joblib.dump(rf_search.best_estimator_, 'models/random_forest_model.pkl')
joblib.dump(xgb_search.best_estimator_, 'models/xgboost_model.pkl')
joblib.dump(svc, 'models/svc_model.pkl')
