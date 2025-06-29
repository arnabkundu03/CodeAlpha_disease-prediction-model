# disease_prediction.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load datasets
heart_df = pd.read_csv("D:/Disease-Pred/Disease-prediction-tool/heart.csv")
diabetes_df = pd.read_csv("D:/Disease-Pred/Disease-prediction-tool/diabetes.csv")
cancer_df = pd.read_csv("D:/Disease-Pred/Disease-prediction-tool/breast_cancer.csv")

# Helper function to train and evaluate models
def train_and_evaluate_models(X, y, dataset_name):
    print(f"\n===== {dataset_name} =====")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))

# Heart Disease Prediction
X_heart = heart_df.drop("target", axis=1)
y_heart = heart_df["target"]
train_and_evaluate_models(X_heart, y_heart, "Heart Disease")

# Diabetes Prediction
X_diabetes = diabetes_df.drop("Outcome", axis=1)
y_diabetes = diabetes_df["Outcome"]
train_and_evaluate_models(X_diabetes, y_diabetes, "Diabetes")

# Breast Cancer Prediction
X_cancer = cancer_df.drop("target", axis=1)
y_cancer = cancer_df["target"]
train_and_evaluate_models(X_cancer, y_cancer, "Breast Cancer")
