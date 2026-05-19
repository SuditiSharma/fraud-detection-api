# train.py
# This script loads the data, trains the model, and saves it

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os

# ── 1. LOAD DATA ──────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("data/creditcard.csv")

print(f"Dataset shape: {df.shape}")
print(f"Fraud cases: {df['Class'].sum()}")
print(f"Legitimate cases: {(df['Class'] == 0).sum()}")

# ── 2. PREPARE DATA ───────────────────────────────────────
# X = features (everything except Class)
# y = target (Class: 0=legitimate, 1=fraud)
X = df.drop("Class", axis=1)
y = df["Class"]

# Split into training and testing sets
# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# ── 3. HANDLE IMBALANCED DATA WITH SMOTE ──────────────────
# Problem: only 0.17% of transactions are fraud
# SMOTE creates synthetic fraud examples so model learns better
print("\nApplying SMOTE to handle class imbalance...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Fraud cases: {y_train_balanced.sum()}")
print(f"After SMOTE - Legitimate cases: {(y_train_balanced == 0).sum()}")

# ── 4. TRAIN MODEL ────────────────────────────────────────
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,    # 100 decision trees
    random_state=42,     # for reproducibility
    n_jobs=-1            # use all CPU cores
)
model.fit(X_train_balanced, y_train_balanced)

# ── 5. EVALUATE MODEL ─────────────────────────────────────
print("\nEvaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# ── 6. SAVE MODEL ─────────────────────────────────────────
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/fraud_model.pkl")
print("\nModel saved to model/fraud_model.pkl")