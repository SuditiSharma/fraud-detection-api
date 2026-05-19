# model.py
# This file loads the saved model and makes predictions

import joblib
import numpy as np
from app.schemas import TransactionInput, PredictionOutput

# Load the model once when the app starts
# We don't want to reload it every time someone makes a request
model = joblib.load("model/fraud_model.pkl")

def predict_fraud(transaction: TransactionInput) -> PredictionOutput:
    # Convert the incoming transaction into a format the model understands
    # The model expects a 2D array — hence the double brackets [[...]]
    features = np.array([[
        transaction.Time,
        transaction.V1, transaction.V2, transaction.V3,
        transaction.V4, transaction.V5, transaction.V6,
        transaction.V7, transaction.V8, transaction.V9,
        transaction.V10, transaction.V11, transaction.V12,
        transaction.V13, transaction.V14, transaction.V15,
        transaction.V16, transaction.V17, transaction.V18,
        transaction.V19, transaction.V20, transaction.V21,
        transaction.V22, transaction.V23, transaction.V24,
        transaction.V25, transaction.V26, transaction.V27,
        transaction.V28, transaction.Amount
    ]])

    # Get prediction — 0 or 1
    prediction = model.predict(features)[0]

    # Get probability — how confident is the model?
    # predict_proba returns [[prob_legitimate, prob_fraud]]
    probability_fraud = model.predict_proba(features)[0][1]

    # Build a human readable message
    if prediction == 1:
        message = f"⚠️ FRAUD DETECTED — {round(probability_fraud * 100, 2)}% confidence"
    else:
        message = f"✅ Legitimate transaction — {round((1 - probability_fraud) * 100, 2)}% confidence"

    return PredictionOutput(
        prediction=int(prediction),
        probability_fraud=round(float(probability_fraud), 4),
        message=message
    )