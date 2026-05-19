# main.py
# This is the entry point of our API
# It defines the routes (endpoints) that users can call

from fastapi import FastAPI
from app.schemas import TransactionInput, PredictionOutput
from app.model import predict_fraud

# Create the FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Detects fraudulent credit card transactions using Machine Learning",
    version="1.0.0"
)

# ── ROOT ENDPOINT ─────────────────────────────────────────
# This is just a welcome message
# Visit http://localhost:8000/ to see it
@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running",
        "docs": "Visit /docs to test the API"
    }

# ── HEALTH CHECK ENDPOINT ─────────────────────────────────
# Used to check if the API is alive
# Visit http://localhost:8000/health
@app.get("/health")
def health():
    return {"status": "healthy"}

# ── PREDICT ENDPOINT ──────────────────────────────────────
# This is the main endpoint
# Send a transaction → get fraud prediction back
# Visit http://localhost:8000/docs to test it interactively
@app.post("/predict", response_model=PredictionOutput)
def predict(transaction: TransactionInput):
    return predict_fraud(transaction)