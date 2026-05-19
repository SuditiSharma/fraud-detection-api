# schemas.py
# This file defines the shape of data coming IN and going OUT of our API

from pydantic import BaseModel

# This is what the API expects to receive
class TransactionInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# This is what the API sends back
class PredictionOutput(BaseModel):
    prediction: int        # 0 = legitimate, 1 = fraud
    probability_fraud: float   # e.g. 0.87 means 87% chance of fraud
    message: str           # human readable result