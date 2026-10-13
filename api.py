import time
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI application
app = FastAPI(
    title="Real-Time Credit Card Fraud Detection Microservice",
    description="Asynchronous low-latency scoring API using Cost-Sensitive Gradient Boosted Trees and Cached RobustScalers",
    version="2.0.0"
)

# Load cached model, scalers, and configuration
try:
    scalers_data = joblib.load('scalers.pkl')
    scaler_amount = scalers_data['scaler_amount']
    scaler_time = scalers_data['scaler_time']
    model = joblib.load('fraud_model.pkl')
    
    with open('model_config.json', 'r') as f:
        config = json.load(f)
    optimal_threshold = config.get('optimal_threshold', 0.35)
except Exception as e:
    print(f"Warning: Artifacts still loading or training: {e}")
    model = None
    scaler_amount = None
    scaler_time = None
    optimal_threshold = 0.35

# Transaction Schema Definition
class TransactionInput(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since first transaction in dataset", example=0.0)
    Amount: float = Field(..., description="Monetary transaction amount", example=149.50)
    # V1 to V28 PCA features
    V1: float = 0.0; V2: float = 0.0; V3: float = 0.0; V4: float = 0.0; V5: float = 0.0
    V6: float = 0.0; V7: float = 0.0; V8: float = 0.0; V9: float = 0.0; V10: float = 0.0
    V11: float = 0.0; V12: float = 0.0; V13: float = 0.0; V14: float = 0.0; V15: float = 0.0
    V16: float = 0.0; V17: float = 0.0; V18: float = 0.0; V19: float = 0.0; V20: float = 0.0
    V21: float = 0.0; V22: float = 0.0; V23: float = 0.0; V24: float = 0.0; V25: float = 0.0
    V26: float = 0.0; V27: float = 0.0; V28: float = 0.0

class PredictionResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    decision_threshold: float
    risk_level: str
    scoring_latency_ms: float
    recommendation: str

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Fraud Detection Scoring Engine",
        "optimal_threshold": optimal_threshold,
        "sla_target": "< 20ms"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_transaction(tx: TransactionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline artifacts are not yet initialized.")
    
    t_start = time.time()
    tx_dict = tx.model_dump()
    df_single = pd.DataFrame([tx_dict])
    
    # Scale Amount and Time using cached scalers (Zero data leakage)
    df_single['Amount'] = scaler_amount.transform(df_single[['Amount']])
    df_single['Time'] = scaler_time.transform(df_single[['Time']])
    
    # Model inference
    proba = float(model.predict_proba(df_single)[0, 1])
    is_fraud = proba >= optimal_threshold
    latency_ms = (time.time() - t_start) * 1000
    
    if proba >= 0.75:
        risk_level = "CRITICAL_THREAT"
        recommendation = "Decline immediately and issue biometric step-up authentication challenge."
    elif proba >= optimal_threshold:
        risk_level = "ELEVATED_RISK"
        recommendation = "Hold for automated SMS verification OTP; risk exceeds economic threshold."
    else:
        risk_level = "LOW_RISK"
        recommendation = "Seamless authorization approval."

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(proba, 4),
        decision_threshold=optimal_threshold,
        risk_level=risk_level,
        scoring_latency_ms=round(latency_ms, 3),
        recommendation=recommendation
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
