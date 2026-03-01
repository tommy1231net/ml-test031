from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# 1. Configuration: Model file settings
MODEL_FILE = 'taxi_payment_model.json'
MAPPING_FILE = 'label_mapping.json'

app = FastAPI()

# 2. CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 3. Model & Mapping Ingestion
try:
    # Load XGBoost model
    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)
    
    # Load Label Mapping (0: Credit Card, etc.)
    with open(MAPPING_FILE, 'r') as f:
        label_mapping = json.load(f)
    # Invert mapping to get: {0: "Credit Card", ...}
    inv_label_mapping = {int(v): k for k, v in label_mapping.items()}
    
    print(f"SUCCESS: Loaded {MODEL_FILE} and {MAPPING_FILE}")
except Exception as e:
    print(f"ERROR: Failed to load model files: {e}")

# 4. Request Schema (Features required for prediction)
class TaxiTripInput(BaseModel):
    trip_start_timestamp: str # Format: "2026-03-01 10:30:00"
    trip_seconds: float
    trip_miles: float
    fare: float
    extras: float
    tolls: float
    pickup_area: int
    dropoff_area: int
    company_id: int

@app.get("/")
def read_root():
    return {"status": "Chicago Taxi Payment Prediction API is active", "model": MODEL_FILE}

@app.post("/predict")
def predict(data: TaxiTripInput):
    # Convert input to DataFrame
    input_dict = data.dict()
    df = pd.DataFrame([input_dict])
    
    # 5. Feature Engineering (Same logic as training)
    # Convert string to datetime and extract time features
    ts = pd.to_datetime(df['trip_start_timestamp'])
    df['hour'] = ts.dt.hour
    df['dayofweek'] = ts.dt.dayofweek
    
    # Define exact feature order used during training
    features = [
        'trip_seconds', 'trip_miles', 'fare', 'extras', 'tolls', 
        'hour', 'dayofweek', 'pickup_area', 'dropoff_area', 'company_id'
    ]
    
    # Select and reorder columns
    X = df[features]
    
    # 6. Execute prediction
    prediction_idx = int(model.predict(X)[0])
    prediction_label = inv_label_mapping.get(prediction_idx, "Unknown")
    
    # Get probability for each class (optional but professional)
    probabilities = model.predict_proba(X)[0].tolist()
    prob_dict = {inv_label_mapping[i]: round(p, 4) for i, p in enumerate(probabilities)}
    
    return {
        "prediction": prediction_label,
        "confidence_scores": prob_dict
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)