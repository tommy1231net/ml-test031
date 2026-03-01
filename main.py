from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import json
import traceback

# 1. Configuration
MODEL_FILE = 'taxi_payment_model.json'
MAPPING_FILE = 'label_mapping.json'

app = FastAPI()

# 2. CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
inv_label_mapping = {}

# 3. Model Loading at Startup
# Using @app.on_event("startup") ensures the server starts even if loading fails initially
@app.on_event("startup")
async def load_model():
    global model, inv_label_mapping
    try:
        if os.path.exists(MODEL_FILE):
            # Using XGBClassifier but loading via native booster for stability
            model = xgb.XGBClassifier()
            model.load_model(MODEL_FILE)
            
            with open(MAPPING_FILE, 'r') as f:
                label_mapping = json.load(f)
            inv_label_mapping = {int(v): k for k, v in label_mapping.items()}
            print(f"SUCCESS: Model and mapping loaded.")
        else:
            print(f"CRITICAL ERROR: {MODEL_FILE} not found in /app directory.")
    except Exception as e:
        print(f"STARTUP ERROR: {str(e)}")
        print(traceback.format_exc())

# 4. Request Schema
class TaxiTripInput(BaseModel):
    trip_start_timestamp: str
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
    # If model is None, we know it failed to load
    model_status = "Loaded" if model is not None else "Failed to Load"
    return {"status": "API is active", "model_status": model_status}

@app.post("/predict")
async def predict(data: TaxiTripInput):
    if model is None:
        return {"error": "Model is not loaded on server. Check logs."}
    
    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Feature Engineering
        ts = pd.to_datetime(df['trip_start_timestamp'], errors='coerce')
        df['hour'] = ts.dt.hour
        df['dayofweek'] = ts.dt.dayofweek
        
        features = ['trip_seconds', 'trip_miles', 'fare', 'extras', 'tolls', 'hour', 'dayofweek', 'pickup_area', 'dropoff_area', 'company_id']
        X = df[features]
        
        # Prediction via Booster to avoid feature mismatch
        dmatrix = xgb.DMatrix(X)
        preds = model.get_booster().predict(dmatrix)
        
        prediction_idx = int(np.argmax(preds, axis=1)[0])
        prediction_label = inv_label_mapping.get(prediction_idx, "Unknown")
        
        prob_dict = {inv_label_mapping[i]: round(float(p), 4) for i, p in enumerate(preds[0])}
        
        return {"prediction": prediction_label, "confidence_scores": prob_dict}
    except Exception as e:
        print(f"RUNTIME ERROR: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Use environment variable PORT or default to 8080
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)