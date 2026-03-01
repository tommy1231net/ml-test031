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
        # Check current directory and files
        current_dir = os.getcwd()
        files_in_dir = os.listdir(current_dir)
        print(f"DIAGNOSTIC: Current directory: {current_dir}")
        print(f"DIAGNOSTIC: Files found: {files_in_dir}")

        if os.path.exists(MODEL_FILE):
            # Load XGBoost model
            model = xgb.XGBClassifier()
            model.load_model(MODEL_FILE)
            
            # Load Mapping
            with open(MAPPING_FILE, 'r') as f:
                label_mapping = json.load(f)
            inv_label_mapping = {int(v): k for k, v in label_mapping.items()}
            print(f"SUCCESS: Model and mapping loaded correctly.")
        else:
            print(f"CRITICAL ERROR: {MODEL_FILE} NOT FOUND. Did you include it in your Docker build?")
            
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
# --- 8. Predict Endpoint with Correct Error Handling and Shape Checks ---
@app.post("/predict")
async def predict(data: TaxiTripInput):
    if model is None:
        return {"error": "Model is not loaded on server. Check logs."}
    
    try: # <--- This starts the try block
        # 1. Prepare data
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # 2. Feature Engineering
        ts = pd.to_datetime(df['trip_start_timestamp'], errors='coerce')
        if ts.isna().any():
            return {"error": "Invalid timestamp format"}

        df['hour'] = ts.dt.hour
        df['dayofweek'] = ts.dt.dayofweek
        
        features = [
            'trip_seconds', 'trip_miles', 'fare', 'extras', 'tolls', 
            'hour', 'dayofweek', 'pickup_area', 'dropoff_area', 'company_id'
        ]
        X = df[features]
        
        # 3. Native Booster Prediction
        dmatrix = xgb.DMatrix(X)
        preds = model.get_booster().predict(dmatrix)
        
        # 4. Handle 1D vs 2D array shapes (Fix for axis 1 error)
        if len(preds.shape) > 1:
            # Multi-sample or standard multi-class (2D)
            prediction_idx = int(np.argmax(preds, axis=1)[0])
            probabilities = preds[0].tolist()
        else:
            # Single sample flat output (1D)
            prediction_idx = int(np.argmax(preds))
            probabilities = preds.tolist()
        
        prediction_label = inv_label_mapping.get(prediction_idx, "Unknown")
        
        # 5. Format Confidence Scores
        prob_dict = {
            inv_label_mapping[i]: round(float(p), 4) 
            for i, p in enumerate(probabilities)
        }
        
        return {
            "prediction": prediction_label,
            "confidence_scores": prob_dict
        }

    except Exception as e: # <--- Make sure this 'except' aligns with 'try'
        import traceback
        print(f"RUNTIME ERROR: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e), "detail": "Internal Server Error"}

if __name__ == "__main__":
    import uvicorn
    # Use environment variable PORT or default to 8080
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)