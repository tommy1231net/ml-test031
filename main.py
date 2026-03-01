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
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILE)
        
        with open(MAPPING_FILE, 'r') as f:
            label_mapping = json.load(f)
    
        # Ensure keys are integers: {0: "Credit Card", 1: "Cash", ...}
        inv_label_mapping = {int(k): v for k, v in label_mapping.items()}
        
        print(f"SUCCESS: Loaded mapping: {inv_label_mapping}")
    except Exception as e:
        print(f"ERROR: {str(e)}")

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
# --- 8. Predict Endpoint with Improved Label Mapping ---
@app.post("/predict")
async def predict(data: TaxiTripInput):
    if model is None:
        return {"error": "Model is not loaded."}
    
    try:
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Feature Engineering (Keep same order as training)
        ts = pd.to_datetime(df['trip_start_timestamp'], errors='coerce')
        df['hour'] = ts.dt.hour
        df['dayofweek'] = ts.dt.dayofweek
        
        features = ['trip_seconds', 'trip_miles', 'fare', 'extras', 'tolls', 'hour', 'dayofweek', 'pickup_area', 'dropoff_area', 'company_id']
        X = df[features]
        
        # Prediction
        dmatrix = xgb.DMatrix(X)
        preds = model.get_booster().predict(dmatrix) # Returns probabilities [p0, p1, p2, p3]
        
        # Flatten preds if it's 2D [[p0, p1, p2, p3]]
        probabilities = preds[0].tolist() if len(preds.shape) > 1 else preds.tolist()
        
        # Get the index of the highest probability
        prediction_idx = int(np.argmax(probabilities))
        
        # Build the result dictionary carefully
        # We ensure that index i matches the label from mapping
        prob_dict = {}
        for i, prob in enumerate(probabilities):
            # Try to get label from mapping, fallback to index string if not found
            label = inv_label_mapping.get(i, f"Class_{i}")
            prob_dict[label] = round(float(prob), 4)

        # The final prediction is the label with the highest score
        prediction_label = inv_label_mapping.get(prediction_idx, "Unknown")
        
        # Debugging: Log the raw index and scores to Cloud Run console
        print(f"DEBUG: idx={prediction_idx}, label={prediction_label}, scores={prob_dict}")

        return {
            "prediction": prediction_label,
            "confidence_scores": prob_dict
        }

    except Exception as e:
        import traceback
        print(f"RUNTIME ERROR: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Use environment variable PORT or default to 8080
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)