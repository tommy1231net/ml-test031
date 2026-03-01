import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
import os
import json

# 1. Configuration & Label Mapping
target = 'label'
features = ['trip_seconds', 'trip_miles', 'fare', 'extras', 'tolls', 'hour', 'dayofweek', 'pickup_area', 'dropoff_area', 'company_id']
label_mapping = {'Credit Card': 0, 'Cash': 1, 'Mobile': 2, 'Prcard': 3}

def preprocess_df(df):
    """Preprocess raw data including feature engineering and encoding"""
    df['hour'] = df['trip_start_timestamp'].dt.hour
    df['dayofweek'] = df['trip_start_timestamp'].dt.dayofweek
    
    cat_cols = ['pickup_area', 'dropoff_area', 'company_id']
    for col in cat_cols:
        df[col] = df[col].astype('category').cat.codes
        
    y = df[target].map(label_mapping)
    X = df[features]
    return X, y

# 2. Load and Train on FULL 2022 Data
train_file = 'chicago_taxi_train_2022.csv'
print(f"Loading FULL Training Data: {train_file}...")
df_train_raw = pd.read_csv(train_file, parse_dates=['trip_start_timestamp'])
X_train_full, y_train_full = preprocess_df(df_train_raw)

# 3. Train Model (Using 100% of 2022 data)
print("Training final XGBoost model on 100% of 2022 data...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softmax',
    num_class=len(label_mapping),
    tree_method='hist',
    random_state=42
)
model.fit(X_train_full, y_train_full)

# 4. Export the Trained Model and Mapping
model.save_model('taxi_payment_model.json')
with open('label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)
print("\nModel and mapping exported successfully.")

# 5. Final Evaluation on 2023 Data (Hold-out Test)
test_file = 'chicago_taxi_test_2023.csv'
if os.path.exists(test_file):
    print(f"\nEvaluating on UNSEEN 2023 Data: {test_file}...")
    df_test_raw = pd.read_csv(test_file, parse_dates=['trip_start_timestamp'])
    X_test, y_test = preprocess_df(df_test_raw)

    y_pred_test = model.predict(X_test)
    print("\n--- Out-of-Time Test Performance (2023) ---")
    print(classification_report(y_test, y_pred_test, target_names=label_mapping.keys()))