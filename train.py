import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def train_model():
    # 1. Load the dataset
    print("Loading CSV data...")
    try:
        df = pd.read_csv('training_data.csv')
    except FileNotFoundError:
        print("Error: training_data.csv not found.")
        return

    # 2. Define features (Ensuring 'input_brand' is excluded)
    cat_cols = ['input_category', 'input_department', 'gender', 'country']
    num_cols = ['input_price', 'age']
    label_col = 'label'

    X_cat = df[cat_cols]
    X_num = df[num_cols]
    y = df[label_col]

    # 3. Preprocessing
    print("Preprocessing data...")
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_encoded = encoder.fit_transform(X_cat)

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    X_final = np.hstack([X_cat_encoded, X_num_scaled])

    # 4. Train Model with Light-weight Parameters
    # We reduced n_estimators and max_depth to keep the model size small (under 100MB)
    print("Training Optimized Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. Save Artifacts with Compression
    print("Saving artifacts with compression...")
    # compress=3 provides a good balance between speed and size reduction
    joblib.dump(model, 'model.pkl', compress=3) 
    joblib.dump(encoder, 'encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(cat_cols, 'cat_cols.pkl')
    joblib.dump(num_cols, 'num_cols.pkl')

    accuracy = model.score(X_test, y_test)
    print(f"Success! Model Accuracy: {accuracy:.4f}")
    
    # Check physical file size
    import os
    size_mb = os.path.getsize('model.pkl') / (1024 * 1024)
    print(f"Final model.pkl size: {size_mb:.2f} MB")

if __name__ == "__main__":
    train_model()