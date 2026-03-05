import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def train_model():
    # 1. Load data from local CSV
    print("Loading CSV data...")
    try:
        df = pd.read_csv('training_data.csv')
    except FileNotFoundError:
        print("Error: training_data.csv not found. Please export your BigQuery result to CSV first.")
        return

    # 2. Features and Label separation
    # Categorical features: labels that need One-Hot Encoding
    cat_cols = ['input_category', 'input_brand', 'input_department', 'gender', 'country']
    # Numerical features: values that need Scaling
    num_cols = ['input_price', 'age']
    # Target label
    label_col = 'label'

    X_cat = df[cat_cols]
    X_num = df[num_cols]
    y = df[label_col]

    print(f"Dataset size: {len(df)} rows")

    # 3. Preprocessing (Individual Encoders)
    print("Preprocessing...")
    
    # Category features -> OneHotEncoding
    # handle_unknown='ignore' is crucial for inference with new/unseen brands
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_cat_encoded = encoder.fit_transform(X_cat)

    # Numerical features -> Scaling
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Combine categorical and numerical features into one matrix
    X_final = np.hstack([X_cat_encoded, X_num_scaled])

    # 4. Train Model
    print("Training Model...")
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    # Using RandomForest - robust for tabular data with diverse features
    model = RandomForestClassifier(
        n_estimators=200,         # 木の数を増やして安定性を高める（50 -> 200）
        max_depth=25,             # 木の深さを深くして、より細かい相関を学習（10 -> 25）
        min_samples_leaf=2,       # 1つの葉に最低2サンプル（5 -> 2）より詳細なパターンを許容
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. Save Artifacts for Inference
    print("Saving artifacts to local...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(encoder, 'encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # We also save the original column names for categorical features 
    # to reconstruct the DataFrame correctly during inference
    joblib.dump(cat_cols, 'cat_cols.pkl')
    joblib.dump(num_cols, 'num_cols.pkl')

    accuracy = model.score(X_test, y_test)
    print(f"Success! Model Accuracy: {accuracy:.4f}")
    print("Artifacts generated: model.pkl, encoder.pkl, scaler.pkl, cat_cols.pkl, num_cols.pkl")

if __name__ == "__main__":
    train_model()