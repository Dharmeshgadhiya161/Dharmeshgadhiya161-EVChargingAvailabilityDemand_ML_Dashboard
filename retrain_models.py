"""
Model Retraining Script for Python 3.13 Compatibility
Regenerates ML models using joblib instead of pickle for better compatibility
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("EV Charging Demand ML - Model Retraining for Python 3.13")
print("=" * 60)

# Load datasets
print("\n[1/4] Loading data...")
try:
    DATASET_PATH = "data_set/ev_dataset.parquet"
    STATION_DATA_PATH = "data_set/station_info.parquet"
    
    dataset = pd.read_parquet(DATASET_PATH)
    station_data = pd.read_parquet(STATION_DATA_PATH)
    print(f"  ✓ Loaded dataset: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    print(f"  ✓ Loaded station data: {station_data.shape[0]} rows")
except Exception as e:
    print(f"  ✗ Error loading data: {e}")
    exit(1)

# Prepare data for RFG Model (Regression)
print("\n[2/4] Preparing Regression Model (RFG)...")
try:
    # Select only numeric features for regression
    regression_features = dataset.select_dtypes(include=['number']).columns.tolist()
    # Remove target columns
    regression_features = [col for col in regression_features 
                          if col not in ['targets_utilization_t+1', 'targets_utilization_t+2', 'station_id']]
    
    print(f"  Using {len(regression_features)} numeric features")
    X_rfg = dataset[regression_features].fillna(0)
    y_rfg = dataset['targets_utilization_t+1'].fillna(0)
    
    # Split data
    X_train_rfg, X_test_rfg, y_train_rfg, y_test_rfg = train_test_split(
        X_rfg, y_rfg, test_size=0.2, random_state=42
    )
    
    # Train RFG Model
    print("  Training Random Forest Regressor...")
    rfg_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    rfg_model.fit(X_train_rfg, y_train_rfg)
    
    # Evaluate
    train_score = rfg_model.score(X_train_rfg, y_train_rfg)
    test_score = rfg_model.score(X_test_rfg, y_test_rfg)
    print(f"  ✓ RFG Model trained")
    print(f"    Train R² Score: {train_score:.4f}")
    print(f"    Test R² Score: {test_score:.4f}")
    
    # Save model and features using joblib
    joblib.dump(rfg_model, 'model/ev_rfg_model.pkl', compress=3)
    joblib.dump(X_train_rfg.columns.tolist(), 'model/ev_rg_features.pkl', compress=3)
    print("  ✓ RFG Model saved: model/ev_rfg_model.pkl")
    print("  ✓ RFG Features saved: model/ev_rg_features.pkl")
    
except Exception as e:
    print(f"  ✗ Error training RFG model: {e}")
    exit(1)

# Prepare data for CLF Model (Classification)
print("\n[3/4] Preparing Classification Model (CLF)...")
try:
    # Use same numeric features for classification
    classification_features = dataset.select_dtypes(include=['number']).columns.tolist()
    classification_features = [col for col in classification_features 
                              if col not in ['targets_utilization_t+1', 'targets_utilization_t+2', 'station_id']]
    
    print(f"  Using {len(classification_features)} numeric features")
    X_clf = dataset[classification_features].fillna(0)
    # Create binary classification: high utilization (>0.7) vs low utilization (<=0.7)
    y_clf = (dataset['targets_utilization_t+1'] > 0.7).astype(int)
    
    # Split data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    # Train CLF Model
    print("  Training Random Forest Classifier...")
    clf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    clf_model.fit(X_train_clf, y_train_clf)
    
    # Evaluate
    train_score = clf_model.score(X_train_clf, y_train_clf)
    test_score = clf_model.score(X_test_clf, y_test_clf)
    print(f"  ✓ CLF Model trained")
    print(f"    Train Accuracy: {train_score:.4f}")
    print(f"    Test Accuracy: {test_score:.4f}")
    
    # Save model and features using joblib
    joblib.dump(clf_model, 'model/ev_clf_model.pkl', compress=3)
    joblib.dump(X_train_clf.columns.tolist(), 'model/clf_features.pkl', compress=3)
    print("  ✓ CLF Model saved: model/ev_clf_model.pkl")
    print("  ✓ CLF Features saved: model/clf_features.pkl")
    
except Exception as e:
    print(f"  ✗ Error training CLF model: {e}")
    exit(1)

# Verify models
print("\n[4/4] Verifying saved models...")
try:
    verify_rfg = joblib.load('model/ev_rfg_model.pkl')
    verify_clf = joblib.load('model/ev_clf_model.pkl')
    verify_rg_feat = joblib.load('model/ev_rg_features.pkl')
    verify_clf_feat = joblib.load('model/clf_features.pkl')
    
    print(f"  ✓ RFG Model verified: {type(verify_rfg).__name__}")
    print(f"  ✓ CLF Model verified: {type(verify_clf).__name__}")
    print(f"  ✓ RFG Features verified: {len(verify_rg_feat)} features")
    print(f"  ✓ CLF Features verified: {len(verify_clf_feat)} features")
    
except Exception as e:
    print(f"  ✗ Verification failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✓ All models successfully retrained and saved!")
print("=" * 60)
print("\nYou can now run: streamlit run app.py")
