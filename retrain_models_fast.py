"""
Fast Model Retraining Script for Python 3.13 Compatibility
Uses a sampled dataset for quick training
"""

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("EV Charging Demand ML - Fast Model Retraining")
print("=" * 60)

# Load datasets
print("\n[1/5] Loading data...")
try:
    DATASET_PATH = "data_set/ev_dataset.parquet"
    STATION_DATA_PATH = "data_set/station_info.parquet"
    
    # Load and sample for faster training
    dataset = pd.read_parquet(DATASET_PATH)
    station_data = pd.read_parquet(STATION_DATA_PATH)
    
    # Sample 50% of data for faster training (can adjust this)
    sample_size = int(len(dataset) * 0.5)
    dataset_sample = dataset.sample(n=sample_size, random_state=42)
    
    print(f"  ✓ Loaded full dataset: {dataset.shape[0]} rows")
    print(f"  ✓ Using sample: {dataset_sample.shape[0]} rows (50%)")
    print(f"  ✓ Loaded station data: {station_data.shape[0]} rows")
except Exception as e:
    print(f"  ✗ Error loading data: {e}")
    exit(1)

# Prepare features
print("\n[2/5] Preparing features...")
try:
    # Select only numeric features
    regression_features = dataset_sample.select_dtypes(include=['number']).columns.tolist()
    regression_features = [col for col in regression_features 
                          if col not in ['targets_utilization_t+1', 'targets_utilization_t+2', 'station_id']]
    
    print(f"  ✓ Found {len(regression_features)} numeric features")
    print(f"  Features: {regression_features[:5]}... (showing first 5)")
except Exception as e:
    print(f"  ✗ Error preparing features: {e}")
    exit(1)

# Prepare and train RFG Model
print("\n[3/5] Training Regression Model (RFG)...")
try:
    X_rfg = dataset_sample[regression_features].fillna(0)
    y_rfg = dataset_sample['targets_utilization_t+1'].fillna(0)
    
    print(f"  Features shape: {X_rfg.shape}")
    print(f"  Target shape: {y_rfg.shape}")
    
    # Split data
    X_train_rfg, X_test_rfg, y_train_rfg, y_test_rfg = train_test_split(
        X_rfg, y_rfg, test_size=0.2, random_state=42
    )
    
    # Train with reduced parameters for speed
    print("  Training Random Forest Regressor (this may take 2-3 minutes)...")
    rfg_model = RandomForestRegressor(
        n_estimators=50,  # Reduced from 100
        max_depth=15,      # Reduced from 20
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=1,  # Single-threaded to avoid memory issues
        verbose=1
    )
    rfg_model.fit(X_train_rfg, y_train_rfg)
    
    train_score = rfg_model.score(X_train_rfg, y_train_rfg)
    test_score = rfg_model.score(X_test_rfg, y_test_rfg)
    print(f"  ✓ RFG Model trained")
    print(f"    Train R² Score: {train_score:.4f}")
    print(f"    Test R² Score: {test_score:.4f}")
    
    # Save with joblib
    joblib.dump(rfg_model, 'model/ev_rfg_model.pkl', compress=3)
    joblib.dump(regression_features, 'model/ev_rg_features.pkl', compress=3)
    print("  ✓ Saved: model/ev_rfg_model.pkl")
    print("  ✓ Saved: model/ev_rg_features.pkl")
    
except Exception as e:
    print(f"  ✗ Error training RFG: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Prepare and train CLF Model
print("\n[4/5] Training Classification Model (CLF)...")
try:
    X_clf = dataset_sample[regression_features].fillna(0)
    y_clf = (dataset_sample['targets_utilization_t+1'] > 0.7).astype(int)
    
    print(f"  Class distribution:")
    print(f"    Low utilization: {(y_clf == 0).sum()} ({(y_clf == 0).sum()/len(y_clf)*100:.1f}%)")
    print(f"    High utilization: {(y_clf == 1).sum()} ({(y_clf == 1).sum()/len(y_clf)*100:.1f}%)")
    
    # Split data
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    
    # Train with reduced parameters
    print("  Training Random Forest Classifier (this may take 2-3 minutes)...")
    clf_model = RandomForestClassifier(
        n_estimators=50,  # Reduced from 100
        max_depth=15,      # Reduced from 20
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=1,  # Single-threaded
        verbose=1
    )
    clf_model.fit(X_train_clf, y_train_clf)
    
    train_score = clf_model.score(X_train_clf, y_train_clf)
    test_score = clf_model.score(X_test_clf, y_test_clf)
    print(f"  ✓ CLF Model trained")
    print(f"    Train Accuracy: {train_score:.4f}")
    print(f"    Test Accuracy: {test_score:.4f}")
    
    # Save with joblib
    joblib.dump(clf_model, 'model/ev_clf_model.pkl', compress=3)
    joblib.dump(regression_features, 'model/clf_features.pkl', compress=3)
    print("  ✓ Saved: model/ev_clf_model.pkl")
    print("  ✓ Saved: model/clf_features.pkl")
    
except Exception as e:
    print(f"  ✗ Error training CLF: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Verify models
print("\n[5/5] Verifying saved models...")
try:
    verify_rfg = joblib.load('model/ev_rfg_model.pkl')
    verify_clf = joblib.load('model/ev_clf_model.pkl')
    verify_rg_feat = joblib.load('model/ev_rg_features.pkl')
    verify_clf_feat = joblib.load('model/clf_features.pkl')
    
    print(f"  ✓ RFG Model: {type(verify_rfg).__name__}")
    print(f"  ✓ CLF Model: {type(verify_clf).__name__}")
    print(f"  ✓ RFG Features: {len(verify_rg_feat)} features")
    print(f"  ✓ CLF Features: {len(verify_clf_feat)} features")
    
except Exception as e:
    print(f"  ✗ Verification failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✓ Models retrained and saved successfully!")
print("=" * 60)
print("\nNext step: Run 'streamlit run app.py'")
