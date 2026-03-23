"""Minimal test to verify app can load without TypeError"""
import pandas as pd
import joblib

RFG_MODEL_PATH = "model/ev_rfg_model.pkl"
CLF_MODEL_PATH = "model/ev_clf_model.pkl"
RG_FEATURES_PATH = "model/ev_rg_features.pkl"
CLF_FEATURES_PATH = "model/clf_features.pkl"

def load_models():
    """Load all ML models and config files."""
    try:
        rfg_model = joblib.load(RFG_MODEL_PATH)
        clf_model = joblib.load(CLF_MODEL_PATH)
        rg_features = joblib.load(RG_FEATURES_PATH)
        clf_features = joblib.load(CLF_FEATURES_PATH)
        return rfg_model, clf_model, rg_features, clf_features
    except Exception as e:
        print(f"Error loading models: {e}")
        return None

# Test unpacking
result = load_models()
if result is None:
    print("ERROR: load_models() returned None")
else:
    rfg_model, clf_model, rg_features, clf_features = result
    print("✓ Models unpacked successfully")
    print(f"  RFG: {type(rfg_model).__name__}")
    print(f"  CLF: {type(clf_model).__name__}")
    print(f"  RG Features: {len(rg_features)} items")
    print(f"  CLF Features: {len(clf_features)} items")
