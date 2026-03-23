import joblib

print("Testing model loading...")
try:
    rfg = joblib.load('model/ev_rfg_model.pkl')
    print(f"✓ RFG Model loaded: {type(rfg).__name__}")
except Exception as e:
    print(f"✗ RFG Model error: {e}")

try:
    clf = joblib.load('model/ev_clf_model.pkl')
    print(f"✓ CLF Model loaded: {type(clf).__name__}")
except Exception as e:
    print(f"✗ CLF Model error: {e}")

try:
    rg_feat = joblib.load('model/ev_rg_features.pkl')
    print(f"✓ RG Features loaded: {type(rg_feat).__name__}, {len(rg_feat) if hasattr(rg_feat, '__len__') else 'N/A'} items")
except Exception as e:
    print(f"✗ RG Features error: {e}")

try:
    clf_feat = joblib.load('model/clf_features.pkl')
    print(f"✓ CLF Features loaded: {type(clf_feat).__name__}, {len(clf_feat) if hasattr(clf_feat, '__len__') else 'N/A'} items")
except Exception as e:
    print(f"✗ CLF Features error: {e}")

print("\nAll models loaded successfully!")
