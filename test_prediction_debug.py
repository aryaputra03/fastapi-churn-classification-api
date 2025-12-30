"""
Debug script to test prediction locally
"""

import pandas as pd
from src.api.ml_service import MLService

# Test data
test_data = {
    "customer_id": "C12345",
    "gender": "Male",
    "tenure": 24,
    "monthly_charges": 75.5,
    "total_charges": 1810.0,
    "contract": "One year",
    "payment_method": "Bank transfer",
    "internet_service": "Fiber optic"
}

print("=" * 60)
print("Testing ML Service")
print("=" * 60)

# Initialize service
ml_service = MLService()

# Check if model loaded
if ml_service.is_model_loaded():
    print("✅ Model loaded successfully")
else:
    print("❌ Model not loaded")
    exit(1)

# Test prediction
try:
    df = pd.DataFrame([test_data])
    print(f"\nInput data:\n{df}")
    
    predictions, probabilities = ml_service.predict(df)
    
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"Prediction: {predictions[0]}")
    print(f"Churn probability: {probabilities[0][1]:.4f}")
    print(f"No churn probability: {probabilities[0][0]:.4f}")
    print("=" * 60)
    print("\n✅ Prediction successful!")
    
except Exception as e:
    print(f"\n❌ Prediction failed: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)