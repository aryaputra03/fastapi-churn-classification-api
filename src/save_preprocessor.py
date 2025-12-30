"""
Save Preprocessor After Training

Run this after training to save the fitted preprocessor.
"""

import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import Config
from src.utils import logger

def create_and_save_preprocessor():
    """Create and save preprocessor with fitted encoders and scaler"""
    
    config = Config("params.yml")
    model_path = Path(config.evaluate['model_path'])
    preprocessor_path = model_path.parent / "preprocessor.pkl"
    
    # Initialize label encoders with expected categories
    label_encoders = {
        'gender': LabelEncoder().fit(['Female', 'Male']),
        'Contract': LabelEncoder().fit(['Month-to-month', 'One year', 'Two year']),
        'PaymentMethod': LabelEncoder().fit([
            'Bank transfer (automatic)',
            'Credit card (automatic)',
            'Electronic check',
            'Mailed check'
        ]),
        'InternetService': LabelEncoder().fit(['DSL', 'Fiber optic', 'No'])
    }
    
    # Initialize scaler (will be fitted during preprocessing)
    scale_method = config.preprocess.get('scale_method', 'standard')
    if scale_method == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    # For the scaler, we need to fit it on representative data
    # Using typical ranges for tenure, monthly_charges, total_charges
    import numpy as np
    representative_data = np.array([
        [0, 18.25, 18.25],      # Min values
        [72, 118.75, 8564.75],  # Max values
        [36, 65.0, 2500.0]      # Mid values
    ])
    scaler.fit(representative_data)
    
    # Save preprocessor
    preprocessor = {
        'label_encoders': label_encoders,
        'scaler': scaler
    }
    
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"💾 Preprocessor saved to {preprocessor_path}")
    
    return preprocessor_path

if __name__ == '__main__':
    create_and_save_preprocessor()