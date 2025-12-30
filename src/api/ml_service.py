"""
ML Service for Loading and Running Model
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
from src.config import Config
from src.utils import logger


class MLService:
    """
    Service class for ML model operations
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize ML service
        
        Args:
            model_path: Path to model file
        """
        self.model = None
        self.model_path = model_path
        self.config = None
        self.model_info = {}
        self.label_encoders = {}
        self.scaler = None
        
        # Try to load model on initialization
        try:
            self.load_model()
        except Exception as e:
            logger.warning(f"Could not load model on init: {str(e)}")
    
    def load_model(self, model_path: str = None) -> None:
        """
        Load ML model from disk
        
        Args:
            model_path: Optional path to model file
        """
        try:
            # Load config
            self.config = Config("params.yml")
            
            # Determine model path
            if model_path is None:
                model_path = self.model_path or self.config.evaluate['model_path']
            
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            self.model_path = str(model_path)
            
            # Try to load saved preprocessor
            preprocessor_path = model_path.parent / "preprocessor.pkl"
            if preprocessor_path.exists():
                logger.info(f"Loading preprocessor from {preprocessor_path}")
                preprocessor = joblib.load(preprocessor_path)
                self.label_encoders = preprocessor.get('label_encoders', {})
                self.scaler = preprocessor.get('scaler')
                logger.info("Preprocessor loaded successfully")
            else:
                logger.warning("No saved preprocessor found, initializing new one")
                self._initialize_preprocessors()
            
            # Store model info
            self.model_info = {
                "model_type": type(self.model).__name__,
                "model_version": "1.0.0",
                "model_path": str(model_path),
                "loaded_at": pd.Timestamp.now().isoformat()
            }
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _initialize_preprocessors(self):
        """Initialize label encoders and scaler with expected mappings"""
        # Initialize label encoders with expected categories
        self.label_encoders = {
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
        
        # Initialize scaler
        scale_method = self.config.preprocess.get('scale_method', 'standard')
        if scale_method == 'standard':
            self.scaler = StandardScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        
        logger.info("Preprocessors initialized")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def preprocess_input(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess input data to match model expectations
        
        Args:
            data: Input dataframe
            
        Returns:
            Preprocessed numpy array ready for prediction
        """
        # Make a copy
        df = data.copy()
        
        # Remove customer_id if present (not used in prediction)
        if 'customer_id' in df.columns:
            df = df.drop(columns=['customer_id'])
        
        # Map input columns to expected feature names
        column_mapping = {
            'gender': 'gender',
            'tenure': 'tenure',
            'monthly_charges': 'MonthlyCharges',
            'total_charges': 'TotalCharges',
            'contract': 'Contract',
            'payment_method': 'PaymentMethod',
            'internet_service': 'InternetService'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Handle missing values in TotalCharges
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'])
        
        # Encode categorical variables
        categorical_cols = ['gender', 'Contract', 'PaymentMethod', 'InternetService']
        for col in categorical_cols:
            if col in df.columns:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except Exception as e:
                    logger.warning(f"Error encoding {col}: {str(e)}")
                    # Fallback: use fit_transform
                    df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        
        # Scale numerical features
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        existing_numerical = [col for col in numerical_cols if col in df.columns]
        
        if existing_numerical:
            # For prediction, we need to fit the scaler on typical ranges
            # This is a simplified approach - ideally load the fitted scaler
            try:
                df[existing_numerical] = self.scaler.fit_transform(df[existing_numerical])
            except Exception as e:
                logger.warning(f"Scaling error: {str(e)}")
        
        # Ensure correct column order
        expected_order = ['gender', 'tenure', 'MonthlyCharges', 'TotalCharges', 
                         'Contract', 'PaymentMethod', 'InternetService']
        
        # Reorder columns
        df = df[expected_order]
        
        return df.values
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            data: Input dataframe
            
        Returns:
            predictions, probabilities
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess input
            X = self.preprocess_input(data)
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_model_loaded():
            return {
                "model_type": "Not loaded",
                "model_version": "N/A",
                "features": [],
                "trained_at": None,
                "accuracy": None
            }
        
        # Try to get feature names
        features = []
        if hasattr(self.model, 'feature_names_in_'):
            features = self.model.feature_names_in_.tolist()
        else:
            features = ['gender', 'tenure', 'MonthlyCharges', 'TotalCharges', 
                       'Contract', 'PaymentMethod', 'InternetService']
        
        return {
            "model_type": self.model_info.get("model_type", "Unknown"),
            "model_version": self.model_info.get("model_version", "1.0.0"),
            "features": features,
            "trained_at": self.model_info.get("loaded_at"),
            "accuracy": None  # Could load from metrics.json
        }