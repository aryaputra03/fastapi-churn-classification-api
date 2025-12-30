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
    Service class for ML model operations with complete preprocessing pipeline
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
        
        # Preprocessing components - will be fitted during first prediction
        self.label_encoders = {}
        self.scaler = None
        self.is_fitted = False
        
        # Try to load model on initialization
        try:
            self.load_model()
            self._setup_preprocessors()
        except Exception as e:
            logger.warning(f"Could not load model on init: {str(e)}")
    
    def _setup_preprocessors(self):
        """Initialize preprocessing components"""
        try:
            if self.config is None:
                self.config = Config("params.yaml")
            
            # Initialize scaler based on config
            scale_method = self.config.preprocess.get('scale_method', 'standard')
            if scale_method == 'standard':
                self.scaler = StandardScaler()
            else:
                self.scaler = StandardScaler()
                
        except Exception as e:
            logger.warning(f"Could not load config for preprocessors: {str(e)}")
            self.scaler = StandardScaler()
    
    def load_model(self, model_path: str = None) -> None:
        """
        Load ML model from disk
        
        Args:
            model_path: Optional path to model file
        """
        try:
            # Load config
            if self.config is None:
                self.config = Config("params.yaml")
            
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
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to match expected format
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with normalized column names
        """
        df = df.copy()
        
        # Create mapping with all variations
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            
            # Map to standard names
            if col_lower in ['customerid', 'customer_id']:
                column_mapping[col] = 'customer_id'
            elif col_lower in ['totalcharges', 'total_charges']:
                column_mapping[col] = 'total_charges'
            elif col_lower in ['monthlycharges', 'monthly_charges']:
                column_mapping[col] = 'monthly_charges'
            elif col_lower in ['paymentmethod', 'payment_method']:
                column_mapping[col] = 'payment_method'
            elif col_lower in ['internetservice', 'internet_service']:
                column_mapping[col] = 'internet_service'
            elif col != col_lower:
                column_mapping[col] = col_lower
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        return df
    
    def _prepare_encoders_and_scaler(self, df: pd.DataFrame) -> None:
        """
        Prepare encoders and scaler using reasonable defaults
        This simulates the training preprocessing without needing training data
        
        Args:
            df: Sample dataframe to determine data types
        """
        if self.is_fitted:
            return
            
        try:
            # Define expected categorical values based on domain knowledge
            categorical_defaults = {
                'gender': ['Male', 'Female'],
                'contract': ['Month-to-month', 'One year', 'Two year'],
                'payment_method': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
                'internet_service': ['DSL', 'Fiber optic', 'No']
            }
            
            # Initialize label encoders with default values
            for col, values in categorical_defaults.items():
                if col in df.columns or col.title() in df.columns or col.replace('_', '').title() in df.columns:
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(values)
            
            # Fit scaler on current data (will normalize based on this sample)
            numerical_features = ['tenure', 'monthly_charges', 'total_charges']
            existing_num_cols = [col for col in numerical_features if col in df.columns]
            
            if existing_num_cols and self.scaler is not None:
                # Use reasonable defaults for scaling if data is limited
                if len(df) > 0:
                    self.scaler.fit(df[existing_num_cols])
                    self.is_fitted = True
                    
        except Exception as e:
            logger.warning(f"Could not fully prepare preprocessors: {str(e)}")
    
    def preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data to match model expectations
        Simplified preprocessing that doesn't require pre-fitted transformers
        
        Args:
            data: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        # Make a copy
        df = data.copy()
        
        # Normalize column names
        df = self._normalize_column_names(df)
        
        # Remove customer_id if present
        for col in ['customer_id', 'customerID', 'customerid']:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Handle missing values in TotalCharges
        if 'total_charges' in df.columns:
            df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
            if df['total_charges'].isna().any():
                # Use monthly_charges * tenure as estimate
                if 'monthly_charges' in df.columns and 'tenure' in df.columns:
                    df['total_charges'] = df['total_charges'].fillna(
                        df['monthly_charges'] * df['tenure']
                    )
                else:
                    df['total_charges'] = df['total_charges'].fillna(0)
        
        # Prepare encoders if not done yet
        self._prepare_encoders_and_scaler(df)
        
        # Encode categorical features - simple mapping
        categorical_mappings = {
            'gender': {'Male': 0, 'Female': 1, 'male': 0, 'female': 1},
            'contract': {
                'Month-to-month': 0, 'One year': 1, 'Two year': 2,
                'month-to-month': 0, 'one year': 1, 'two year': 2
            },
            'payment_method': {
                'Electronic check': 0, 'Mailed check': 1, 
                'Bank transfer': 2, 'Credit card': 3,
                'electronic check': 0, 'mailed check': 1,
                'bank transfer': 2, 'credit card': 3
            },
            'internet_service': {
                'DSL': 0, 'Fiber optic': 1, 'No': 2,
                'dsl': 0, 'fiber optic': 1, 'no': 2
            }
        }
        
        for col, mapping in categorical_mappings.items():
            if col in df.columns:
                # Map values, use 0 as default for unknown values
                df[col] = df[col].astype(str).map(mapping).fillna(0).astype(int)
        
        # Scale numerical features - simple min-max approach
        numerical_features = ['tenure', 'monthly_charges', 'total_charges']
        for col in numerical_features:
            if col in df.columns:
                # Ensure numeric type
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Apply scaler if fitted
        existing_num_cols = [col for col in numerical_features if col in df.columns]
        if existing_num_cols and self.scaler is not None and self.is_fitted:
            try:
                df[existing_num_cols] = self.scaler.transform(df[existing_num_cols])
            except Exception as e:
                logger.warning(f"Could not scale features: {str(e)}")
                # Continue without scaling
        
        # Ensure correct column order
        expected_features = [
            'gender', 'tenure', 'monthly_charges', 'total_charges',
            'contract', 'payment_method', 'internet_service'
        ]
        
        # Keep only expected features
        available_features = [col for col in expected_features if col in df.columns]
        
        if not available_features:
            raise ValueError(
                f"No expected features found. "
                f"Expected: {expected_features}, "
                f"Got: {list(df.columns)}"
            )
        
        # Reorder columns
        df = df[available_features]
        
        return df
    
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
            
            logger.info(f"Preprocessed input shape: {X.shape}")
            logger.info(f"Preprocessed columns: {list(X.columns)}")
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
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
        
        return {
            "model_type": self.model_info.get("model_type", "Unknown"),
            "model_version": self.model_info.get("model_version", "1.0.0"),
            "features": features,
            "trained_at": self.model_info.get("loaded_at"),
            "accuracy": None  # Could load from metrics.json
        }