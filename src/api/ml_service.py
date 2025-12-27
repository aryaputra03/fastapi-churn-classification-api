"""
ML Service for Loading and Running Model
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
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
            self.config = Config("params.yaml")

            if model_path is None:
                model_path = self.model_path or self.config.evaluate['model_path']
            
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            logger.info(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
            self.model_path = str(model_path)
            
            self.model_info = {
                "model_type": type(self.model).__name__,
                "model_version": "1.0.0",
                "model_path": str(model_path),
                "loaded_at": pd.Timestamp.now().isoformat()
            }

            logger.info(" Model loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
        
    def preprocess_input(self, data: pd.DataFrame)->pd.DataFrame:
        """
        Preprocess input data to match model expectations
        
        Args:
            data: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        df = data.copy()

        if "customer_id" in df.columns:
            df = df.drop(columns=['customer_id'])
        
        expeted_features = [
            'gender', 'tenure', 'monthly_charges', 'total_charges',
            'contract', 'payment_method', 'internet_service'
        ]

        df = df[expeted_features]

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
            X = self.preprocess_input(data)
            
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
        
        features = []
        if hasattr(self.model, 'feature_names_in_'):
            features = self.model.feature_names_in_.tolist()
        
        return {
            "model_type": self.model_info.get("model_type", "Unknown"),
            "model_version": self.model_info.get("model_version", "1.0.0"),
            "features" : features,
            "trained_at": self.model_info.get("loaded_at"),
            "accuracy": None
        }
