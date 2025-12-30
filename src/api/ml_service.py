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
    Supports both standalone models and sklearn Pipeline objects
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
        self.is_pipeline = False
        
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
            
            # Check if it's a Pipeline
            from sklearn.pipeline import Pipeline
            self.is_pipeline = isinstance(self.model, Pipeline)
            
            if self.is_pipeline:
                logger.info("Detected sklearn Pipeline - preprocessing will be handled by pipeline")
                model_type = type(self.model.named_steps.get('model', self.model)).__name__
            else:
                logger.info("Detected standalone model")
                model_type = type(self.model).__name__
            
            # Store model info
            self.model_info = {
                "model_type": model_type,
                "model_version": "1.0.0",
                "model_path": str(model_path),
                "loaded_at": pd.Timestamp.now().isoformat(),
                "is_pipeline": self.is_pipeline
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
        
        # Create mapping - keep original case variations for Pipeline compatibility
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            
            # Only normalize to lowercase with underscores
            if col != col_lower:
                column_mapping[col] = col_lower
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        return df
    
    def preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data
        
        For Pipeline models: minimal preprocessing (just normalize column names)
        For standalone models: would need full preprocessing (not implemented in this version)
        
        Args:
            data: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        df = data.copy()
        
        if self.is_pipeline:
            # Pipeline handles all preprocessing internally
            # Just normalize column names and ensure proper structure
            df = self._normalize_column_names(df)
            
            # Remove customer_id if present (not used in prediction)
            for col in ['customer_id', 'customerID', 'customerid']:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            # Convert to numeric where needed
            numeric_cols = ['tenure', 'monthly_charges', 'total_charges']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle missing TotalCharges
            if 'total_charges' in df.columns and df['total_charges'].isna().any():
                if 'monthly_charges' in df.columns and 'tenure' in df.columns:
                    df['total_charges'] = df['total_charges'].fillna(
                        df['monthly_charges'] * df['tenure']
                    )
                else:
                    df['total_charges'] = df['total_charges'].fillna(0)
            
            return df
        else:
            # For standalone models, would need full preprocessing
            # This version assumes Pipeline is being used
            raise NotImplementedError(
                "Standalone model preprocessing not implemented. "
                "Please use training_pipeline.py to create a Pipeline model."
            )
    
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
            # Preprocess input (minimal for Pipeline)
            X = self.preprocess_input(data)
            
            logger.info(f"Input shape after preprocessing: {X.shape}")
            logger.info(f"Input columns: {list(X.columns)}")
            
            if self.is_pipeline:
                # Pipeline expects DataFrame with original column structure
                predictions = self.model.predict(X)
                probabilities = self.model.predict_proba(X)
            else:
                # Standalone model would need array input
                predictions = self.model.predict(X.values)
                probabilities = self.model.predict_proba(X.values)
            
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
        try:
            if self.is_pipeline:
                # Try to get features from preprocessor step
                preprocess_step = self.model.named_steps.get('preprocess')
                if preprocess_step and hasattr(preprocess_step, 'feature_names_'):
                    features = preprocess_step.feature_names_
            elif hasattr(self.model, 'feature_names_in_'):
                features = self.model.feature_names_in_.tolist()
        except Exception:
            pass
        
        return {
            "model_type": self.model_info.get("model_type", "Unknown"),
            "model_version": self.model_info.get("model_version", "1.0.0"),
            "features": features,
            "trained_at": self.model_info.get("loaded_at"),
            "accuracy": None  # Could load from metrics.json
        }