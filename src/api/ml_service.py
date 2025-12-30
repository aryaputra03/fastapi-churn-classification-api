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
        
        # Preprocessing components
        self.label_encoders = {}
        self.scaler = None
        self._setup_preprocessors()
        
        # Try to load model on initialization
        try:
            self.load_model()
        except Exception as e:
            logger.warning(f"Could not load model on init: {str(e)}")
    
    def _setup_preprocessors(self):
        """Initialize preprocessing components"""
        try:
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
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in input data
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        
        # Convert TotalCharges to numeric if present
        if 'TotalCharges' in df.columns or 'total_charges' in df.columns:
            col_name = 'TotalCharges' if 'TotalCharges' in df.columns else 'total_charges'
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        # Fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()
        
        categorical_features = self.config.preprocess.get('categorical_features', [])
        
        for col in categorical_features:
            # Handle both lowercase and original case column names
            actual_col = None
            if col in df.columns:
                actual_col = col
            elif col.lower() in df.columns:
                actual_col = col.lower()
            elif col.replace('_', '').lower() in [c.lower() for c in df.columns]:
                # Find matching column ignoring case and underscores
                for df_col in df.columns:
                    if df_col.replace('_', '').lower() == col.replace('_', '').lower():
                        actual_col = df_col
                        break
            
            if actual_col is not None:
                # Initialize encoder if not exists
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    # Fit on all possible values for this feature
                    # For new/unseen values, we'll handle them separately
                
                try:
                    # Get unique values
                    unique_vals = df[actual_col].unique()
                    
                    # Fit encoder if not fitted or update with new values
                    if not hasattr(self.label_encoders[col], 'classes_'):
                        self.label_encoders[col].fit(df[actual_col].astype(str))
                    else:
                        # Handle unseen categories
                        existing_classes = set(self.label_encoders[col].classes_)
                        new_classes = set(unique_vals.astype(str)) - existing_classes
                        
                        if new_classes:
                            # Add new classes to encoder
                            all_classes = list(existing_classes) + list(new_classes)
                            self.label_encoders[col].classes_ = np.array(all_classes)
                    
                    # Transform the column
                    df[actual_col] = self.label_encoders[col].transform(df[actual_col].astype(str))
                    
                except Exception as e:
                    logger.warning(f"Could not encode {actual_col}: {str(e)}")
        
        return df
    
    def _scale_numerical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with scaled numerical features
        """
        df = df.copy()
        
        numerical_features = self.config.preprocess.get('numerical_features', [])
        
        # Find actual column names (handle case variations)
        existing_cols = []
        for col in numerical_features:
            if col in df.columns:
                existing_cols.append(col)
            elif col.lower() in df.columns:
                existing_cols.append(col.lower())
            else:
                # Try to find column ignoring case and underscores
                for df_col in df.columns:
                    if df_col.replace('_', '').lower() == col.replace('_', '').lower():
                        existing_cols.append(df_col)
                        break
        
        if existing_cols:
            try:
                # Fit scaler if not fitted
                if not hasattr(self.scaler, 'mean_'):
                    self.scaler.fit(df[existing_cols])
                
                # Transform
                df[existing_cols] = self.scaler.transform(df[existing_cols])
                
            except Exception as e:
                logger.warning(f"Could not scale numerical features: {str(e)}")
        
        return df
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to match expected format
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with normalized column names
        """
        df = df.copy()
        
        # Mapping from possible input names to expected names
        column_mapping = {
            'customerid': 'customer_id',
            'customer_id': 'customer_id',
            'totalcharges': 'total_charges',
            'total_charges': 'total_charges',
            'monthlycharges': 'monthly_charges',
            'monthly_charges': 'monthly_charges',
            'paymentmethod': 'payment_method',
            'payment_method': 'payment_method',
            'internetservice': 'internet_service',
            'internet_service': 'internet_service',
        }
        
        # Rename columns
        rename_dict = {}
        for col in df.columns:
            col_lower = col.lower().replace(' ', '_')
            if col_lower in column_mapping:
                rename_dict[col] = column_mapping[col_lower]
            elif col.lower() != col:
                rename_dict[col] = col.lower()
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        return df
    
    def preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data to match model expectations
        Complete preprocessing pipeline including encoding and scaling
        
        Args:
            data: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        # Make a copy
        df = data.copy()
        
        # Normalize column names
        df = self._normalize_column_names(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Remove customer_id if present (not used in prediction)
        customer_id_cols = ['customer_id', 'customerID', 'customerid']
        for col in customer_id_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Encode categorical features
        df = self._encode_categorical(df)
        
        # Scale numerical features
        df = self._scale_numerical(df)
        
        # Ensure correct column order
        expected_features = [
            'gender', 'tenure', 'monthly_charges', 'total_charges',
            'contract', 'payment_method', 'internet_service'
        ]
        
        # Select only expected features that exist
        available_features = [col for col in expected_features if col in df.columns]
        
        if not available_features:
            # Try alternative column names
            alt_mapping = {
                'Gender': 'gender',
                'Tenure': 'tenure',
                'MonthlyCharges': 'monthly_charges',
                'TotalCharges': 'total_charges',
                'Contract': 'contract',
                'PaymentMethod': 'payment_method',
                'InternetService': 'internet_service'
            }
            
            for old_col, new_col in alt_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
            
            available_features = [col for col in expected_features if col in df.columns]
        
        if available_features:
            df = df[available_features]
        else:
            raise ValueError(f"No expected features found in input data. Available columns: {list(df.columns)}")
        
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
            # Preprocess input (includes all transformations)
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
        
        return {
            "model_type": self.model_info.get("model_type", "Unknown"),
            "model_version": self.model_info.get("model_version", "1.0.0"),
            "features": features,
            "trained_at": self.model_info.get("loaded_at"),
            "accuracy": None  # Could load from metrics.json
        }