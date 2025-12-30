import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from src.config import Config
from src.utils import logger


class MLService:
    """
    Service class for ML model operations
    Uses the SAME preprocessing as DataPreprocessor from preprocess.py
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
        
        # Preprocessing components - match preprocess.py exactly
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []
        
        # Try to load model on initialization
        try:
            self.load_model()
            self._setup_preprocessors()
        except Exception as e:
            logger.warning(f"Could not load model on init: {str(e)}")
    
    def _setup_preprocessors(self):
        """Initialize preprocessing components - match preprocess.py"""
        try:
            if self.config is None:
                self.config = Config("params.yaml")
            
            # Initialize scaler based on config - EXACTLY like preprocess.py
            scale_method = self.config.preprocess.get('scale_method', 'standard')
            if scale_method == 'standard':
                self.scaler = StandardScaler()
            elif scale_method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                logger.warning(f"Unknown scaling method {scale_method}, defaulting to standard scaler.")
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
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values - SAME as preprocess.py
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with handled missing values
        """
        strategy = self.config.preprocess.get('handling_missing', 'median')
        logger.info(f"Handling missing values using strategy: {strategy}")

        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            logger.info(f"Total missing values before: {missing_before}")

        # Convert TotalCharges to numeric - SAME as preprocess.py
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        if strategy == 'drop':
            df = df.dropna()
        elif strategy == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val.iloc[0])
        
        elif strategy == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    mode_val = df[col].mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].fillna(mode_val.iloc[0])

        missing_after = df.isnull().sum().sum()
        if missing_after > 0:
            logger.info(f"  Missing values after: {missing_after}")

        return df
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features - SAME as preprocess.py
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with engineered features
        """
        fe_config = self.config.preprocess.get('feature_engineering', {})

        if not fe_config:
            return df
        
        logger.info("Create Engineered Feature")

        # Tenure bins - SAME as preprocess.py
        if fe_config.get("create_tenure_bins", False) and 'tenure' in df.columns:
            df['tenure_group'] = pd.cut(
                df['tenure'],
                bins=[0, 12, 24, 48, 72],
                labels=['0-1yr', '1-2yr', '2-4yr', '4-6yr']
            )
            logger.info("Created Tenure Group")

        # Charge ratio - SAME as preprocess.py
        if fe_config.get("create_charge_ratio", False):
            if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
                df['charge_ratio'] = df['TotalCharges']/(df['MonthlyCharges']+1e-6)
                logger.info("Created Charge Ratio")
        
        return df
    
    def encode_categorical(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Encode categorical column - SAME as preprocess.py
        
        Args:
            df: DataFrame
            column: Column name to encode
            
        Returns:
            DataFrame with encoded column
        """
        if column not in df.columns:
            logger.warning(f"Column not found: {column}")
            return df
        
        # Use existing encoder or create new one
        if column not in self.label_encoders:
            self.label_encoders[column] = LabelEncoder()
            # Fit on current data
            self.label_encoders[column].fit(df[column].astype(str))
        else:
            # Handle unseen categories
            le = self.label_encoders[column]
            current_vals = df[column].astype(str).unique()
            existing_classes = set(le.classes_)
            new_classes = set(current_vals) - existing_classes
            
            if new_classes:
                # Expand encoder classes
                all_classes = list(existing_classes) + list(new_classes)
                le.classes_ = np.array(all_classes)

        try:
            df[column] = self.label_encoders[column].transform(df[column].astype(str))
            logger.info(f"Encoded: {column} ({len(self.label_encoders[column].classes_)} classes)")
        except Exception as e:
            logger.error(f"Failed to encode {column}: {str(e)}")
        
        return df
    
    def scale_numerical(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Scale numerical columns - SAME as preprocess.py
        
        Args:
            df: DataFrame
            columns: List of column names to scale
            
        Returns:
            DataFrame with scaled columns
        """
        existing_cols = [col for col in columns if col in df.columns]

        if not existing_cols:
            logger.warning("No numerical columns found to scale")
            return df
        
        logger.info("Scaling numerical columns")

        try:
            # Fit scaler if not fitted yet
            if not hasattr(self.scaler, 'mean_') and not hasattr(self.scaler, 'data_min_'):
                logger.info("Fitting scaler on current data")
                self.scaler.fit(df[existing_cols])
            
            # Transform
            df[existing_cols] = self.scaler.transform(df[existing_cols])
            logger.info(f"Scaled {len(existing_cols)} features")
        except Exception as e:
            logger.error(f"Scaling failed: {str(e)}")
        
        return df
    
    def preprocess_input(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess input data - SAME pipeline as preprocess.py
        
        Args:
            data: Input dataframe
            
        Returns:
            Tuple of (preprocessed_dataframe, customer_ids)
        """
        logger.info("Starting preprocessing pipeline")
        
        df = data.copy()
        
        # Store customer IDs if present - they will be dropped for prediction
        customer_ids = None
        customer_id_cols = ['customer_id', 'customerID', 'customerid']
        for col in customer_id_cols:
            if col in df.columns:
                customer_ids = df[col].copy()
                break
        
        # Step 1: Handle missing values - SAME as preprocess.py
        df = self.handle_missing_values(df)
        
        # Step 2: Feature engineering - SAME as preprocess.py
        df = self.feature_engineering(df)
        
        # Step 3: Encode categorical features - SAME as preprocess.py
        categorical_features = self.config.preprocess.get('categorical_features', [])
        for col in categorical_features:
            if col in df.columns:
                df = self.encode_categorical(df, col)
        
        # Step 4: Scale numerical features - SAME as preprocess.py
        numerical_features = self.config.preprocess.get('numerical_features', [])
        df = self.scale_numerical(df, numerical_features)
        
        # Remove customer_id and target columns for prediction
        exclude_cols = customer_id_cols + [self.config.preprocess.get('target', 'Churn')]
        for col in exclude_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        logger.info("Preprocessing completed")
        
        return df, customer_ids
    
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
            # Preprocess input - returns preprocessed data and customer_ids
            X, customer_ids = self.preprocess_input(data)
            
            logger.info(f"Preprocessed input shape: {X.shape}")
            logger.info(f"Preprocessed columns: {list(X.columns)}")
            
            # Make predictions
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
        if hasattr(self.model, 'feature_names_in_'):
            features = self.model.feature_names_in_.tolist()
        
        return {
            "model_type": self.model_info.get("model_type", "Unknown"),
            "model_version": self.model_info.get("model_version", "1.0.0"),
            "features": features,
            "trained_at": self.model_info.get("loaded_at"),
            "accuracy": None  # Could load from metrics.json
        }