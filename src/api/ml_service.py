"""
ML Service for Loading and Running Model

Includes complete preprocessing pipeline matching preprocess.py
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, List
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import Config
from src.utils import logger


class DataPreprocessor:
    """
    Preprocessing class matching src/preprocess.py
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def fit_transform(self, df: pd.DataFrame, config: Config) -> pd.DataFrame:
        """Fit and transform data (for training)"""
        df = df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._feature_engineering(df, config)
        
        # Encode categorical
        categorical_features = config.preprocess.get('categorical_features', [])
        for col in categorical_features:
            if col in df.columns:
                df = self._encode_categorical(df, col, fit=True)
        
        # Scale numerical
        numerical_features = config.preprocess.get('numerical_features', [])
        existing_numerical = [col for col in numerical_features if col in df.columns]
        if existing_numerical:
            df[existing_numerical] = self.scaler.fit_transform(df[existing_numerical])
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col not in ['customerID', 'Churn']]
        
        return df
    
    def transform(self, df: pd.DataFrame, config: Config) -> pd.DataFrame:
        """Transform data (for prediction)"""
        df = df.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._feature_engineering(df, config)
        
        # Encode categorical
        categorical_features = config.preprocess.get('categorical_features', [])
        for col in categorical_features:
            if col in df.columns:
                df = self._encode_categorical(df, col, fit=False)
        
        # Scale numerical
        numerical_features = config.preprocess.get('numerical_features', [])
        existing_numerical = [col for col in numerical_features if col in df.columns]
        if existing_numerical:
            df[existing_numerical] = self.scaler.transform(df[existing_numerical])
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values"""
        # Convert TotalCharges to numeric
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].median() if df['TotalCharges'].median() else 0, inplace=True)
        
        # Fill numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median() if df[col].median() else 0, inplace=True)
        
        # Fill categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        return df
    
    def _feature_engineering(self, df: pd.DataFrame, config: Config) -> pd.DataFrame:
        """Create engineered features"""
        fe_config = config.preprocess.get('feature_engineering', {})
        
        if not fe_config:
            return df
        
        # Create tenure bins
        if fe_config.get('create_tenure_bins', False) and 'tenure' in df.columns:
            df['tenure_group'] = pd.cut(
                df['tenure'],
                bins=[0, 12, 24, 48, 72],
                labels=[0, 1, 2, 3],
                include_lowest=True
            )
            df['tenure_group'] = df['tenure_group'].astype(int)
        
        # Create charge ratio
        if fe_config.get('create_charge_ratio', False):
            if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
                df['charge_ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1e-6)
                df['charge_ratio'] = df['charge_ratio'].clip(0, 100)  # Reasonable limits
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, column: str, fit: bool = True) -> pd.DataFrame:
        """Encode categorical column"""
        if column not in df.columns:
            return df
        
        if fit:
            # Fit new encoder
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            self.label_encoders[column] = le
        else:
            # Use existing encoder
            if column in self.label_encoders:
                le = self.label_encoders[column]
                # Handle unseen categories
                df[column] = df[column].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            else:
                logger.warning(f"No encoder found for {column}, using default encoding")
                df[column] = 0
        
        return df


class MLService:
    """
    Service class for ML model operations with complete preprocessing
    """
    
    def __init__(self, model_path: str = None, config_path: str = "params.yaml"):
        """
        Initialize ML service
        
        Args:
            model_path: Path to model file
            config_path: Path to config file
        """
        self.model = None
        self.preprocessor = DataPreprocessor()
        self.model_path = model_path
        self.config_path = config_path
        self.config = None
        self.model_info = {}
        
        # Load config
        try:
            self.config = Config(config_path)
        except Exception as e:
            logger.warning(f"Could not load config: {str(e)}")
        
        # Try to load model on initialization
        try:
            self.load_model()
        except Exception as e:
            logger.warning(f"Could not load model on init: {str(e)}")
    
    def load_model(self, model_path: str = None) -> None:
        """
        Load ML model and preprocessor from disk
        
        Args:
            model_path: Optional path to model file
        """
        try:
            # Load config if not loaded
            if self.config is None:
                self.config = Config(self.config_path)
            
            # Determine model path
            if model_path is None:
                model_path = self.model_path or self.config.evaluate.get('model_path', 'models/churn_model.pkl')
            
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model
            logger.info(f"📂 Loading model from {model_path}")
            self.model = joblib.load(model_path)
            self.model_path = str(model_path)
            
            # Try to load preprocessor if it exists
            preprocessor_path = model_path.parent / "preprocessor.pkl"
            if preprocessor_path.exists():
                logger.info(f"📂 Loading preprocessor from {preprocessor_path}")
                self.preprocessor = joblib.load(preprocessor_path)
            else:
                logger.warning("⚠️  Preprocessor not found, using fresh preprocessor")
                # Initialize fresh preprocessor with config
                self._initialize_preprocessor_from_config()
            
            # Store model info
            self.model_info = {
                "model_type": type(self.model).__name__,
                "model_version": "1.0.0",
                "model_path": str(model_path),
                "loaded_at": pd.Timestamp.now().isoformat()
            }
            
            logger.info("✅ Model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {str(e)}")
            raise
    
    def _initialize_preprocessor_from_config(self):
        """Initialize preprocessor encoders from config defaults"""
        if self.config is None:
            return
        
        # Get categorical features from config
        categorical_features = self.config.preprocess.get('categorical_features', [])
        
        # Initialize label encoders with expected categories
        category_defaults = {
            'gender': ['Male', 'Female'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaymentMethod': [
                'Electronic check', 'Mailed check', 
                'Bank transfer (automatic)', 'Credit card (automatic)',
                'Bank transfer', 'Credit card'  # Variations
            ],
            'InternetService': ['DSL', 'Fiber optic', 'No']
        }
        
        for feature in categorical_features:
            if feature in category_defaults:
                le = LabelEncoder()
                le.fit(category_defaults[feature])
                self.preprocessor.label_encoders[feature] = le
                logger.info(f"  ✓ Initialized encoder for {feature}")
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def preprocess_input(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess input data to match model expectations
        
        Args:
            data: Input dataframe
            
        Returns:
            Preprocessed dataframe
        """
        try:
            # Make a copy
            df = data.copy()
            
            # Store customer_id if present
            has_customer_id = 'customerID' in df.columns or 'customer_id' in df.columns
            
            # Rename customer_id to customerID for consistency
            if 'customer_id' in df.columns and 'customerID' not in df.columns:
                df.rename(columns={'customer_id': 'customerID'}, inplace=True)
            
            # Apply preprocessing pipeline
            df = self.preprocessor.transform(df, self.config)
            
            # Remove customer_id and target if present
            columns_to_remove = ['customerID', 'customer_id', 'Churn']
            df = df.drop(columns=[col for col in columns_to_remove if col in df.columns], errors='ignore')
            
            # Get expected features
            expected_features = self._get_expected_features()
            
            # Ensure all expected features are present
            for feature in expected_features:
                if feature not in df.columns:
                    logger.warning(f"⚠️  Missing feature: {feature}, adding with default value 0")
                    df[feature] = 0
            
            # Reorder columns to match expected features
            df = df[expected_features]
            
            logger.info(f"✓ Preprocessed {len(df)} samples with {len(df.columns)} features")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Preprocessing failed: {str(e)}")
            raise
    
    def _get_expected_features(self) -> List[str]:
        """Get list of expected features for model"""
        # Priority 1: Use stored feature names from preprocessor
        if hasattr(self.preprocessor, 'feature_names') and self.preprocessor.feature_names:
            return self.preprocessor.feature_names
        
        # Priority 2: Use model's feature names if available
        if hasattr(self.model, 'feature_names_in_'):
            return self.model.feature_names_in_.tolist()
        
        # Priority 3: Build from config
        if self.config:
            features = []
            
            # Numerical features
            numerical = self.config.preprocess.get('numerical_features', [])
            features.extend(numerical)
            
            # Categorical features
            categorical = self.config.preprocess.get('categorical_features', [])
            features.extend(categorical)
            
            # Engineered features
            fe_config = self.config.preprocess.get('feature_engineering', {})
            if fe_config.get('create_tenure_bins'):
                features.append('tenure_group')
            if fe_config.get('create_charge_ratio'):
                features.append('charge_ratio')
            
            return features
        
        # Fallback: Use default feature list
        logger.warning("⚠️  Using default feature list")
        return [
            'gender', 'tenure', 'MonthlyCharges', 'TotalCharges',
            'Contract', 'PaymentMethod', 'InternetService'
        ]
    
    def predict(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions
        
        Args:
            data: Input dataframe with raw features
            
        Returns:
            predictions, probabilities
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Preprocess input
            logger.info(f"🔮 Making predictions for {len(data)} samples...")
            X = self.preprocess_input(data)
            
            # Make predictions
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            logger.info(f"✅ Predictions completed: {predictions.shape[0]} samples")
            
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {str(e)}")
            import traceback
            traceback.print_exc()
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
        
        # Get features
        features = self._get_expected_features()
        
        return {
            "model_type": self.model_info.get("model_type", "Unknown"),
            "model_version": self.model_info.get("model_version", "1.0.0"),
            "features": features,
            "trained_at": self.model_info.get("loaded_at"),
            "accuracy": None  # Could load from metrics.json
        }
    
    def save_preprocessor(self, path: str = None):
        """
        Save preprocessor to disk for reuse
        
        Args:
            path: Path to save preprocessor
        """
        if path is None:
            if self.model_path:
                model_dir = Path(self.model_path).parent
                path = model_dir / "preprocessor.pkl"
            else:
                path = "models/preprocessor.pkl"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.preprocessor, path)
        logger.info(f"💾 Preprocessor saved to {path}")