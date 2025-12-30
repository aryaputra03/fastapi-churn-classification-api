"""
ML Service for Loading, Training, and Running Model
Integrated with training pipeline from train_pipeline.py
"""
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import Config
from src.preprocess import DataPreprocessor
from src.train import ModelTrainer
from src.utils import load_data, logger, Timer
from src.pipelines.train_pipeline import PreprocessorWrapper


class MLService:
    """
    Service class for ML model operations including training and prediction
    """
    def __init__(self, config_path: str = "params.yml", model_path: str = None):
        """
        Initialize ML service
        
        Args:
            config_path: Path to configuration file
            model_path: Path to model file
        """
        self.model = None
        self.model_path = model_path
        self.config = Config(config_path)
        self.config.validate()
        self.model_info = {}
        self.pipeline = None

    def load_model(self, model_path: str = None) -> None:
        """
        Load ML model from disk
        
        Args:
            model_path: Optional path to model file
        """
        try:
            if model_path is None:
                model_path = self.model_path or self.config.evaluate.get('model_path')
            
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            logger.info(f"Loading model from {model_path}")
            self.pipeline = joblib.load(model_path)
            self.model = self.pipeline  # For backward compatibility
            self.model_path = str(model_path)
            
            self.model_info = {
                "model_type": type(self.pipeline.named_steps.get('model', self.pipeline)).__name__,
                "model_version": "1.0.0",
                "model_path": str(model_path),
                "loaded_at": pd.Timestamp.now().isoformat()
            }

            logger.info("Model loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.pipeline is not None

    def train_model(self, override_model_output: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the training pipeline
        
        Args:
            override_model_output: Optional path to save trained model
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE")
            logger.info("=" * 60)

            raw_path = self.config.data.get('raw_path')
            if raw_path is None:
                raise ValueError("raw_path not set in config.data")
            
            logger.info(f"Loading raw data from: {raw_path}")
            df_raw = load_data(raw_path)

            logger.info("Fitting DataPreprocessor to raw data (to obtain encoded target)")
            preprocessor = DataPreprocessor(self.config)
            df_processed = preprocessor.preprocess(df_raw.copy())
            
            target_col = self.config.preprocess.get("target")
            if target_col is None or target_col not in df_processed.columns:
                raise ValueError("Target column missing after preprocessing")
            
            y = df_processed[target_col].values

            pre_wrapper = PreprocessorWrapper(
                config=self.config, 
                fitted_preprocessor=preprocessor
            )

            trainer = ModelTrainer(self.config)
            base_model = trainer.initialize_model()

            self.pipeline = Pipeline(steps=[
                ("preprocess", pre_wrapper),
                ("model", base_model)
            ])

            test_size = self.config.data.get('test_size', 0.2)
            val_size = self.config.data.get('val_size', 0.1)
            random_state = self.config.data.get('random_state', 42)

            logger.info("Splitting data for train/test (stratified on encoded target)")
            x_train_raw, x_test_raw, y_train, y_test = train_test_split(
                df_raw,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )

            if val_size and val_size > 0:
                val_ratio = val_size / (1 - test_size)
                x_train_raw, x_val_raw, y_train, y_val = train_test_split(
                    x_train_raw,
                    y_train,
                    test_size=val_ratio,
                    random_state=random_state,
                    stratify=y_train
                )
            else:
                x_val_raw = None
                y_val = None
            
            logger.info("Training pipeline...")
            with Timer("Pipeline fit"):
                self.pipeline.fit(x_train_raw, y_train)
            
            logger.info("Evaluating model")
            metrics = {}
            
            train_score = self.pipeline.score(x_train_raw, y_train)
            logger.info(f"Training Score: {train_score:.4f}")
            metrics['train_score'] = train_score

            if x_val_raw is not None:
                val_score = self.pipeline.score(x_val_raw, y_val)
                logger.info(f"Validation Score: {val_score:.4f}")
                metrics['val_score'] = val_score
            
            test_score = self.pipeline.score(x_test_raw, y_test)
            logger.info(f"Test Score: {test_score:.4f}")
            metrics['test_score'] = test_score
            
            try:
                cv = self.config.train.get('cv', 5)
                if cv and cv > 1:
                    logger.info(f"Running cross-validation (cv={cv})")
                    cv_scores = cross_val_score(self.pipeline, df_raw, y, cv=cv)
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    logger.info(f"CV Mean: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
                    metrics['cv_mean'] = cv_mean
                    metrics['cv_std'] = cv_std
            except Exception as e:
                logger.warning(f"Cross-validation failed: {e}")

            model_output = override_model_output or self.config.evaluate.get('model_path')
            if model_output is None:
                raise ValueError("evaluate.model_path must be set in config or provided as override")    

            model_output_path = Path(model_output)
            model_output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.pipeline, model_output_path)
            logger.info(f"Pipeline saved to: {model_output_path}")
            
            self.model_path = str(model_output_path)
            self.model = self.pipeline

            try:
                feature_names = pre_wrapper.feature_names_
                if feature_names:
                    features_path = model_output_path.with_suffix(".features.txt")
                    with open(features_path, 'w') as f:
                        for feat in feature_names:
                            f.write(f"{feat}\n")
                    logger.info(f"Feature names saved to: {features_path}")
                    metrics['feature_count'] = len(feature_names)
            except Exception:
                logger.debug("Could not save features list")

            self.model_info = {
                "model_type": type(base_model).__name__,
                "model_version": "1.0.0",
                "model_path": str(model_output_path),
                "trained_at": pd.Timestamp.now().isoformat(),
                "metrics": metrics
            }

            logger.info("\nTRAINING PIPELINE COMPLETED SUCCESSFULLY")
            return metrics
            
        except Exception as e:
            logger.error("\nTRAINING PIPELINE FAILED")
            logger.error(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def predict(self, data: pd.DataFrame):
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")

        try:
            df = data.copy()

            # optional: drop id if not used in training
            if "customer_id" in df.columns:
                df = df.drop(columns=["customer_id"])

            preds = self.model.predict(df)
            probs = self.model.predict_proba(df)

            return preds, probs

        except Exception as e:
            logger.exception("Prediction failed")
            raise

    
    def predict_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make prediction for a single sample
        
        Args:
            data: Dictionary containing feature values
            
        Returns:
            Dictionary with prediction and probability
        """
        df = pd.DataFrame([data])
        predictions, probabilities = self.predict(df)
        
        return {
            "prediction": int(predictions[0]),
            "probability": float(probabilities[0][1]), 
            "probabilities": probabilities[0].tolist()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_model_loaded():
            return {
                "status": "not_loaded",
                "model_type": "Not loaded",
                "model_version": "N/A",
                "features": [],
                "trained_at": None,
                "metrics": {}
            }
        
        features = []
        try:
            if hasattr(self.pipeline.named_steps['preprocess'], 'feature_names_'):
                features = self.pipeline.named_steps['preprocess'].feature_names_
        except Exception:
            pass
        
        return {
            "status": "loaded",
            "model_type": self.model_info.get("model_type", "Unknown"),
            "model_version": self.model_info.get("model_version", "1.0.0"),
            "model_path": self.model_path,
            "features": features,
            "trained_at": self.model_info.get("trained_at"),
            "metrics": self.model_info.get("metrics", {})
        }
    
    def evaluate(self, test_data_path: str = None) -> Dict[str, float]:
        """
        Evaluate model on test data
        
        Args:
            test_data_path: Optional path to test data
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded. Call load_model() or train_model() first.")
        
        try:
            if test_data_path is None:
                test_data_path = self.config.data.get('raw_path')
            
            logger.info(f"Loading test data from: {test_data_path}")
            df_raw = load_data(test_data_path)
            
            preprocessor = DataPreprocessor(self.config)
            df_processed = preprocessor.preprocess(df_raw.copy())
            
            target_col = self.config.preprocess.get("target")
            y_true = df_processed[target_col].values

            predictions, probabilities = self.predict(df_raw)
            
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            metrics = {
                "accuracy": float(accuracy_score(y_true, predictions)),
                "precision": float(precision_score(y_true, predictions, average='weighted')),
                "recall": float(recall_score(y_true, predictions, average='weighted')),
                "f1_score": float(f1_score(y_true, predictions, average='weighted'))
            }
            
            logger.info("Evaluation Metrics:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise


def run_training(config_path: str = "params.yml", override_model_output: Optional[str] = None) -> int:
    """
    Execute the training pipeline (standalone function)
    
    Args:
        config_path: Path to configuration file
        override_model_output: Optional path to save trained model
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        service = MLService(config_path=config_path)
        service.train_model(override_model_output=override_model_output)
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    import sys
    
    service = MLService(config_path="params.yml")
    
    print("Training model...")
    metrics = service.train_model()
    print(f"\nTraining completed with metrics: {metrics}")
    
    info = service.get_model_info()
    print(f"\nModel Info: {info}")
    
    sys.exit(0)