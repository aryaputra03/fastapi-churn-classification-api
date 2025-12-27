"""
CRUD Operations for Database
"""
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from src.api import schemas
from typing import List, Dict, Any
from datetime import datetime, timedelta

def create_prediction_log(
        db: Session,
        customer_id: str,
        prediction: int,
        probability: float,
        input_data: Dict[str, Any]
) -> schemas.PredictionLog:
    """Create a new prediction log"""
    db_prediction = schemas.PredictionLog(
        customer_id=customer_id,
        prediction=prediction,
        probability=probability,
        input_data=input_data
    )
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    return db_prediction

def get_prediction(
        db: Session,
        skip: int = 0,
        limit: int = 100
) -> List[schemas.PredictionLog]:
    """Get prediction logs with pagination"""
    return db.query(schemas.PredictionLog)\
        .order_by(desc(schemas.PredictionLog.created_at))\
        .offset(skip)\
        .limit(limit)\
        .all()

def get_customer_predictions(
        db: Session,
        customer_id: str
) -> List[schemas.PredictionLog]:
    """Get all predictions for a specific customer"""
    return db.query(schemas.PredictionLog)\
        .filter(schemas.PredictionLog.customer_id == customer_id)\
        .order_by(desc(schemas.PredictionLog.created_at))\
        .all()

def get_prediction_statistics(db: Session) -> Dict[str, Any]:
    """Get prediction statistics"""
    total_predictions = db.query(func.count(schemas.PredictionLog.id)).scalar()

    churn_predictions = db.query(func.count(schemas.PredictionLog.id))\
        .filter(schemas.PredictionLog.prediction == 1)\
        .scalar()
    
    avg_churn_probability = db.query(func.avg(schemas.PredictionLog.probability))\
        .filter(schemas.PredictionLog.prediction == 1)\
        .scalar()
    
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_predictions = db.query(func.count(schemas.PredictionLog.id))\
        .filter(schemas.PredictionLog.created_at >= yesterday)\
        .scalar()
    return{
     "total_predictions": total_predictions or 0,
     "Churn_Predictions" : churn_predictions or 0,
     "No_Churn_Predictions": (total_predictions or 0) - (churn_predictions or 0),
     "Churn_rate" : (churn_predictions / total_predictions) * 100 if total_predictions else 0,
     "avg_Churn_Probability" : float(avg_churn_probability) if avg_churn_probability else 0.0,
     "recent_predictions": recent_predictions or 0
    }

def create_customer(
        db: Session,
        customer_data: Dict[str, Any]
) -> schemas.Customer:
    """Create or update customer"""
    existing = db.query(schemas.Customer)\
        .filter(schemas.Customer.customer_id == customer_data['customer_id'])\
        .first()
    
    if existing:
        for key, value in customer_data.items():
            setattr(existing, key, value)
        db.commit()
        db.refresh(existing)
        return existing
    else:
        db_customer = schemas.Customer(**customer_data)
        db.add(db_customer)
        db.commit()
        db.refresh(db_customer)
        return db_customer

def get_customer(db: Session, customer_id: str) -> schemas.Customer:
    """Get customer by ID"""
    return db.query(schemas.Customer)\
        .filter(schemas.Customer.customer_id == customer_id)\
        .first()

def save_model_metrics(
        db: Session,
        model_version: str,
        metrics: Dict[str, Any]
) -> schemas.ModelMetrics:
    """Save model performance metrics"""
    db_metrics = schemas.ModelMetrics(
        model_version=model_version,
        accuracy=metrics.get("accuracy"),
        precision=metrics.get("precision"),
        recall=metrics.get('recall'),
        f1_score=metrics.get("f1_score"),
        roc_auc=metrics.get("roc_auc"),
        confusion_matrix=metrics.get("confusion_matrix")
    )
    db.add(db_metrics)
    db.commit()
    db.refresh(db_metrics)
    return db_metrics

def get_latest_model_metrics(db: Session) -> schemas.ModelMetrics:
    """Get latest model metrics"""
    return db.query(schemas.ModelMetrics)\
        .order_by(desc(schemas.ModelMetrics.created_at))\
        .first()
