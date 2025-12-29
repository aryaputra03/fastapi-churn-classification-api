"""
FastAPI Main Application

REST API for customer churn prediction with database logging.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import List
import pandas as pd
from datetime import datetime
import io

from src.api.models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionHistoryResponse,
)

from src.api.database import get_db, engine
from src.api import crud
from src.api.ml_service import MLService
from src.utils import logger
from sqlalchemy.orm import Session

from src.api.database import Base
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Churn Prediciton API",
    description="ML API for customer churn prediction with tracking",
    version="1.0.0",
    docs_url="/docs",
    redocs_url="/redocs",        
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ml_service = MLService()

# ==========================================
# Health & Info Endpoints
# ==========================================
@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        model_loaded = ml_service.is_model_loaded()

        return HealthResponse(
            status='health' if model_loaded else 'unhealthy',
            model_loaded=model_loaded,
            timestamp=datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            timestamp=datetime.utcnow(),
            error=str(e)
        )

@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    try:
        info = ml_service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Prediction Endpoints
# ==========================================
@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    background_task: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Predict churn for a single customer
    
    Example request:
    ```json
    {
        "customer_id": "C12345",
        "gender": "Male",
        "tenure": 24,
        "monthly_charges": 75.5,
        "total_charges": 1810.0,
        "contract": "One year",
        "payment_method": "Bank transfer",
        "internet_service": "Fiber optic"
    }
    ```
    """
    try:
        input_data = pd.DataFrame([request.dict()])
        prediction, probability = ml_service.predict(input_data)

        response = PredictionResponse(
            customer_id=request.customer_id,
            prediction=int(prediction[0]),
            churn_probability=float(probability[0][1]),
            no_churn_probability=float(probability[0][0]),
            timestamp=datetime.utcnow()
        )
    
        background_task.add_task(
            crud.create_prediction_log,
            db=db,
            customer_id=request.customer_id,
            prediction=response.prediction,
            probability=response.churn_probability,
            input_data=request.dict()
        )
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    request: BatchPredictionRequest,
    background_task: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Predict churn for multiple customers
    
    Example request:
    ```json
    {
        "customers": [
            {
                "customer_id": "C001",
                "gender": "Male",
                "tenure": 24,
                ...
            },
            {
                "customer_id": "C002",
                "gender": "Female",
                "tenure": 12,
                ...
            }
        ]
    }
    ```
    """
    try:
        customer_data = [customer.dict() for customer in request.customers]
        input_data = pd.DataFrame(customer_data)

        prediction, probabilities = ml_service.predict(input_data)

        responses = []
        for i, customer in enumerate(request.customers):
            response = PredictionResponse(
                customer_id=customer.customer_id,
                prediction=int(prediction[i]),
                churn_probability=float(probabilities[i][1]),
                no_churn_probability=float(probabilities[i][0]),
                timestamp=datetime.utcnow()
            )
            responses.append(response)

            background_task.add_task(
                crud.create_prediction_log,
                db=db,
                customer_id=customer.customer_id,
                prediction=response.prediction,
                probability=response.churn_probability,
                input_data=customer.dict()
            )

        return response
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("predict/csv")
async def predict_csv(
    file: UploadFile = File(...),
    background_task: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Predict churn from CSV file upload
    
    CSV should contain columns:
    - customer_id, gender, tenure, monthly_charges, total_charges,
      contract, payment_method, internet_service
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        required_col = [
            'customer_id', 'gender', 'tenure', 'monthly_charges',
            'total_charges', 'contract', 'payment_method', 'internet_service'
        ]
        missing_cols = set(required_col) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        predictions, probabilities = ml_service.predict(df)

        df['prediciton'] = predictions
        df['churn_probability'] = probabilities[:, 1]
        df['no_churn_probability'] = probabilities[:, 0]

        results = df.to_dict('records')

        return JSONResponse(content={
            "total_predictions": len(results),
            "predictions": results
        })
    
    except Exception as e:
        logger.error(f"CSV prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# ==========================================
# History & Analytics Endpoints
# ==========================================
@app.get("/predictions/history", response_model=List[PredictionHistoryResponse])
async def get_prediction_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(0, le=100, ge=1),
    db: Session = Depends(get_db)
):
    """Get prediction history from database"""
    try:
        predictions = crud.get_prediction(db, skip=skip, limit=limit)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/predictions/customer/{customer_id}", response_model=List[PredictionHistoryResponse])
async def get_customer_prediction(
    customer_id: str,
    db: Session = get_db
):
    """Get prediction history for specific customer"""
    try:
        predictions = crud.get_customer_predictions(db, customer_id)
        if not predictions:
            raise HTTPException(status_code=404, detail="Customer not found")
        return predictions
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("analytics/summary")
async def get_analytics_summary(
    db: Session = Depends(get_db)
):
    """Get prediction analytics summary"""
    try:
        stats = crud.get_prediction_statistics(db)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# Model Management Endpoints
# ==========================================
@app.post("/model/reload")
async def reload_model():
    try:
        ml_service.load_model()
        return {"message":"Model Reload Sucessfuly", "timestamp":datetime.utcnow()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

# ==========================================
# Startup & Shutdown Events
# ==========================================
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting Churn Prediction API...")
    try:
        ml_service.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Churn Prediction API...")

# ==========================================
# Main Entry Point
# ==========================================
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level='info'
    )
