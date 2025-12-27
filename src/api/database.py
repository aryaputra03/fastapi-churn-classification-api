"""
Database Configuration with SQLAlchemy
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./churn_predictions.db"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

Sessionlocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    """
    Dependency to get database session
    
    Usage in FastAPI:
        @app.get("/")
        def read_root(db: Session = Depends(get_db)):
            ...
    """
    db = Sessionlocal()
    try:
        yield db
    finally:
        db.close()

