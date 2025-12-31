# 🐳 Customer Churn Classification - Complete MLOps Project

[![CI/CD Pipeline](https://github.com/yourusername/churn-classification-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/yourusername/churn-classification-mlops/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![DVC](https://img.shields.io/badge/data-dvc-9cf.svg)](https://dvc.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-00a393.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready MLOps project for customer churn prediction featuring Docker containerization, DVC data versioning, FastAPI REST API, database integration, and automated CI/CD with GitHub Actions.

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [ML Pipeline Usage](#-ml-pipeline-usage)
- [API Usage](#-api-usage)
- [Docker Workflows](#-docker-workflows)
- [DVC Integration](#-dvc-integration)
- [Database Management](#-database-management)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Monitoring & Analytics](#-monitoring--analytics)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## ✨ Features

### Core ML Features
- 🤖 **Complete ML Pipeline**: End-to-end churn prediction workflow
- 🐳 **Docker Containerization**: Fully containerized with multi-stage builds
- 📊 **DVC Integration**: Data and model versioning with DVC
- 🔄 **CI/CD Automation**: Automated testing and deployment with GitHub Actions
- 🧪 **Comprehensive Testing**: Unit tests with pytest and coverage reporting
- 📈 **Metrics Tracking**: Automated evaluation and performance monitoring
- 🔧 **Configurable**: YAML-based configuration for easy experimentation

### API Features
- 🚀 **REST API**: FastAPI-based prediction service
- 💾 **Database Integration**: SQLAlchemy with SQLite/PostgreSQL support
- 📝 **Request Validation**: Pydantic models for data validation
- 📚 **Interactive Docs**: Automatic Swagger UI & ReDoc generation
- 📊 **Prediction Logging**: Complete history tracking in database
- 📈 **Real-time Analytics**: Statistics and performance monitoring
- 🔄 **Batch Processing**: Support for multiple predictions
- 📁 **CSV Upload**: File-based batch predictions

### Technical Stack
- **ML Framework**: Scikit-learn
- **API Framework**: FastAPI, Uvicorn
- **Database**: SQLAlchemy (SQLite, PostgreSQL)
- **Containerization**: Docker, Docker Compose
- **Data Versioning**: DVC (supports S3, GDrive, local)
- **CI/CD**: GitHub Actions
- **Testing**: Pytest, Coverage
- **Linting**: Ruff, Black, Flake8

---

## 📁 Project Structure

```
churn-classification-mlops/
├── .github/
│   └── workflows/
│       ├── ci-cd.yml                  # Main CI/CD pipeline
│       └── docker-publish.yml         # Docker image publishing
│
├── data/
│   ├── raw/                           # Raw data (tracked by DVC)
│   ├── processed/                     # Processed data (tracked by DVC)
│   └── README.md
│
├── models/                            # Trained models (tracked by DVC)
│   └── churn_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── config.py                      # Configuration management
│   ├── utils.py                       # Utility functions
│   ├── preprocess.py                  # Data preprocessing
│   ├── train.py                       # Model training
│   ├── evaluate.py                    # Model evaluation
│   ├── predict.py                     # Inference/prediction
│   │
│   └── api/                           # FastAPI application
│       ├── __init__.py
│       ├── main.py                    # API entry point
│       ├── models.py                  # Pydantic models
│       ├── database.py                # Database configuration
│       ├── crud.py                    # Database operations
│       └── schemas.py                 # Request/Response schemas
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   └── test_api.py                    # API endpoint tests
│
├── docker/
│   ├── Dockerfile                     # Production Docker image
│   ├── Dockerfile.dev                 # Development Docker image
│   ├── Dockerfile.api                 # API-specific Docker image
│   └── docker-compose.yml             # Multi-container orchestration
│
├── scripts/
│   ├── setup_dvc.sh                   # DVC initialization script
│   ├── run_pipeline.sh                # Complete pipeline runner
│   ├── docker_build.sh                # Docker build helper
│   └── docker_compose_run.sh          # Docker Compose helper
│
├── plots/                             # Visualization outputs
│   └── confusion_matrix.json
│
├── .dvc/                              # DVC configuration
│   └── config
│
├── dvc.yaml                           # DVC pipeline definition
├── dvc.lock                           # DVC pipeline lock file
├── params.yaml                        # Hyperparameters & config
├── metrics.json                       # Model metrics output
│
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Development dependencies
├── setup.py                           # Package setup
├── Makefile                           # Automation commands
├── .gitignore
├── .dockerignore
├── .dvcignore
└── README.md
```

---

## 🛠️ Prerequisites

### Required
- **Python** 3.9 or higher
- **Docker** 20.10 or higher
- **Docker Compose** 2.0 or higher
- **Git** 2.30 or higher

### Optional
- **DVC** 3.30 or higher (for data versioning)
- **PostgreSQL** 13+ (for production database)
- **Make** (for using Makefile commands)

### Windows Users
- **Docker Desktop** with WSL2 backend
- **WSL2** (Ubuntu 20.04 or later recommended)

---

## 🚀 Quick Start

### Complete Setup (ML + API)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/churn-classification-mlops.git
cd churn-classification-mlops

# 2. Install all dependencies
make install-dev

# 3. Setup project (initialize DVC, database, etc.)
make setup

# 4. Generate sample data
make data

# 5. Run ML pipeline
make pipeline

# 6. Start API server
make api-run

# 7. Access API documentation
# Open browser: http://localhost:8000/docs
```

### Docker Quick Start

```bash
# Start all services (ML + API + Database)
docker-compose up

# Access:
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Jupyter: http://localhost:8888
```

---

## 💻 Installation

### Local Development Setup

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install all dependencies (including API)
pip install -r requirements-dev.txt
pip install fastapi uvicorn sqlalchemy pydantic

# 4. Verify installation
python -c "import src; import fastapi; print('✓ Installation successful')"
pytest --version
docker --version
```

### Docker Setup

```bash
# Build all images
make docker-build-all

# Or build individually:
docker build -t churn-classifier:latest -f docker/Dockerfile .
docker build -t churn-classifier:dev -f docker/Dockerfile.dev .
docker build -t churn-api:latest -f docker/Dockerfile.api .
```

### Database Setup

```bash
# Initialize SQLite database (default)
make db-init

# Or manually:
python -c "from src.api.database import init_db; init_db()"

# For PostgreSQL (production):
export DATABASE_URL="postgresql://user:password@localhost/churn_db"
make db-init
```

---

## 📚 ML Pipeline Usage

### Running Individual Steps

#### 1. Data Preprocessing
```bash
# Local
python -m src.preprocess --config params.yaml

# Docker
docker run --rm \
  -v $(pwd)/data:/app/data \
  churn-classifier:latest \
  python -m src.preprocess

# Make
make preprocess
```

#### 2. Model Training
```bash
# Local
python -m src.train --config params.yaml

# Docker
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  churn-classifier:latest \
  python -m src.train

# Make
make train
```

#### 3. Model Evaluation
```bash
# Local
python -m src.evaluate --config params.yaml

# Docker
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  churn-classifier:latest \
  python -m src.evaluate

# Make
make evaluate
```

### Running Complete Pipeline

```bash
# Option 1: Using Make
make pipeline

# Option 2: Using DVC
dvc repro

# Option 3: Using Docker Compose
docker-compose up ml-pipeline-full

# Option 4: Using script
./scripts/run_pipeline.sh
```

---

## 🚀 API Usage

### Starting the API Server

```bash
# Development mode (auto-reload)
uvicorn src.api.main:app --reload

# Production mode
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# Using Make
make api-run

# Using Docker
docker-compose up api
```

### API Endpoints Overview

#### Health & Information
```bash
# Root endpoint
curl http://localhost:8000/

# Health check
curl http://localhost:8000/health

# Model information
curl http://localhost:8000/model/info
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST001",
    "gender": "Female",
    "tenure": 12,
    "monthly_charges": 70.35,
    "total_charges": 844.20,
    "contract": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic"
  }'

# Response:
{
  "customer_id": "CUST001",
  "prediction": 1,
  "churn_label": "Will Churn",
  "probability": 0.78,
  "confidence": "high",
  "prediction_id": 1
}
```

#### Batch Predictions
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [
      {
        "customer_id": "CUST001",
        "gender": "Female",
        "tenure": 12,
        "monthly_charges": 70.35,
        "total_charges": 844.20,
        "contract": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic"
      },
      {
        "customer_id": "CUST002",
        "gender": "Male",
        "tenure": 48,
        "monthly_charges": 45.50,
        "total_charges": 2184.00,
        "contract": "Two year",
        "payment_method": "Bank transfer",
        "internet_service": "DSL"
      }
    ]
  }'
```

#### CSV Upload
```bash
curl -X POST "http://localhost:8000/predict/csv" \
  -F "file=@customers.csv"
```

#### Prediction History
```bash
# All predictions
curl http://localhost:8000/predictions/history?limit=10

# Customer-specific predictions
curl http://localhost:8000/predictions/customer/CUST001

# Analytics summary
curl http://localhost:8000/analytics/summary
```

### Interactive API Documentation

Visit these URLs when the API is running:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Both provide interactive documentation where you can test all endpoints directly in your browser.

---

## 🐳 Docker Workflows

### Development Workflow

```bash
# Start development environment
docker-compose up ml-dev

# Access Jupyter notebook
# URL: http://localhost:8888

# Run pipeline inside container
docker exec -it churn-ml-dev dvc repro
```

### Production Workflow

```bash
# Build production images
make docker-build-all

# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Multi-Service Setup

```yaml
# docker-compose.yml includes:
services:
  postgres:      # PostgreSQL database
  api:           # API with SQLite
  api-postgres:  # API with PostgreSQL
  ml-training:   # ML training service
  ml-dev:        # Development environment
  dvc-pipeline:  # DVC pipeline runner
```

---

## 📊 DVC Integration

### Basic DVC Commands

```bash
# Initialize DVC
dvc init

# Add remote storage
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Pull data from remote
dvc pull

# Run pipeline
dvc repro

# Push data/models to remote
dvc push

# Show pipeline DAG
dvc dag

# Show metrics
dvc metrics show

# Compare experiments
dvc metrics diff
```

### Track New Data

```bash
# Add data file
dvc add data/raw/new_data.csv

# Commit DVC file
git add data/raw/new_data.csv.dvc .gitignore
git commit -m "Add new data"

# Push to remote
dvc push
```

---

## 💾 Database Management

### Database Schema

The API uses these main tables:

#### PredictionLog
Stores all prediction requests and results
```sql
- id: Primary key
- customer_id: Customer identifier
- prediction: Churn prediction (0/1)
- probability: Churn probability
- input_data: JSON of input features
- created_at: Timestamp
```

#### Customer
Stores customer information
```sql
- id: Primary key
- customer_id: Unique customer ID
- gender, tenure, monthly_charges, etc.
- is_active: Customer status
- created_at: Registration timestamp
```

#### ModelMetrics
Tracks model performance over time
```sql
- id: Primary key
- model_version: Model identifier
- accuracy, precision, recall, f1_score
- confusion_matrix: JSON format
- created_at: Evaluation timestamp
```

### Database Operations

```bash
# Initialize database
make db-init

# Reset database
make db-reset

# View database stats
sqlite3 churn_predictions.db "SELECT COUNT(*) FROM prediction_logs;"

# Export predictions
sqlite3 churn_predictions.db ".mode csv" ".output predictions.csv" "SELECT * FROM prediction_logs;"
```

### PostgreSQL Setup (Production)

```bash
# Start PostgreSQL container
docker-compose up postgres

# Connect to database
docker exec -it churn-postgres psql -U churn_user -d churn_db

# Run API with PostgreSQL
export DATABASE_URL="postgresql://churn_user:churn_password@localhost:5432/churn_db"
uvicorn src.api.main:app --reload
```

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The automated pipeline runs on push and pull requests:

```
Push/PR → Lint → Test → DVC Pipeline → Docker Build → Integration Tests → Deploy
```

### Pipeline Stages

1. **Lint**: Code quality checks (Ruff, Black, Flake8)
2. **Test**: Unit tests with coverage (Python 3.9, 3.10, 3.11)
3. **DVC Pipeline**: Run ML pipeline in Docker
4. **Docker Build**: Build and push images
5. **Integration Tests**: End-to-end testing
6. **API Tests**: Test all endpoints

### Setup GitHub Secrets

Required secrets:
```
DOCKER_USERNAME: your-dockerhub-username
DOCKER_PASSWORD: your-dockerhub-token
AWS_ACCESS_KEY_ID: your-aws-key (if using S3)
AWS_SECRET_ACCESS_KEY: your-aws-secret (if using S3)
DATABASE_URL: production database URL
```

---

## ⚙️ Configuration

### params.yaml Structure

```yaml
data:
  raw_path: data/raw/churn_data.csv
  processed_path: data/processed/churn_processed.csv

preprocess:
  test_size: 0.2
  random_state: 42
  numerical_features:
    - tenure
    - MonthlyCharges
    - TotalCharges
  categorical_features:
    - gender
    - Contract
    - PaymentMethod

train:
  model_type: random_forest
  n_estimators: 100
  max_depth: 10
  min_samples_split: 5
  random_state: 42

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  reload: false

database:
  type: sqlite  # or postgresql
  url: sqlite:///churn_predictions.db
```

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Database
DATABASE_URL=sqlite:///churn_predictions.db
# DATABASE_URL=postgresql://user:pass@host:5432/db

# Model
MODEL_PATH=models/churn_model.pkl

# DVC
DVC_REMOTE_URL=s3://my-bucket/dvc-storage
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
```

---

## 🧪 Testing

### Run All Tests

```bash
# All tests
make test

# ML pipeline tests only
pytest tests/test_*.py -v --ignore=tests/test_api.py

# API tests only
make api-test

# With coverage
make test-coverage
```

### Test Specific Components

```bash
# Test preprocessing
pytest tests/test_preprocess.py -v

# Test API endpoints
pytest tests/test_api.py -v

# Test with keyword
pytest tests/ -k "test_predict" -v
```

### Coverage Report

```bash
# Terminal report
pytest tests/ --cov=src --cov-report=term-missing

# HTML report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## 🚀 Deployment

### Local Production Deployment

```bash
# Using Gunicorn + Uvicorn workers
gunicorn src.api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

### Docker Production Deployment

```bash
# Pull latest image
docker pull yourusername/churn-api:latest

# Run container
docker run -d \
  --name churn-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -e DATABASE_URL=postgresql://user:pass@host/db \
  yourusername/churn-api:latest

# Check logs
docker logs churn-api -f

# Health check
curl http://localhost:8000/health
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-api
  template:
    metadata:
      labels:
        app: churn-api
    spec:
      containers:
      - name: churn-api
        image: yourusername/churn-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
        volumeMounts:
        - name: models
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: churn-api-service
spec:
  selector:
    app: churn-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Apply:
```bash
kubectl apply -f k8s-deployment.yaml
kubectl get pods
kubectl get services
```

---

## 📈 Monitoring & Analytics

### Application Metrics

```bash
# Get API analytics
curl http://localhost:8000/analytics/summary

# Response:
{
  "total_predictions": 1250,
  "total_customers": 450,
  "churn_rate": 24.96,
  "recent_predictions_24h": 45,
  "avg_churn_probability": 0.32,
  "model_version": "v1.0.0"
}
```

### Database Monitoring

```bash
# Check prediction volume
sqlite3 churn_predictions.db \
  "SELECT DATE(created_at), COUNT(*) 
   FROM prediction_logs 
   GROUP BY DATE(created_at);"

# Check churn rate
sqlite3 churn_predictions.db \
  "SELECT 
    AVG(prediction) * 100 as churn_rate,
    AVG(probability) as avg_probability
   FROM prediction_logs;"
```

### Logging

```bash
# View application logs
tail -f logs/api.log

# View Docker logs
docker logs churn-api -f

# View access logs
tail -f logs/access.log
```

---

## 🐛 Troubleshooting

### Common Issues & Solutions

#### Model Not Loading
```bash
# Check model file
ls -lh models/churn_model.pkl

# Reload model via API
curl -X POST http://localhost:8000/model/reload

# Retrain model
make train
```

#### Database Errors
```bash
# Reset database
make db-reset

# Check database connection
python -c "from src.api.database import get_db; next(get_db())"

# Check tables
sqlite3 churn_predictions.db ".tables"
```

#### Docker Issues
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker-compose build --no-cache

# Check container logs
docker-compose logs api
```

#### Port Already in Use
```bash
# Find process using port 8000
lsof -ti:8000

# Kill process
kill -9 $(lsof -ti:8000)

# Or use different port
uvicorn src.api.main:app --port 8001
```

#### API Connection Refused
```bash
# Check if API is running
curl http://localhost:8000/health

# Check Docker network
docker network ls
docker network inspect churn-classification-mlops_default

# Restart services
docker-compose restart api
```

#### DVC Remote Issues
```bash
# Check DVC config
dvc remote list
dvc config -l

# Test connection
dvc pull --verbose

# Re-configure remote
dvc remote modify myremote --local url s3://new-bucket
```

---

## 🔒 Security Best Practices

### Production Checklist

- [ ] Use HTTPS/TLS certificates
- [ ] Implement API authentication (JWT/OAuth2)
- [ ] Add rate limiting
- [ ] Configure CORS properly
- [ ] Use PostgreSQL with strong credentials
- [ ] Store secrets in environment variables
- [ ] Enable database connection pooling
- [ ] Implement request logging
- [ ] Set up monitoring alerts
- [ ] Regular security updates

### Example Security Configuration

```python
# In src/api/main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(data: CustomerData):
    # ... prediction logic
    pass
```

---

## 📖 Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [DVC Documentation](https://dvc.org/doc)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

### Tutorials
- [MLOps Best Practices](https://ml-ops.org/)
- [FastAPI Full Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [Docker for Data Science](https://www.docker.com/blog/tag/data-science/)
- [DVC with Docker](https://dvc.org/doc/use-cases/versioning-data-and-model-files)

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests for new features
5. Ensure all tests pass (`make test`)
6. Commit your changes (`git commit -m 'Add AmazingFeature'`)
7. Push to the branch (`git push origin feature/AmazingFeature`)
8. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Ganendra Geanza Aryaputra (Stavanger)**

- GitHub: [@aryaputra03](https://github.com/aryaputra03)
- LinkedIn: [aryaputra](https://www.linkedin.com/in/ganendra-geanza-aryaputra-b8071a194)
- Email: Aryaganendra45@gmail.com

---

## 🙏 Acknowledgments

- Scikit-learn team for the excellent ML framework
- FastAPI team for the amazing web framework
- Docker team for containerization technology
- DVC team for data versioning tools
- GitHub for Actions CI/CD platform
- The open-source community

---

## 📞 Contact & Support

- **Email**: Aryaganendra45@gmail.com
- **GitHub**: [@aryaputra03](https://github.com/aryaputra03)
- **LinkedIn**: [Ganendra Geanza Aryaputra](https://www.linkedin.com/in/ganendra-geanza-aryaputra-b8071a194)
- **Project Repository**: [Docker_Churn_Classifier](https://github.com/aryaputra03/Docker_Churn_Classifier)
- **Issues**: [GitHub Issues](https://github.com/aryaputra03/Docker_Churn_Classifier/issues)

---

## 🎯 Project Roadmap

### Completed ✅
- Complete ML pipeline with DVC
- Docker containerization
- FastAPI REST API
- Database integration
- CI/CD with GitHub Actions
- Comprehensive testing
- Interactive API documentation

### In Progress 🚧
- Advanced monitoring dashboard
- Model A/B testing framework
- Real-time prediction streaming

### Planned 📋
- Web UI dashboard
- Advanced analytics features
- Multi-model ensemble support
- Automated retraining pipeline
- Cloud deployment templates (AWS, GCP, Azure)

---

<p align="center">
  <strong>Made with ❤️ for MLOps enthusiasts</strong>
</p>

<p align="center">
  ⭐ Star this repo if you find it helpful!
</p>

<p align="center">
  <a href="#-table-of-contents">Back to Top ↑</a>
</p>