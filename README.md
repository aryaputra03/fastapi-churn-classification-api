# 🐳 Customer Churn Classification - Complete MLOps Project

[![CI/CD Pipeline](https://github.com/yourusername/churn-classification-mlops/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/yourusername/churn-classification-mlops/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![DVC](https://img.shields.io/badge/data-dvc-9cf.svg)](https://dvc.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready MLOps project for customer churn prediction featuring Docker containerization, DVC data versioning, and automated CI/CD with GitHub Actions.

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Docker Workflows](#-docker-workflows)
- [DVC Integration](#-dvc-integration)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Configuration](#-configuration)
- [Development](#-development)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### Core Features
- 🤖 **Machine Learning Pipeline**: Complete end-to-end churn prediction workflow
- 🐳 **Docker Containerization**: Fully containerized with multi-stage builds
- 📊 **DVC Integration**: Data and model versioning with DVC
- 🔄 **CI/CD Automation**: Automated testing and deployment with GitHub Actions
- 🧪 **Comprehensive Testing**: Unit tests with pytest and coverage reporting
- 📈 **Metrics Tracking**: Automated evaluation and performance monitoring
- 🔧 **Configurable**: YAML-based configuration for easy experimentation

### Technical Stack
- **ML Framework**: Scikit-learn
- **Containerization**: Docker, Docker Compose
- **Data Versioning**: DVC (supports S3, GDrive, local)
- **CI/CD**: GitHub Actions
- **Testing**: Pytest, Coverage
- **Linting**: Ruff, Black, Flake8
- **Notebook**: Jupyter

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
│   └── predict.py                     # Inference/prediction
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py
│   ├── test_train.py
│   └── test_evaluate.py
│
├── docker/
│   ├── Dockerfile                     # Production Docker image
│   ├── Dockerfile.dev                 # Development Docker image
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
- **Make** (for using Makefile commands)

### Windows Users
- **Docker Desktop** with WSL2 backend
- **WSL2** (Ubuntu 20.04 or later recommended)

Installation on Windows:
```powershell
# Install Docker Desktop from official website
# Enable WSL2 backend in Docker Desktop settings

# Verify installation
docker --version
docker-compose --version
wsl --version
```

---

## 🚀 Quick Start

### Option 1: Using Make (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/yourusername/churn-classification-mlops.git
cd churn-classification-mlops

# 2. Install dependencies
make install-dev

# 3. Setup project
make setup

# 4. Generate sample data
make data

# 5. Run complete pipeline
make pipeline

# 6. View results
cat metrics.json
```

### Option 2: Using Scripts

```bash
# 1. Clone repository
git clone https://github.com/yourusername/churn-classification-mlops.git
cd churn-classification-mlops

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements-dev.txt

# 4. Make scripts executable
chmod +x scripts/*.sh

# 5. Run pipeline
./scripts/run_pipeline.sh
```

### Option 3: Using Docker

```bash
# 1. Clone repository
git clone https://github.com/yourusername/churn-classification-mlops.git
cd churn-classification-mlops

# 2. Build Docker image
./scripts/docker_build.sh

# 3. Run pipeline in container
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/params.yaml:/app/params.yaml \
  churn-classifier:latest
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

# 3. Install development dependencies
pip install -r requirements-dev.txt

# 4. Verify installation
python -c "import src; print(src.get_version())"
pytest --version
docker --version
```

### Docker Setup

```bash
# Build production image
docker build -t churn-classifier:latest -f docker/Dockerfile .

# Build development image
docker build -t churn-classifier:dev -f docker/Dockerfile.dev .

# Or use the build script
./scripts/docker_build.sh
./scripts/docker_build.sh --dev
```

### DVC Setup

```bash
# Initialize DVC
dvc init

# Add remote storage (choose one):

# Local storage (for testing)
dvc remote add -d myremote /tmp/dvc-storage

# Amazon S3
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Google Drive
dvc remote add -d myremote gdrive://your-folder-id

# Or use setup script
./scripts/setup_dvc.sh
```

---

## 📚 Usage

### Running Individual Steps

#### 1. Data Preprocessing
```bash
# Local
python -m src.preprocess --config params.yaml

# Docker
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/params.yaml:/app/params.yaml \
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
  -v $(pwd)/params.yaml:/app/params.yaml \
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
  -v $(pwd)/params.yaml:/app/params.yaml \
  -v $(pwd)/metrics.json:/app/metrics.json \
  churn-classifier:latest \
  python -m src.evaluate

# Make
make evaluate
```

#### 4. Making Predictions
```bash
# Local
python -m src.predict \
  --input data/new_customers.csv \
  --output predictions.csv

# Docker
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  churn-classifier:latest \
  python -m src.predict --input /app/data/new_customers.csv

# Make
make predict INPUT=data/new_customers.csv
```

### Running Complete Pipeline

```bash
# Option 1: Using script
./scripts/run_pipeline.sh

# Option 2: Using Make
make pipeline

# Option 3: Using DVC
dvc repro

# Option 4: Using Docker Compose
docker-compose -f docker/docker-compose.yml up ml-pipeline-full
```

---

## 🐳 Docker Workflows

### Development Workflow (Inside Container Approach)

Best for experimentation and development:

```bash
# Start development container with live code mounting
docker run -it --rm \
  -v $(pwd):/app \
  -w /app \
  -p 8888:8888 \
  churn-classifier:dev \
  bash

# Inside container:
dvc repro
pytest tests/
jupyter notebook --ip=0.0.0.0 --allow-root
```

### Production Workflow (Outside Container Approach)

Best for automated pipelines:

```bash
# Run specific stages with Docker
python -m src.train  # Uses Docker internally via DVC pipeline
dvc repro  # Orchestrates Docker containers for each stage
```

### Docker Compose Multi-Service

```bash
# Start all services
docker-compose -f docker/docker-compose.yml up

# Start specific service
docker-compose -f docker/docker-compose.yml up ml-training
docker-compose -f docker/docker-compose.yml up ml-dev
docker-compose -f docker/docker-compose.yml up dvc-pipeline

# Stop all services
docker-compose -f docker/docker-compose.yml down

# Or use helper script
./scripts/docker_compose_run.sh
```

### Jupyter Notebook in Docker

```bash
# Start Jupyter
docker-compose -f docker/docker-compose.yml up ml-dev

# Access at: http://localhost:8888

# Or manually:
docker run -it --rm \
  -v $(pwd):/app \
  -p 8888:8888 \
  churn-classifier:dev \
  jupyter notebook --ip=0.0.0.0 --allow-root
```

---

## 📊 DVC Integration

### Basic DVC Commands

```bash
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

### DVC + Docker Workflows

#### Inside Container (Development)
```bash
# Mount workspace into container
docker run -v $(pwd):/app -w /app -it churn-classifier:dev dvc repro
```

**Pros:**
- Live code changes
- No git clone needed
- Results automatically on host

**Use case:** Experimentation, development

#### Outside Container (Production)
```yaml
# dvc.yaml stage with Docker
stages:
  train:
    cmd: docker run churn-classifier python -m src.train
    deps:
      - data/processed/
    outs:
      - models/model.pkl
```

**Pros:**
- Managed by DVC
- Automatic image updates
- Production-ready

**Use case:** Production pipelines, CI/CD

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The CI/CD pipeline automatically runs on push and pull requests:

```
┌─────────────┐
│ Push/PR     │
└──────┬──────┘
       │
       ├─► Lint (Ruff, Black, Flake8)
       │        │
       │        ✓
       │
       ├─► Test (Python 3.9, 3.10, 3.11)
       │        │
       │        ├─► Unit Tests
       │        ├─► Coverage Report
       │        └─► Upload to Codecov
       │        │
       │        ✓
       │
       ├─► DVC Pipeline (main branch only)
       │        │
       │        ├─► Pull data
       │        ├─► Run pipeline in Docker
       │        └─► Push results
       │        │
       │        ✓
       │
       ├─► Docker Build & Push
       │        │
       │        ├─► Build production image
       │        ├─► Build dev image
       │        └─► Push to registry
       │        │
       │        ✓
       │
       ├─► Integration Tests
       │        │
       │        ├─► Test preprocessing
       │        ├─► Test training
       │        └─► Test evaluation
       │        │
       │        ✓
       │
       └─► Status Check
                │
                └─► ✅ Success / ❌ Failure
```

### Setup GitHub Secrets

Required secrets for CI/CD:

```
DOCKER_USERNAME: your-dockerhub-username
DOCKER_PASSWORD: your-dockerhub-token
AWS_ACCESS_KEY_ID: your-aws-key (if using S3)
AWS_SECRET_ACCESS_KEY: your-aws-secret (if using S3)
```

Add secrets:
1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add each secret

### Manual Workflow Trigger

Trigger workflow manually from GitHub UI:
1. Go to Actions tab
2. Select "ML CI/CD Pipeline"
3. Click "Run workflow"

---

## ⚙️ Configuration

### Modify Hyperparameters

Edit `params.yaml`:

```yaml
train:
  model_type: random_forest
  n_estimators: 100          # Change to 200
  max_depth: 10              # Change to 15
  min_samples_split: 5
  random_state: 42
```

Then rerun:
```bash
dvc repro
# or
make pipeline
```

### Change Data Paths

```yaml
data:
  raw_path: data/raw/my_custom_data.csv
  processed_path: data/processed/my_processed_data.csv
```

### Add New Features

```yaml
preprocess:
  numerical_features:
    - tenure
    - MonthlyCharges
    - TotalCharges
    - my_new_feature  # Add your feature
  
  feature_engineering:
    create_tenure_bins: true
    create_charge_ratio: true
    my_new_transformation: true  # Add your transformation
```

---

## 🔧 Development

### Local Development Setup

```bash
# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Install in editable mode
pip install -e .

# Run linting
make lint

# Format code
make format

# Run tests
make test
```

### Adding New Features

```bash
# 1. Create feature branch
git checkout -b feature/new-awesome-feature

# 2. Implement feature
# Edit files...

# 3. Add tests
# Create tests/test_new_feature.py

# 4. Run tests
make test

# 5. Update DVC pipeline if needed
# Edit dvc.yaml

# 6. Test pipeline
dvc repro

# 7. Commit changes
git add .
git commit -m "feat: add awesome new feature"

# 8. Push and create PR
git push origin feature/new-awesome-feature
```

### Code Quality

```bash
# Run all quality checks
make lint

# Auto-format code
make format

# Type checking (if mypy configured)
mypy src/

# Security check
bandit -r src/
```

---

## 🧪 Testing

### Run All Tests

```bash
# Using pytest
pytest tests/ -v

# Using Make
make test

# With coverage
make test-coverage

# In Docker
docker run --rm churn-classifier:dev pytest tests/ -v
```

### Run Specific Tests

```bash
# Test specific file
pytest tests/test_train.py -v

# Test specific function
pytest tests/test_train.py::test_model_training -v

# Test with keyword
pytest tests/ -k "test_preprocess" -v
```

### Coverage Report

```bash
# Terminal report
pytest tests/ --cov=src --cov-report=term-missing

# HTML report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Using Make
make test-coverage
```

---

## 🚀 Deployment

### Deploy to Production Server

```bash
# 1. Pull Docker image
docker pull yourusername/churn-classifier:latest

# 2. Run container
docker run -d \
  --name churn-api \
  -v /path/to/data:/app/data \
  -v /path/to/models:/app/models \
  -p 8080:8080 \
  yourusername/churn-classifier:latest

# 3. Check logs
docker logs churn-api

# 4. Test endpoint
curl http://localhost:8080/predict -X POST -d @sample.json
```

### Deploy to Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: churn-classifier
spec:
  replicas: 3
  selector:
    matchLabels:
      app: churn-classifier
  template:
    metadata:
      labels:
        app: churn-classifier
    spec:
      containers:
      - name: churn-classifier
        image: yourusername/churn-classifier:latest
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: models
          mountPath: /app/models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
```

Apply:
```bash
kubectl apply -f k8s-deployment.yaml
```

---

## 🐛 Troubleshooting

### Common Issues

#### Docker Permission Denied (Linux)
```bash
sudo usermod -aG docker $USER
newgrp docker
```

#### WSL2 Memory Issues (Windows)
Create `.wslconfig` in Windows user directory:
```ini
[wsl2]
memory=8GB
processors=4
```

Then restart WSL:
```powershell
wsl --shutdown
```

#### DVC Remote Connection Issues
```bash
# Check DVC config
dvc remote list
dvc config -l

# Test connection
dvc pull --verbose

# Re-configure remote
dvc remote modify myremote --local access_key_id YOUR_KEY
dvc remote modify myremote --local secret_access_key YOUR_SECRET
```

#### Module Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=$(pwd):$PYTHONPATH

# Or install package in editable mode
pip install -e .
```

#### Docker Build Fails
```bash
# Clear Docker cache
docker system prune -a

# Build without cache
docker build --no-cache -t churn-classifier:latest -f docker/Dockerfile .
```

## 📖 Additional Resources

### Documentation
- [Docker Documentation](https://docs.docker.com/)
- [DVC Documentation](https://dvc.org/doc)
- [GitHub Actions](https://docs.github.com/actions)
- [Scikit-learn](https://scikit-learn.org/)

### Tutorials
- [MLOps Best Practices](https://ml-ops.org/)
- [Docker for Data Science](https://www.docker.com/blog/tag/data-science/)
- [DVC with Docker Tutorial](https://dvc.org/doc/use-cases/versioning-data-and-model-files)

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

### Quick Contribution Guide

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Stavanger** - *Initial work* - [@aryaputra03](https://github.com/aryaputra03)

---

## 🙏 Acknowledgments

- Thanks to the open-source community
- Scikit-learn team for the ML framework
- Docker team for containerization technology
- DVC team for data versioning tools
- GitHub for Actions CI/CD platform

---

## 📞 Contact

- Email: Aryaganendra45@gmail.com
- GitHub: [@aryaputra03](https://github.com/aryaputra03)
- LinkedIn: [aryaputra](https://www.linkedin.com/in/ganendra-geanza-aryaputra-b8071a194)
- Project Link: [https://github.com/aryaputra03/Docker_Churn_Classifier](https://github.com/aryaputra03/Docker_Churn_Classifier)

---

<p align="center">
  Made with ❤️ for MLOps enthusiasts
</p>

<p align="center">
  ⭐ Star this repo if you find it helpful!
</p>