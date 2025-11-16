# 💳 Credit Fraud Detection - Production MLOps Pipeline

A production-ready MLOps pipeline for credit card fraud detection with automated testing, monitoring, and zero-downtime deployments.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.16+-0194E2?style=flat&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=flat&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?style=flat&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?style=flat&logo=grafana&logoColor=white)](https://grafana.com/)

---

## 📋 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Training & Deployment Workflow](#-training--deployment-workflow)
- [Production Features](#-production-features)
- [API Endpoints](#-api-endpoints)
- [Monitoring & Observability](#-monitoring--observability)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features

### 🤖 **Model Training & Registry**
- ✅ Scikit-learn pipeline with StandardScaler + LogisticRegression
- ✅ MLflow experiment tracking and model registry
- ✅ Modern alias-based promotion (`production`, `challenger`, `champion`)
- ✅ DVC pipeline for reproducible training
- ✅ Automated model versioning

### 🚀 **Production Deployment**
- ✅ FastAPI REST API serving predictions
- ✅ Zero-downtime model updates via hot-reload
- ✅ Docker Compose multi-service orchestration
- ✅ Health checks and graceful degradation
- ✅ Prometheus metrics export

### 📊 **Monitoring & Observability**
- ✅ PostgreSQL prediction database with full audit trail
- ✅ Real-time data drift detection with Evidently
- ✅ Production health monitoring dashboard
- ✅ Prometheus + Grafana stack
- ✅ Latency and fraud rate tracking

### 🧪 **Testing & Validation**
- ✅ Automated pytest suite (API + model tests)
- ✅ Pre-deployment model validation
- ✅ Performance threshold checks (AUC > 0.90)
- ✅ CI/CD pipeline with GitHub Actions

### 🔄 **MLOps Best Practices**
- ✅ Automated promotion workflows
- ✅ Model validation before deployment
- ✅ Prediction logging and audit trails
- ✅ Data drift monitoring
- ✅ Reproducible training pipelines

---

## 🏗️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| 🐍 **ML Framework** | ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=flat&logo=scikit-learn&logoColor=white) | Model training and inference |
| 🚀 **API** | ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white) | High-performance REST API |
| 📊 **Experiment Tracking** | ![MLflow](https://img.shields.io/badge/MLflow-2.16-0194E2?style=flat&logo=mlflow&logoColor=white) | Model registry and tracking |
| 🗄️ **Database** | ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15-336791?style=flat&logo=postgresql&logoColor=white) | Prediction audit trail |
| 📈 **Metrics** | ![Prometheus](https://img.shields.io/badge/Prometheus-Latest-E6522C?style=flat&logo=prometheus&logoColor=white) | Metrics collection |
| 📉 **Visualization** | ![Grafana](https://img.shields.io/badge/Grafana-Latest-F46800?style=flat&logo=grafana&logoColor=white) | Dashboards and alerts |
| 🐳 **Containerization** | ![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white) | Service orchestration |
| 🔬 **Drift Detection** | ![Evidently](https://img.shields.io/badge/Evidently-0.4-FF6B6B?style=flat) | Data quality monitoring |
| 🧪 **Testing** | ![pytest](https://img.shields.io/badge/pytest-8.3-0A9EDC?style=flat&logo=pytest&logoColor=white) | Automated testing |
| 📦 **Pipeline** | ![DVC](https://img.shields.io/badge/DVC-3.56-945DD6?style=flat&logo=dvc&logoColor=white) | Data versioning |

---

## 🚀 Quick Start

### 1️⃣ Prerequisites

- ✅ Docker Desktop installed and running
- ✅ Python 3.11+ with virtual environment
- ✅ Git

### 2️⃣ Setup Virtual Environment

```powershell
# Create and activate virtual environment
python -m venv creditfraud
creditfraud\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Start Docker Services

```powershell
# Build and start all services
docker-compose -f infra/docker-compose.yaml up --build -d

# Verify containers are running
docker ps
```

### 4️⃣ Access Services

| Service | URL | Credentials |
|---------|-----|-------------|
| 🚀 **API (FastAPI)** | http://localhost:8000 | - |
| 📊 **MLflow UI** | http://localhost:5000 | - |
| 📈 **Prometheus** | http://localhost:9090 | - |
| 📉 **Grafana** | http://localhost:3000 | admin/admin |
| 🗄️ **PostgreSQL** | localhost:5432 | mlops/mlops_password |
| 📖 **API Docs** | http://localhost:8000/docs | - |

### 5️⃣ Health Check

```powershell
# Check API health
Invoke-RestMethod -Uri "http://localhost:8000/health"
# Expected: {"status":"ok"}

# Check model info
Invoke-RestMethod -Uri "http://localhost:8000/model_info"
```

---

## 🎯 Training & Deployment Workflow

### Step 1: Train a Model

```powershell
# Activate virtual environment
creditfraud\Scripts\activate

# Set MLflow tracking URI
$env:MLFLOW_TRACKING_URI="http://localhost:5000"

# Train the model
python -m src.ml.train
```

**Output:**
```
✅ Training complete!
Run ID: abc123...
AUC: 0.9756
Model Path: models/latest.joblib
MLflow Model URI: runs:/abc123.../model

To promote this model to production:
  python scripts/promote_model.py --version <VERSION> --alias production --reload-app
```

### Step 2: Validate Model

```powershell
# Validate model version 5
python scripts/validate_model.py --version 5

# Validate and auto-promote if it passes
python scripts/validate_model.py --version 5 --auto-promote
```

**Validation Checks:**
- ✅ Model loads successfully
- ✅ AUC > 0.90 threshold
- ✅ Predictions in valid range [0, 1]
- ✅ No NaN predictions
- ✅ Performance vs current production model

### Step 3: List Available Versions

```powershell
python scripts/promote_model.py --list
```

**Output:**
```
Available versions for model 'credit-fraud':
--------------------------------------------------------------------------------
Version    Run ID                              Aliases              Status
--------------------------------------------------------------------------------
5          abc123...                           None                 READY
4          def456...                           None                 READY
3          ghi789...                           production           READY
```

### Step 4: Promote to Production

```powershell
# Promote version 5 with automatic app reload
python scripts/promote_model.py --version 5 --alias production --reload-app
```

**What happens:**
1. ✅ Sets "production" alias to version 5 in MLflow
2. ✅ Calls app's `/reload` endpoint
3. ✅ App loads new model without restart (zero-downtime)

### Step 5: Verify Deployment

```powershell
# Check which model is serving
Invoke-RestMethod -Uri "http://localhost:8000/model_info"
```

**Expected:**
```json
{
  "model": {
    "name": "credit-fraud",
    "alias": "production",
    "version": "5",
    "source": "alias"
  },
  "errors": {
    "alias": null,
    "stage": null
  }
}
```

### Step 6: Make Predictions

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -ContentType "application/json" -Body '{
  "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.38,
  "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
  "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
  "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
  "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
  "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
  "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
  "Amount": 149.62
}'
```

**Response:**
```json
{
  "fraud_probability": 0.1663,
  "prediction": 0,
  "model_version": "5"
}
```

---

## 🎛️ Production Features

### 1. Prediction Database

All predictions are automatically logged to PostgreSQL:

```powershell
# Check prediction count
docker exec -it postgres psql -U mlops -d predictions -c "SELECT COUNT(*) FROM predictions;"

# View recent predictions
docker exec -it postgres psql -U mlops -d predictions -c "
  SELECT timestamp, fraud_probability, prediction, model_version, latency_ms
  FROM predictions
  ORDER BY timestamp DESC
  LIMIT 5;
"

# Analyze by model version
docker exec -it postgres psql -U mlops -d predictions -c "
  SELECT model_version, COUNT(*) as count, AVG(fraud_probability) as avg_prob
  FROM predictions
  GROUP BY model_version;
"
```

**Database Schema:**
```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP,
    features JSON,
    fraud_probability FLOAT,
    prediction INTEGER,
    model_version VARCHAR,
    model_name VARCHAR,
    latency_ms FLOAT
);
```

### 2. Production Monitoring

```powershell
# Monitor last 1 hour
python -m src.monitoring.production_monitor --hours 1

# Monitor last 24 hours
python -m src.monitoring.production_monitor --hours 24
```

**Output:**
```
🔍 Production Monitoring Report - Last 1 hours
============================================================
📊 Total Predictions: 4
🎯 Fraud Rate: 0.00%
📈 Average Fraud Probability: 0.1663
⚡ Average Latency: 128.03 ms

🤖 Model Versions:
  Version 5: 3 predictions (75.0%)

🔬 Running Drift Analysis...
✅ Drift report saved to: reports/production_drift_20251116_112309.html
⚠️  ALERT: Data drift detected!
```

**Monitoring Features:**
- 📈 Prediction volume tracking
- 🎯 Fraud rate trends
- ⚡ API latency analysis
- 🤖 Model version distribution
- 🔬 Data drift detection (Evidently)
- 📊 HTML drift reports
- ⚠️ Automated alerts

### 3. Model Validation

```powershell
python scripts/validate_model.py --version 5
```

**Output:**
```
======================================================================
🔍 MODEL VALIDATION - credit-fraud v5
======================================================================

[1/5] Checking model exists...
  ✓ Model version 5 found
    Run ID: 077d401bfc6c4a7493912db9729d8c09
    Status: READY

[2/5] Loading test data...
  ✓ Loaded 56962 test samples

[3/5] Loading model and computing metrics...
  ✓ AUC Score: 0.9756
    Precision (fraud): 0.8523
    Recall (fraud): 0.7891
    F1-Score (fraud): 0.8193

[4/5] Running validation checks...
  ✓ PASS: AUC above threshold (0.90)
  ✓ PASS: Predictions in valid range
  ✓ PASS: No NaN predictions

[5/5] Comparing with production model...
  Production AUC: 0.9745
  New Model AUC: 0.9756
  Improvement: +0.0011 (+0.11%)
  ✓ PASS: Performance acceptable vs production

======================================================================
✅ VALIDATION PASSED
======================================================================

Model credit-fraud version 5 is ready for promotion!
  • AUC: 0.9756
  • All validation checks passed
```

---

## 🔌 API Endpoints

### Health & Info

```powershell
# Health check
GET http://localhost:8000/health

# Model information
GET http://localhost:8000/model_info

# Prometheus metrics
GET http://localhost:8000/metrics
```

### Model Management

```powershell
# Reload model (after promotion)
POST http://localhost:8000/reload
```

### Predictions

```powershell
# Make prediction
POST http://localhost:8000/predict
Content-Type: application/json

{
  "V1": -1.35,
  "V2": -0.07,
  ...
  "V28": -0.02,
  "Amount": 149.62
}
```

**Interactive API Docs:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 📊 Monitoring & Observability

### Prometheus Metrics

Access: http://localhost:9090

**Available Metrics:**
```
# Request count by endpoint
http_requests_total{method="POST", endpoint="/predict", status="200"}

# Request latency histogram
http_request_duration_seconds{endpoint="/predict"}

# Request latency summary
http_request_duration_seconds_sum
http_request_duration_seconds_count
```

**Sample Queries:**
```promql
# Request rate
rate(http_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"4..|5.."}[5m])
```

### Grafana Dashboards

Access: http://localhost:3000 (admin/admin)

**Setup:**
1. Add Prometheus data source: http://prometheus:9090
2. Create dashboard with panels:
   - Request rate per endpoint
   - P50/P95/P99 latency
   - Error rate
   - Prediction distribution
   - Model version usage

### Drift Detection

```powershell
# Generate drift report
python -m src.monitoring.production_monitor --hours 24
```

**Output:**
- HTML report in `reports/` folder
- Data quality metrics
- Feature drift analysis
- Distribution comparisons
- Alerts for significant drift

---

## 🧪 Testing

### Run All Tests

```powershell
# Activate virtual environment
creditfraud\Scripts\activate

# Run all tests
pytest -v

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

### Test Categories

**1. API Tests** (`tests/test_api.py`)
- ✅ Health endpoint
- ✅ Model info endpoint
- ✅ Prometheus metrics
- ✅ Valid predictions
- ✅ Invalid input handling
- ✅ Model reload

**2. Model Tests** (`tests/test_model.py`)
- ✅ Model exists
- ✅ Performance threshold (AUC > 0.90)
- ✅ Prediction range validation
- ✅ Data quality checks

### CI/CD Pipeline

Tests run automatically via GitHub Actions on:
- Every push to `main` or `develop`
- Every pull request to `main`

**Pipeline Stages:**
1. 🧪 Run pytest with coverage
2. 🎨 Lint with black, flake8, isort
3. 🐳 Build Docker images
4. 📢 Notify on status

---

## 📁 Project Structure

```
CreditFraudDetection/
├── 📁 configs/
│   ├── base.yaml              # Base configuration
│   └── training.yaml          # Training hyperparameters
├── 📁 data/
│   └── processed/             # DVC-managed processed data
│       ├── train.parquet
│       ├── test.parquet
│       └── reference.parquet  # For drift detection
├── 📁 docker/
│   ├── Dockerfile.app         # FastAPI app image
│   └── Dockerfile.mlflow      # MLflow server image
├── 📁 infra/
│   ├── docker-compose.yaml    # Service orchestration
│   └── prometheus/
│       └── prometheus.yml     # Prometheus config
├── 📁 mlflow/
│   └── mlflow.db              # MLflow metadata database
├── 📁 mlruns/                 # MLflow artifacts (shared volume)
├── 📁 models/
│   └── latest.joblib          # Local model backup
├── 📁 reports/                # Drift and monitoring reports
├── 📁 scripts/
│   ├── promote_model.py       # Promote + reload
│   ├── promote_and_restart.py # Promote + restart
│   ├── validate_model.py      # Model validation
│   └── README.md              # Promotion workflow docs
├── 📁 src/
│   ├── app/
│   │   ├── api.py             # FastAPI endpoints
│   │   └── metrics.py         # Prometheus metrics
│   ├── database/
│   │   └── models.py          # PostgreSQL models
│   ├── ml/
│   │   ├── data.py            # Data preparation
│   │   ├── train.py           # Model training
│   │   └── evaluate.py        # Model evaluation
│   └── monitoring/
│       ├── drift_job.py       # Evidently drift detection
│       └── production_monitor.py  # Production monitoring
├── 📁 tests/
│   ├── test_api.py            # API endpoint tests
│   └── test_model.py          # Model validation tests
├── 📁 .github/
│   └── workflows/
│       └── ci.yaml            # GitHub Actions CI/CD
├── dvc.yaml                   # DVC pipeline definition
├── dvc.lock                   # DVC pipeline lock file
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🐳 Docker Commands

### Start/Stop Services

```powershell
# Start all services
docker-compose -f infra/docker-compose.yaml up -d

# Stop all services
docker-compose -f infra/docker-compose.yaml down

# Rebuild specific service
docker-compose -f infra/docker-compose.yaml build app
docker-compose -f infra/docker-compose.yaml up -d app

# Restart specific service
docker-compose -f infra/docker-compose.yaml restart app
docker-compose -f infra/docker-compose.yaml restart mlflow

# View logs
docker logs app
docker logs mlflow
docker logs -f app  # Follow logs
```

### Container Management

```powershell
# Check running containers
docker ps

# Execute command in container
docker exec -it app sh
docker exec -it mlflow sh

# Check resource usage
docker stats

# Clean up stopped containers
docker-compose -f infra/docker-compose.yaml down -v
```

---

## 🐛 Troubleshooting

### Model Not Loading

**Check model info:**
```powershell
Invoke-RestMethod -Uri "http://localhost:8000/model_info"
```

If `"source": "local"`, the model failed to load from MLflow. Check:
1. MLflow is running: `docker ps | findstr mlflow`
2. Artifacts exist: `dir mlruns\<experiment-id>\<run-id>\artifacts\model`
3. Reload the app: `Invoke-RestMethod -Uri "http://localhost:8000/reload" -Method Post`

### Containers Won't Start

```powershell
# Check logs
docker logs mlflow
docker logs app

# Common fixes
docker-compose -f infra/docker-compose.yaml down
docker-compose -f infra/docker-compose.yaml up --build -d
```

### Database Connection Issues

```powershell
# Check PostgreSQL logs
docker logs postgres

# Restart PostgreSQL
docker-compose -f infra/docker-compose.yaml restart postgres

# Check if app can connect
docker logs app | findstr -i database
```

### Predictions Not Being Logged

1. Check app logs: `docker logs app`
2. Verify environment variable: `docker exec app printenv DATABASE_URL`
3. Check if table exists:
   ```powershell
   docker exec -it postgres psql -U mlops -d predictions -c "\dt"
   ```

### Tests Failing

```powershell
# Make sure virtual environment is activated
creditfraud\Scripts\activate

# Install test dependencies
pip install pytest pytest-cov

# Run tests with verbose output
pytest -vv
```

### Training Fails

```powershell
# Make sure virtual environment is activated
creditfraud\Scripts\activate

# Set MLflow URI
$env:MLFLOW_TRACKING_URI="http://localhost:5000"

# Train again
python -m src.ml.train
```

---

## 📚 Key Learnings

### MLflow Modern Practices

✅ **DO**: Use aliases (`production`, `challenger`, `champion`)
❌ **DON'T**: Use deprecated stages (`Staging`, `Production`)

### Model Promotion Workflow

1. **Train** → Creates versioned model
2. **Validate** → Check performance thresholds
3. **Promote** → Set alias to version
4. **Reload/Restart** → App serves new model

### Zero-Downtime Deployment

Use `/reload` endpoint for instant model updates without container restart.

### Feature Engineering

- **Time** feature excluded from training (only for record-keeping)
- Features: V1-V28 (PCA-transformed) + Amount
- StandardScaler applied before LogisticRegression

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and test: `pytest -v`
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Submit pull request
