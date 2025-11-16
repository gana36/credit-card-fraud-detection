# Setup Guide for New Features

This guide helps you set up and test the newly added production features.

## 🎯 What Was Added

1. ✅ **Automated Testing** - pytest suite for API and model validation
2. ✅ **Prediction Database** - PostgreSQL for storing predictions
3. ✅ **Production Monitoring** - Track model health and data drift
4. ✅ **Model Validation** - Automated quality checks before promotion
5. ✅ **CI/CD Pipeline** - GitHub Actions for automated testing

---

## 🚀 Quick Setup

### 1. Update Dependencies

```bash
# Activate virtual environment
creditfraud\Scripts\activate

# Install new dependencies
pip install -r requirements.txt
```

### 2. Restart Docker Services

```bash
# Stop existing services
docker-compose -f infra/docker-compose.yaml down

# Start with new PostgreSQL database
docker-compose -f infra/docker-compose.yaml up -d

# Verify all services are running
docker ps
```

You should see 6 containers:
- `postgres` (new!)
- `app`
- `mlflow`
- `prometheus`
- `grafana`

### 3. Verify Database Connection

```bash
# Check PostgreSQL is healthy
docker exec -it postgres pg_isready -U mlops

# Expected output: postgres:5432 - accepting connections
```

---

## 🧪 Test the New Features

### A. Run Automated Tests

```bash
# Run all tests
pytest -v

# Expected: All tests should pass (or skip if no model trained yet)
```

### B. Test Prediction Logging

```bash
# Make a prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.38,
    "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
    "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
    "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
    "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62
  }'

# Check if it was logged to database
docker exec -it postgres psql -U mlops -d predictions -c "SELECT COUNT(*) FROM predictions;"
```

### C. Test Model Validation

```bash
# Train a model first (if not done already)
set MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.ml.train

# Validate the model (use the version number from training output)
python scripts/validate_model.py --version 1

# Expected: Validation report with all checks passing
```

### D. Test Production Monitoring

```bash
# First, make some predictions to have data
# (Use the curl command from step B multiple times)

# Run monitoring
python -m src.monitoring.production_monitor --hours 1

# Expected: Report showing prediction statistics
```

---

## 📊 View Prediction Data

### Option 1: Command Line

```bash
# Connect to PostgreSQL
docker exec -it postgres psql -U mlops -d predictions

# View recent predictions
SELECT
    timestamp,
    fraud_probability,
    prediction,
    model_version,
    latency_ms
FROM predictions
ORDER BY timestamp DESC
LIMIT 10;

# Exit
\q
```

### Option 2: Database Client

Use any PostgreSQL client:
- **Host**: localhost
- **Port**: 5432
- **Database**: predictions
- **User**: mlops
- **Password**: mlops_password

---

## 🔄 Complete Workflow Test

Test the entire workflow end-to-end:

```bash
# 1. Train a model
set MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.ml.train

# 2. Validate the model (replace 1 with your version)
python scripts/validate_model.py --version 1

# 3. Promote to production with validation
python scripts/validate_model.py --version 1 --auto-promote

# 4. Make predictions
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{...}'

# 5. Monitor production
python -m src.monitoring.production_monitor --hours 1

# 6. Run tests
pytest tests/ -v
```

---

## 🐛 Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL logs
docker logs postgres

# Restart PostgreSQL
docker-compose -f infra/docker-compose.yaml restart postgres

# Check if app can connect
docker logs app | grep -i database
```

### Predictions Not Being Logged

1. Check app logs: `docker logs app`
2. Verify environment variable: `docker exec app env | grep DATABASE_URL`
3. Check if table exists:
   ```bash
   docker exec -it postgres psql -U mlops -d predictions -c "\dt"
   ```

### Tests Failing

```bash
# Make sure virtual environment is activated
creditfraud\Scripts\activate

# Install test dependencies
pip install pytest pytest-cov

# Run tests with verbose output
pytest -vv
```

---

## 📝 What Changed in Files

### Modified Files:
- `src/app/api.py` - Added prediction logging
- `infra/docker-compose.yaml` - Added PostgreSQL service
- `requirements.txt` - Added sqlalchemy, pytest-cov, requests

### New Files:
- `tests/test_api.py` - API endpoint tests
- `tests/test_model.py` - Model validation tests
- `src/database/models.py` - Database models
- `src/monitoring/production_monitor.py` - Production monitoring
- `scripts/validate_model.py` - Model validation script
- `.github/workflows/ci.yaml` - CI/CD pipeline

---

## 🎓 Next Steps

1. **Set up GitHub Actions**: Push to GitHub to enable CI/CD
2. **Create monitoring dashboards**: Set up Grafana dashboards for predictions
3. **Schedule monitoring**: Set up cron job for periodic monitoring
4. **Add alerts**: Configure alerting based on drift/performance

---

## 💡 Tips

- Run `pytest` before every commit
- Use `--auto-promote` for automated deployments
- Check monitoring reports daily in production
- Keep prediction database clean (add retention policy)
- Monitor database size: `docker exec postgres du -sh /var/lib/postgresql/data`

---

## 📚 More Information

- **Testing Guide**: Run `pytest --help`
- **Model Validation**: See `scripts/validate_model.py --help`
- **Production Monitoring**: See `python -m src.monitoring.production_monitor --help`
- **Main README**: See `README.md` for complete documentation
