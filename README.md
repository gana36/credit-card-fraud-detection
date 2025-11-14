# Credit Fraud Detection - Local MLOps (Minimal)

## Stack
- Docker Compose: app (FastAPI), MLflow, Prometheus, Grafana
- DVC: data + pipeline (prepare -> train -> evaluate)
- Evidently: drift report

## Quickstart
1. Python deps (optional if using Docker only)
   - `pip install -r requirements.txt`
2. DVC init and track data
   - `dvc init`
   - `dvc repro` (runs prepare -> train -> evaluate)
3. Run local services
   - `docker compose -f infra/docker-compose.yaml up --build`
   - App: http://localhost:8000/health
   - Metrics: http://localhost:8000/metrics
   - MLflow: http://localhost:5000
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)
4. Generate drift report
   - `python -m src.monitoring.drift_job`

## Notes
- The API loads `models/latest.joblib` if present.
- MLflow uses local SQLite and `./mlruns` by default.
- S3/Cloud configs are deferred to later steps.
