# Model Promotion Scripts

These scripts help you promote models to production using MLflow's modern **alias-based** system (replacing the deprecated stage-based system).

## Background

MLflow deprecated the stage-based promotion system (Staging/Production). The new approach uses **aliases** like `production`, `challenger`, `champion`, etc.

## Available Scripts

### 1. `promote_model.py` - Promote & Reload (Recommended)

Promotes a model and reloads it in the running app via the `/reload` endpoint (no restart required).

**Usage:**
```bash
# List all available versions
python scripts/promote_model.py --list

# Promote version 5 to production
python scripts/promote_model.py --version 5 --alias production

# Promote and automatically reload the app
python scripts/promote_model.py --version 5 --alias production --reload-app
```

**Advantages:**
- No downtime
- Fast reload
- Works with running containers

**Requirements:**
- App must be running
- `/reload` endpoint must be accessible

### 2. `promote_and_restart.py` - Promote & Restart Container

Promotes a model and restarts the Docker container (ensures fresh start).

**Usage:**
```bash
# List all available versions
python scripts/promote_and_restart.py --list

# Promote version 5 to production and restart container
python scripts/promote_and_restart.py --version 5 --alias production
```

**Advantages:**
- Guarantees clean state
- Good for major updates

**Disadvantages:**
- Brief downtime during restart

## Complete Workflow

### Step 1: Train a Model

```bash
python src/ml/train.py
```

This will:
- Train a new model
- Log it to MLflow
- Register it with auto-incrementing version number

### Step 2: Check Available Versions

```bash
python scripts/promote_model.py --list
```

Output:
```
Available versions for model 'credit-fraud':
--------------------------------------------------------------------------------
Version    Run ID                              Aliases              Status
--------------------------------------------------------------------------------
5          abc123def456                        None                 READY
4          def456ghi789                        production           READY
3          ghi789jkl012                        None                 READY
```

### Step 3: Promote to Production

**Option A: Using reload endpoint (recommended)**
```bash
python scripts/promote_model.py --version 5 --alias production --reload-app
```

**Option B: Using container restart**
```bash
python scripts/promote_and_restart.py --version 5 --alias production
```

### Step 4: Verify the Model

```bash
# Check loaded model info
curl http://localhost:8000/model_info

# Expected output:
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

### Step 5: Test Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.35,
    "V2": -0.07,
    "V3": 2.53,
    ...
  }'
```

## Common Aliases

- `production` - Currently serving model
- `challenger` - Candidate for A/B testing
- `champion` - Best performing model
- `baseline` - Reference model for comparison

You can use any alias name that makes sense for your workflow.

## Troubleshooting

### App won't reload

If `--reload-app` fails, manually restart:
```bash
docker-compose -f infra/docker-compose.yaml restart app
```

### Can't connect to MLflow

Make sure MLflow is running:
```bash
docker-compose -f infra/docker-compose.yaml ps mlflow
```

Set the tracking URI:
```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

### Model not found

Check if the model is registered:
```bash
python scripts/promote_model.py --list
```

If no versions exist, train a model first:
```bash
python src/ml/train.py
```

## Environment Variables

- `MLFLOW_TRACKING_URI` - MLflow server URL (default: http://localhost:5000)
- `MODEL_NAME` - Registered model name (default: credit-fraud)
- `APP_URL` - App server URL for reload (default: http://localhost:8000)

## Manual Promotion via MLflow UI

You can also promote models through the MLflow web UI:

1. Open http://localhost:5000
2. Go to Models → credit-fraud
3. Click on a version
4. Click "Set alias"
5. Enter alias name (e.g., "production")
6. Reload the app:
   ```bash
   curl -X POST http://localhost:8000/reload
   ```

## CI/CD Integration

These scripts can be integrated into your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Promote model to production
  run: |
    python scripts/promote_model.py \
      --version ${{ steps.train.outputs.version }} \
      --alias production \
      --reload-app \
      --app-url ${{ secrets.PROD_APP_URL }}
```
