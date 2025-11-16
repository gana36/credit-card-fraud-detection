import os
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response, JSONResponse
from .metrics import REQUEST_COUNT, REQUEST_LATENCY
import joblib
import time
from datetime import datetime

app = FastAPI(title="Credit Fraud API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.getenv("MODEL_PATH", "models/latest.joblib")
MODEL_NAME = os.getenv("MODEL_NAME", "credit-fraud")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "production")
MODEL_STAGE = os.getenv("MODEL_STAGE", "Production")
_model = None
_model_info = {
    "source": None,  # alias | stage | local
    "name": None,
    "alias": None,
    "stage": None,
    "version": None,
    "errors": {
        "alias": None,
        "stage": None,
    },
}

# Configure MLflow tracking (works both in and out of containers)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Initialize database on startup
try:
    from src.database.models import init_db, get_db, Prediction
    init_db()
    DB_ENABLED = True
except Exception as e:
    print(f"Database not available: {e}")
    DB_ENABLED = False

def load_model():
    global _model
    global _model_info
    if _model is not None:
        return _model
    # Try Model Registry by alias first (preferred), then stage (deprecated UI), else local file
    try:
        if MODEL_ALIAS:
            uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
            _model = mlflow.sklearn.load_model(uri)
            try:
                client = MlflowClient()
                mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
                _model_info.update({
                    "source": "alias",
                    "name": MODEL_NAME,
                    "alias": MODEL_ALIAS,
                    "stage": None,
                    "version": mv.version,
                })
            except Exception:
                _model_info.update({"source": "alias", "name": MODEL_NAME, "alias": MODEL_ALIAS})
            return _model
    except Exception as e:
        _model_info["errors"]["alias"] = str(e)
    try:
        if MODEL_STAGE:
            uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
            _model = mlflow.sklearn.load_model(uri)
            try:
                client = MlflowClient()
                latest = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
                ver = latest[0].version if latest else None
                _model_info.update({
                    "source": "stage",
                    "name": MODEL_NAME,
                    "alias": None,
                    "stage": MODEL_STAGE,
                    "version": ver,
                })
            except Exception:
                _model_info.update({"source": "stage", "name": MODEL_NAME, "stage": MODEL_STAGE})
            return _model
    except Exception as e:
        _model_info["errors"]["stage"] = str(e)
        # Fallback to local file artifact
        if os.path.exists(MODEL_PATH):
            _model = joblib.load(MODEL_PATH)
            _model_info.update({
                "source": "local",
                "name": None,
                "alias": None,
                "stage": None,
                "version": None,
            })
            return _model
        return None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/model_info")
async def model_info():
    # Ensure model is loaded to populate info
    if _model is None:
        load_model()
    return {
        "model": {
            "name": _model_info.get("name"),
            "alias": _model_info.get("alias"),
            "stage": _model_info.get("stage"),
            "version": _model_info.get("version"),
            "source": _model_info.get("source"),
        },
        "errors": _model_info.get("errors"),
    }

@app.post("/reload")
async def reload_model():
    """Reload the model from MLflow registry. Use this after promoting a new model version."""
    global _model
    global _model_info

    # Clear the cached model
    _model = None
    _model_info = {
        "source": None,
        "name": None,
        "alias": None,
        "stage": None,
        "version": None,
        "errors": {
            "alias": None,
            "stage": None,
        },
    }

    # Load the new model
    try:
        model = load_model()
        if model is None:
            return JSONResponse(
                {"status": "error", "message": "Failed to load model", "model_info": _model_info},
                status_code=503
            )
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model": {
                "name": _model_info.get("name"),
                "alias": _model_info.get("alias"),
                "stage": _model_info.get("stage"),
                "version": _model_info.get("version"),
                "source": _model_info.get("source"),
            }
        }
    except Exception as e:
        return JSONResponse(
            {"status": "error", "message": str(e), "model_info": _model_info},
            status_code=500
        )
@app.post("/predict")
async def predict(payload: dict, db=Depends(get_db) if DB_ENABLED else None):
    start_time = time.time()

    with REQUEST_LATENCY.labels("/predict").time():
        model = load_model()
        if model is None:
            REQUEST_COUNT.labels("POST", "/predict", 503).inc()
            return JSONResponse({"error": "model not available"}, status_code=503)
        try:
            # Expect a flat dict of feature_name: value
            import numpy as np
            import pandas as pd
            X = pd.DataFrame([payload])
            proba = float(model.predict_proba(X)[:, 1][0])
            prediction = 1 if proba >= 0.5 else 0

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Store prediction in database
            if DB_ENABLED and db is not None:
                try:
                    db_prediction = Prediction(
                        timestamp=datetime.utcnow(),
                        features=payload,
                        fraud_probability=proba,
                        prediction=prediction,
                        model_version=_model_info.get("version"),
                        model_name=_model_info.get("name"),
                        latency_ms=latency_ms
                    )
                    db.add(db_prediction)
                    db.commit()
                except Exception as db_error:
                    print(f"Failed to log prediction to database: {db_error}")
                    db.rollback()

            REQUEST_COUNT.labels("POST", "/predict", 200).inc()
            return {
                "fraud_probability": proba,
                "prediction": prediction,
                "model_version": _model_info.get("version")
            }
        except Exception as e:
            REQUEST_COUNT.labels("POST", "/predict", 400).inc()
            return JSONResponse({"error": str(e)}, status_code=400)
