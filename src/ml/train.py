import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .data import prepare_data

import os, mlflow

# Connect to MLflow server
# Server uses file:///mlruns for both backend and artifacts
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("credit-fraud")
def load_configs():
    import yaml
    with open("configs/base.yaml", "r") as f:
        base = yaml.safe_load(f)
    with open("configs/training.yaml", "r") as f:
        train_cfg = yaml.safe_load(f)
    return base, train_cfg


def train():
    base, cfg = load_configs()
    train_pq, test_pq = prepare_data(test_size=base["test_size"], random_state=base["random_state"])  # idempotent

    train_df = pd.read_parquet(train_pq)
    test_df = pd.read_parquet(test_pq)

    # Drop Time feature - not useful for prediction, only for record-keeping
    X_train = train_df.drop(["Class", "Time"], axis=1, errors='ignore')
    y_train = train_df["Class"]
    X_test = test_df.drop(["Class", "Time"], axis=1, errors='ignore')
    y_test = test_df["Class"]

    steps = []
    if cfg.get("features", {}).get("scale", True):
        steps.append(("scaler", StandardScaler()))

    model_cfg = cfg["model"]
    if model_cfg["type"] == "logistic_regression":
        clf = LogisticRegression(**model_cfg["params"])
    else:
        raise ValueError("Unsupported model type")

    steps.append(("clf", clf))
    pipe = Pipeline(steps)

    mlflow.set_experiment("credit-fraud")
    with mlflow.start_run() as run:
        pipe.fit(X_train, y_train)
        preds = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, preds)
        mlflow.log_metric("auc", float(auc))
        mlflow.log_params({"model": model_cfg["type"], **model_cfg["params"]})

        # Save model locally as backup
        os.makedirs("models", exist_ok=True)
        model_path = "models/latest.joblib"
        import joblib
        joblib.dump(pipe, model_path)

        # Log model to MLflow with artifacts
        model_name = os.getenv("MODEL_NAME", "credit-fraud")

        run_id = run.info.run_id
        experiment_id = run.info.experiment_id

        # IMPORTANT: Log the sklearn model which handles artifacts automatically
        # This creates the model registry entry AND saves artifacts
        model_info = mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=model_name
        )

        # WORKAROUND: When using HTTP tracking, artifacts aren't written to local mlruns
        # We need to manually save them so the Docker containers can access them
        import shutil
        import cloudpickle

        artifacts_dest = os.path.join("mlruns", str(experiment_id), run_id, "artifacts", "model")
        os.makedirs(artifacts_dest, exist_ok=True)

        # Save the model using cloudpickle (MLflow's preferred format)
        model_pkl_path = os.path.join(artifacts_dest, "model.pkl")
        with open(model_pkl_path, 'wb') as f:
            cloudpickle.dump(pipe, f)

        # Create MLmodel metadata file
        import sklearn
        mlmodel_content = f"""artifact_path: model
flavors:
  python_function:
    env:
      conda: conda.yaml
      virtualenv: python_env.yaml
    loader_module: mlflow.sklearn
    model_path: model.pkl
    predict_fn: predict
    python_version: 3.11.0
  sklearn:
    code: null
    pickled_model: model.pkl
    serialization_format: cloudpickle
    sklearn_version: {sklearn.__version__}
mlflow_version: 2.16.2
model_size_bytes: {os.path.getsize(model_pkl_path)}
model_uuid: {run_id}
run_id: {run_id}
utc_time_created: '{int(run.info.start_time)}'
"""
        mlmodel_path = os.path.join(artifacts_dest, "MLmodel")
        with open(mlmodel_path, "w") as f:
            f.write(mlmodel_content)

        print(f"\n✅ Training complete!")
        print(f"Run ID: {run_id}")
        print(f"AUC: {auc:.4f}")
        print(f"Model Path: {model_path}")
        print(f"MLflow Model URI: {model_info.model_uri}")
        print(f"Artifacts saved to: {artifacts_dest}")
        print(f"\nTo promote this model to production:")
        print(f"  python scripts/promote_model.py --version <VERSION> --alias production --reload-app")


if __name__ == "__main__":
    train()


