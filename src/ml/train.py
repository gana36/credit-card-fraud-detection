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

    X_train = train_df.drop("Class", axis=1)
    y_train = train_df["Class"]
    X_test = test_df.drop("Class", axis=1)
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
        os.makedirs("models", exist_ok=True)
        model_path = "models/latest.joblib"
        import joblib
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(model_path)
        model_name = os.getenv("MODEL_NAME", "credit-fraud")
        mlflow.sklearn.log_model(pipe, artifact_path="model", registered_model_name=model_name)
        print({"run_id": run.info.run_id, "auc": float(auc), "model_path": model_path})


if __name__ == "__main__":
    train()
