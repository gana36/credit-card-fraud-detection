import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import joblib


def evaluate(model_path: str = "models/latest.joblib", test_path: str = "data/processed/test.parquet"):
    model = joblib.load(model_path)
    df = pd.read_parquet(test_path)
    X_test = df.drop("Class", axis=1)
    y_test = df["Class"]
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_prob)
    print({"auc": float(auc), "report": report})


if __name__ == "__main__":
    evaluate()
