import os
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(raw_csv: str = "creditcard.csv", out_dir: str = "data/processed", test_size: float = 0.2, random_state: int = 42):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(raw_csv)
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    train = X_train.copy()
    train["Class"] = y_train.values
    test = X_test.copy()
    test["Class"] = y_test.values
    train.to_parquet(os.path.join(out_dir, "train.parquet"), index=False)
    test.to_parquet(os.path.join(out_dir, "test.parquet"), index=False)
    # Reference/current for monitoring (simple mapping)
    train.sample(min(len(train), 10000), random_state=random_state).to_parquet(os.path.join(out_dir, "reference.parquet"), index=False)
    test.sample(min(len(test), 5000), random_state=random_state).to_parquet(os.path.join(out_dir, "current.parquet"), index=False)
    return os.path.join(out_dir, "train.parquet"), os.path.join(out_dir, "test.parquet")


if __name__ == "__main__":
    prepare_data()
