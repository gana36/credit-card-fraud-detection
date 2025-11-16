"""
Model validation tests
"""
import pytest
import pandas as pd
import os
from sklearn.metrics import roc_auc_score


@pytest.fixture
def test_data():
    """Load test data if available"""
    test_path = "data/processed/test.parquet"
    if not os.path.exists(test_path):
        pytest.skip("Test data not available")
    return pd.read_parquet(test_path)


def test_model_exists():
    """Test that a trained model exists"""
    local_model = "models/latest.joblib"
    if not os.path.exists(local_model):
        pytest.skip("Model not found (expected in local dev environment)")


def test_model_performance(test_data):
    """Test model achieves minimum performance threshold"""
    import joblib

    model_path = "models/latest.joblib"
    if not os.path.exists(model_path):
        pytest.skip("Model not found")

    model = joblib.load(model_path)

    X_test = test_data.drop("Class", axis=1)
    y_test = test_data["Class"]

    predictions = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, predictions)

    # Minimum AUC threshold
    assert auc > 0.90, f"Model AUC {auc:.4f} below threshold 0.90"
    print(f"Model AUC: {auc:.4f}")


def test_model_prediction_range():
    """Test model outputs are in valid range"""
    import joblib

    model_path = "models/latest.joblib"
    if not os.path.exists(model_path):
        pytest.skip("Model not found")

    model = joblib.load(model_path)

    # Get expected feature names from the model
    try:
        expected_features = model.feature_names_in_.tolist()
    except AttributeError:
        # Fallback if model doesn't have feature_names_in_
        expected_features = [f"V{i}" for i in range(1, 29)] + ["Amount"]

    # Create sample with all expected features in correct order
    sample = {feature: 0.0 for feature in expected_features}

    # Add realistic values
    if "Amount" in sample:
        sample["Amount"] = 100.0

    df = pd.DataFrame([sample])
    prediction = model.predict_proba(df)[:, 1][0]

    assert 0 <= prediction <= 1, f"Prediction {prediction} out of range [0, 1]"


def test_data_quality(test_data):
    """Test data quality checks"""
    # No missing values
    assert test_data.isnull().sum().sum() == 0, "Test data contains null values"

    # Minimum rows
    assert len(test_data) > 100, "Test data too small"

    # Expected columns
    expected_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    for col in expected_cols:
        assert col in test_data.columns, f"Missing column: {col}"

    # Class balance
    class_dist = test_data["Class"].value_counts(normalize=True)
    print(f"Class distribution: {class_dist.to_dict()}")
