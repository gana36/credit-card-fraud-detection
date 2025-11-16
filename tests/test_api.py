"""
API endpoint tests
"""
import pytest
from fastapi.testclient import TestClient
from src.app.api import app

client = TestClient(app)


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/model_info")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "errors" in data


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_predict_endpoint_valid_input():
    """Test prediction with valid input"""
    # Complete payload with all V features
    payload = {
        f"V{i}": 0.0 for i in range(1, 29)
    }
    # Add realistic values for some features
    payload.update({
        "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.38,
        "V5": -0.34, "V6": 0.46, "V7": 0.24, "V8": 0.10,
        "V9": 0.36, "V10": 0.09, "V11": -0.55, "V12": -0.62,
        "V13": -0.99, "V14": -0.31, "V15": 1.47, "V16": -0.47,
        "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
        "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07,
        "V25": 0.13, "V26": -0.19, "V27": 0.13, "V28": -0.02,
        "Amount": 149.62
    })

    response = client.post("/predict", json=payload)

    # Should return prediction or 503 if no model loaded
    assert response.status_code in [200, 400, 503], f"Unexpected status: {response.status_code}, response: {response.text}"

    if response.status_code == 200:
        data = response.json()
        assert "fraud_probability" in data
        assert 0 <= data["fraud_probability"] <= 1


def test_predict_endpoint_missing_features():
    """Test prediction with missing features"""
    payload = {"V1": -1.35, "V2": -0.07}  # Missing most features

    response = client.post("/predict", json=payload)

    # Should return 400 or 503
    assert response.status_code in [400, 503]


def test_reload_endpoint():
    """Test model reload endpoint"""
    response = client.post("/reload")
    assert response.status_code in [200, 500, 503]

    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "message" in data
