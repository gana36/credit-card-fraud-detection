"""
Test script to verify MLflow artifact logging works correctly.
This will help diagnose where artifacts are being saved.
"""
import os
import mlflow
import tempfile

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test-artifacts")

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment: {mlflow.get_experiment_by_name('test-artifacts')}")

# Create a simple test artifact
with mlflow.start_run() as run:
    # Log a simple text file
    test_file = "test_artifact.txt"
    with open(test_file, "w") as f:
        f.write("This is a test artifact")

    mlflow.log_artifact(test_file)
    mlflow.log_param("test_param", "value")
    mlflow.log_metric("test_metric", 0.95)

    print(f"\nRun ID: {run.info.run_id}")
    print(f"Artifact URI: {run.info.artifact_uri}")
    print(f"Experiment ID: {run.info.experiment_id}")

    # Clean up local test file
    os.remove(test_file)

print("\n" + "="*60)
print("Test complete! Now check:")
print(f"1. MLflow UI: http://localhost:5000")
print(f"2. Local mlruns folder")
print(f"3. Container mlruns: docker exec mlflow ls -la /mlruns/")
print("="*60)
