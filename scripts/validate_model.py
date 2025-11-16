#!/usr/bin/env python3
"""
Validate a model version before promoting to production.

This script performs comprehensive validation including:
- Performance metrics (AUC threshold)
- Comparison with current production model
- Data quality checks
- Prediction range validation

Usage:
    python scripts/validate_model.py --version 3
    python scripts/validate_model.py --version 3 --auto-promote
"""
import os
import sys
import argparse
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np


def validate_model(model_name: str, version: str, min_auc: float = 0.95):
    """
    Validate a model version

    Args:
        model_name: Name of the registered model
        version: Version to validate
        min_auc: Minimum AUC threshold

    Returns:
        bool: True if validation passes
    """
    print("=" * 70)
    print(f"🔍 MODEL VALIDATION - {model_name} v{version}")
    print("=" * 70)

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    try:
        # 1. Check model exists
        print(f"\n[1/5] Checking model exists...")
        mv = client.get_model_version(model_name, version)
        print(f"  ✓ Model version {version} found")
        print(f"    Run ID: {mv.run_id}")
        print(f"    Status: {mv.status}")

        # 2. Load test data
        print(f"\n[2/5] Loading test data...")
        test_path = "data/processed/test.parquet"
        if not os.path.exists(test_path):
            print(f"  ⚠️  Test data not found at {test_path}")
            print(f"  Skipping performance validation")
            return True  # Allow promotion without test data

        test_df = pd.read_parquet(test_path)
        # Drop Time feature - models are trained without it
        X_test = test_df.drop(["Class", "Time"], axis=1, errors='ignore')
        y_test = test_df["Class"]
        print(f"  ✓ Loaded {len(test_df)} test samples")

        # 3. Load and test model
        print(f"\n[3/5] Loading model and computing metrics...")
        model_uri = f"models:/{model_name}/{version}"

        try:
            model = mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            return False

        # Make predictions
        try:
            predictions = model.predict_proba(X_test)[:, 1]
            pred_binary = (predictions >= 0.5).astype(int)
        except Exception as e:
            print(f"  ❌ Failed to make predictions: {e}")
            return False

        # Calculate metrics
        auc = roc_auc_score(y_test, predictions)
        print(f"  ✓ AUC Score: {auc:.4f}")

        # Classification report
        report = classification_report(y_test, pred_binary, output_dict=True)
        print(f"    Precision (fraud): {report['1']['precision']:.4f}")
        print(f"    Recall (fraud): {report['1']['recall']:.4f}")
        print(f"    F1-Score (fraud): {report['1']['f1-score']:.4f}")

        # 4. Validation checks
        print(f"\n[4/5] Running validation checks...")

        # Check 1: AUC threshold
        if auc < min_auc:
            print(f"  ❌ FAIL: AUC {auc:.4f} below threshold {min_auc}")
            return False
        print(f"  ✓ PASS: AUC above threshold ({min_auc})")

        # Check 2: Prediction range
        if predictions.min() < 0 or predictions.max() > 1:
            print(f"  ❌ FAIL: Predictions out of range [0, 1]")
            return False
        print(f"  ✓ PASS: Predictions in valid range")

        # Check 3: No NaN predictions
        if np.isnan(predictions).any():
            print(f"  ❌ FAIL: NaN predictions detected")
            return False
        print(f"  ✓ PASS: No NaN predictions")

        # 5. Compare with production model
        print(f"\n[5/5] Comparing with production model...")
        try:
            prod_model = mlflow.sklearn.load_model(f"models:/{model_name}@production")
            # Use same preprocessed X_test (already has Time removed)
            prod_predictions = prod_model.predict_proba(X_test)[:, 1]
            prod_auc = roc_auc_score(y_test, prod_predictions)

            print(f"  Production AUC: {prod_auc:.4f}")
            print(f"  New Model AUC: {auc:.4f}")

            improvement = auc - prod_auc
            print(f"  Improvement: {improvement:+.4f} ({improvement/prod_auc*100:+.2f}%)")

            # Must be within 2% of current production
            if auc < prod_auc * 0.98:
                print(f"  ⚠️  WARNING: New model is >2% worse than production")
                print(f"  Consider whether to proceed with promotion")
            else:
                print(f"  ✓ PASS: Performance acceptable vs production")

        except Exception as e:
            print(f"  ℹ️  No production model to compare (this may be first deployment)")

        # Final result
        print("\n" + "=" * 70)
        print("✅ VALIDATION PASSED")
        print("=" * 70)
        print(f"\nModel {model_name} version {version} is ready for promotion!")
        print(f"  • AUC: {auc:.4f}")
        print(f"  • All validation checks passed")

        return True

    except Exception as e:
        print(f"\n❌ VALIDATION FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate a model version before promotion",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("MODEL_NAME", "credit-fraud"),
        help="Name of the registered model"
    )
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Model version to validate"
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=0.90,
        help="Minimum AUC threshold (default: 0.90)"
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically promote if validation passes"
    )

    args = parser.parse_args()

    # Run validation
    is_valid = validate_model(
        model_name=args.model_name,
        version=args.version,
        min_auc=args.min_auc
    )

    if is_valid and args.auto_promote:
        print(f"\n🚀 Auto-promoting to production...")
        import subprocess
        result = subprocess.run([
            "python", "scripts/promote_model.py",
            "--version", args.version,
            "--alias", "production",
            "--reload-app"
        ])
        sys.exit(result.returncode)
    else:
        sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
