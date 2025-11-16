"""
Production monitoring - tracks model performance and data drift
Runs periodically to check production health
"""
import os
import pandas as pd
import mlflow
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import ColumnDriftMetric


def fetch_recent_predictions(hours=24):
    """Fetch predictions from database"""
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql://mlops:mlops_password@localhost:5432/predictions"
    )

    engine = create_engine(DATABASE_URL)

    # Calculate time window
    since = datetime.utcnow() - timedelta(hours=hours)

    query = text("""
        SELECT
            timestamp,
            features,
            fraud_probability,
            prediction,
            model_version,
            latency_ms
        FROM predictions
        WHERE timestamp >= :since
        ORDER BY timestamp DESC
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"since": since})

    return df


def monitor_production(hours=24):
    """
    Monitor production model health

    Args:
        hours: Number of hours of data to analyze
    """
    print(f"🔍 Production Monitoring Report - Last {hours} hours")
    print("=" * 60)

    try:
        # Fetch recent predictions
        predictions_df = fetch_recent_predictions(hours=hours)

        if len(predictions_df) == 0:
            print("⚠️  No predictions found in the last {hours} hours")
            return

        print(f"📊 Total Predictions: {len(predictions_df)}")

        # Extract features from JSON column
        features_df = pd.json_normalize(predictions_df['features'])

        # Basic statistics
        fraud_rate = predictions_df['prediction'].mean()
        avg_proba = predictions_df['fraud_probability'].mean()
        avg_latency = predictions_df['latency_ms'].mean()

        print(f"🎯 Fraud Rate: {fraud_rate:.2%}")
        print(f"📈 Average Fraud Probability: {avg_proba:.4f}")
        print(f"⚡ Average Latency: {avg_latency:.2f} ms")

        # Model version distribution
        version_counts = predictions_df['model_version'].value_counts()
        print(f"\n🤖 Model Versions:")
        for version, count in version_counts.items():
            print(f"  Version {version}: {count} predictions ({count/len(predictions_df):.1%})")

        # Load reference data for drift detection
        reference_path = "data/processed/reference.parquet"
        if os.path.exists(reference_path):
            print(f"\n🔬 Running Drift Analysis...")
            reference_df = pd.read_parquet(reference_path)

            # Drop Time column from reference data (not used in model)
            reference_df = reference_df.drop(columns=['Time'], errors='ignore')

            # Prepare current data (combine features with target)
            current_df = features_df.copy()
            current_df['Class'] = predictions_df['prediction'].values

            # Ensure both dataframes have same columns
            common_columns = list(set(reference_df.columns) & set(current_df.columns))
            reference_df = reference_df[common_columns]
            current_df = current_df[common_columns]

            # Generate drift report
            report = Report(metrics=[
                DataDriftPreset(),
                DataQualityPreset()
            ])

            report.run(
                reference_data=reference_df.sample(min(1000, len(reference_df))),
                current_data=current_df.head(1000)
            )

            # Save report
            os.makedirs("reports", exist_ok=True)
            report_path = f"reports/production_drift_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            report.save_html(report_path)

            print(f"✅ Drift report saved to: {report_path}")

            # Extract drift metrics
            drift_result = report.as_dict()
            dataset_drift = drift_result['metrics'][0]['result']['dataset_drift']

            if dataset_drift:
                print("⚠️  ALERT: Data drift detected!")
            else:
                print("✅ No significant drift detected")

        # Performance alerts
        if avg_latency > 1000:  # > 1 second
            print(f"⚠️  ALERT: High latency detected ({avg_latency:.2f} ms)")

        if fraud_rate > 0.10:  # > 10% fraud rate
            print(f"⚠️  ALERT: High fraud rate detected ({fraud_rate:.2%})")

        print("\n" + "=" * 60)
        print(f"✅ Monitoring complete at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"❌ Error during monitoring: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor production model health")
    parser.add_argument("--hours", type=int, default=24, help="Hours of data to analyze")
    args = parser.parse_args()

    monitor_production(hours=args.hours)
