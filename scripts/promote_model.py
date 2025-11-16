#!/usr/bin/env python3
"""
Promote a model version to production using MLflow aliases.

MLflow deprecated the stage-based promotion (Staging/Production).
Now we use aliases like 'production', 'challenger', 'champion', etc.

Usage:
    python scripts/promote_model.py --version 5 --alias production
    python scripts/promote_model.py --version 5 --alias production --reload-app
"""
import os
import sys
import argparse
import mlflow
from mlflow.tracking import MlflowClient
import requests


def promote_model(model_name: str, version: str, alias: str, reload_app: bool = False, app_url: str = None):
    """
    Promote a model version by setting an alias in MLflow.

    Args:
        model_name: Name of the registered model
        version: Version number to promote
        alias: Alias to set (e.g., 'production', 'challenger', 'champion')
        reload_app: Whether to trigger app reload after promotion
        app_url: URL of the app server (for reload)
    """
    # Configure MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    client = MlflowClient()

    print(f"Connecting to MLflow at: {mlflow_uri}")
    print(f"Model: {model_name}")
    print(f"Version: {version}")
    print(f"Alias: {alias}")
    print("-" * 50)

    try:
        # Get the model version to verify it exists
        mv = client.get_model_version(model_name, version)
        print(f"Found model version {version}")
        print(f"  Run ID: {mv.run_id}")
        print(f"  Status: {mv.status}")
        print(f"  Current aliases: {mv.aliases if hasattr(mv, 'aliases') else 'N/A'}")

        # Set the alias (this will move the alias from any previous version)
        print(f"\nSetting alias '{alias}' to version {version}...")
        client.set_registered_model_alias(model_name, alias, version)
        print(f"Successfully set alias '{alias}' to version {version}")

        # Verify the alias was set
        mv_updated = client.get_model_version(model_name, version)
        print(f"Updated aliases: {mv_updated.aliases if hasattr(mv_updated, 'aliases') else 'N/A'}")

        # Optionally reload the app
        if reload_app:
            if not app_url:
                app_url = os.getenv("APP_URL", "http://localhost:8000")

            print(f"\nTriggering app reload at {app_url}/reload...")
            try:
                response = requests.post(f"{app_url}/reload", timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    print(f"App reload successful!")
                    print(f"  Status: {result.get('status')}")
                    print(f"  Message: {result.get('message')}")
                    if 'model' in result:
                        print(f"  Loaded model: {result['model']}")
                else:
                    print(f"App reload failed with status code: {response.status_code}")
                    print(f"Response: {response.text}")
                    return False
            except requests.exceptions.RequestException as e:
                print(f"Failed to reload app: {e}")
                print("You may need to restart the app container manually:")
                print("  docker-compose -f infra/docker-compose.yaml restart app")
                return False
        else:
            print("\nNote: App was not reloaded. To serve the new model, either:")
            print(f"  1. Call the reload endpoint: curl -X POST {app_url or 'http://localhost:8000'}/reload")
            print("  2. Restart the app container: docker-compose -f infra/docker-compose.yaml restart app")

        return True

    except Exception as e:
        print(f"Error promoting model: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_versions(model_name: str):
    """List all versions of a model with their aliases."""
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        print(f"\nAvailable versions for model '{model_name}':")
        print("-" * 80)
        print(f"{'Version':<10} {'Run ID':<35} {'Aliases':<20} {'Status':<15}")
        print("-" * 80)

        for mv in sorted(versions, key=lambda x: int(x.version), reverse=True):
            aliases = ", ".join(mv.aliases) if hasattr(mv, 'aliases') and mv.aliases else "None"
            print(f"{mv.version:<10} {mv.run_id:<35} {aliases:<20} {mv.status:<15}")

        return True
    except Exception as e:
        print(f"Error listing versions: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Promote a model version in MLflow using aliases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all versions
  python scripts/promote_model.py --model-name credit-fraud --list

  # Promote version 5 to production
  python scripts/promote_model.py --model-name credit-fraud --version 5 --alias production

  # Promote and reload the app
  python scripts/promote_model.py --model-name credit-fraud --version 5 --alias production --reload-app

  # Promote with custom app URL
  python scripts/promote_model.py --model-name credit-fraud --version 5 --alias production --reload-app --app-url http://localhost:8000
        """
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("MODEL_NAME", "credit-fraud"),
        help="Name of the registered model (default: credit-fraud or MODEL_NAME env var)"
    )
    parser.add_argument(
        "--version",
        type=str,
        help="Model version number to promote"
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="production",
        help="Alias to set (default: production). Common aliases: production, challenger, champion"
    )
    parser.add_argument(
        "--reload-app",
        action="store_true",
        help="Trigger app reload after promotion via POST /reload endpoint"
    )
    parser.add_argument(
        "--app-url",
        type=str,
        help="URL of the app server (default: http://localhost:8000 or APP_URL env var)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all versions of the model with their aliases"
    )

    args = parser.parse_args()

    if args.list:
        success = list_versions(args.model_name)
        sys.exit(0 if success else 1)

    if not args.version:
        parser.error("--version is required (or use --list to see available versions)")

    success = promote_model(
        model_name=args.model_name,
        version=args.version,
        alias=args.alias,
        reload_app=args.reload_app,
        app_url=args.app_url
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
