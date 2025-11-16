#!/usr/bin/env python3
"""
Promote a model version and restart the app container.

This script combines model promotion with Docker container restart for cases
where the /reload endpoint doesn't work or you prefer a full restart.

Usage:
    python scripts/promote_and_restart.py --version 5 --alias production
"""
import os
import sys
import argparse
import subprocess
import mlflow
from mlflow.tracking import MlflowClient


def run_command(cmd: list, description: str):
    """Run a shell command and return success status."""
    print(f"\n{description}...")
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def promote_and_restart(model_name: str, version: str, alias: str, compose_file: str):
    """
    Promote a model version and restart the app container.

    Args:
        model_name: Name of the registered model
        version: Version number to promote
        alias: Alias to set (e.g., 'production')
        compose_file: Path to docker-compose.yaml
    """
    # Configure MLflow
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_uri)

    client = MlflowClient()

    print("=" * 80)
    print("MODEL PROMOTION AND RESTART")
    print("=" * 80)
    print(f"MLflow URI: {mlflow_uri}")
    print(f"Model: {model_name}")
    print(f"Version: {version}")
    print(f"Alias: {alias}")
    print(f"Compose file: {compose_file}")
    print("=" * 80)

    try:
        # Step 1: Get the model version to verify it exists
        print("\n[1/3] Verifying model version exists...")
        mv = client.get_model_version(model_name, version)
        print(f"Found model version {version}")
        print(f"  Run ID: {mv.run_id}")
        print(f"  Status: {mv.status}")
        current_aliases = mv.aliases if hasattr(mv, 'aliases') else []
        print(f"  Current aliases: {current_aliases if current_aliases else 'None'}")

        # Step 2: Set the alias
        print(f"\n[2/3] Setting alias '{alias}' to version {version}...")
        client.set_registered_model_alias(model_name, alias, version)
        print(f"Successfully set alias '{alias}' to version {version}")

        # Verify the alias was set
        mv_updated = client.get_model_version(model_name, version)
        updated_aliases = mv_updated.aliases if hasattr(mv_updated, 'aliases') else []
        print(f"  Updated aliases: {updated_aliases}")

        # Step 3: Restart the app container
        print(f"\n[3/3] Restarting app container...")
        compose_cmd = ["docker-compose", "-f", compose_file, "restart", "app"]

        success = run_command(compose_cmd, "Restarting app container")

        if success:
            print("\n" + "=" * 80)
            print("SUCCESS!")
            print("=" * 80)
            print(f"Model version {version} is now aliased as '{alias}'")
            print("The app container has been restarted and will load the new model")
            print(f"\nCheck model info: curl http://localhost:8000/model_info")
            print(f"Test prediction: curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{{...}}'")
            return True
        else:
            print("\n" + "=" * 80)
            print("PARTIAL SUCCESS")
            print("=" * 80)
            print(f"Model promotion succeeded, but container restart failed")
            print("Please manually restart the container:")
            print(f"  docker-compose -f {compose_file} restart app")
            return False

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR")
        print("=" * 80)
        print(f"Failed to promote model: {e}")
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
        description="Promote a model version and restart the app container",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all versions
  python scripts/promote_and_restart.py --model-name credit-fraud --list

  # Promote version 5 to production and restart app
  python scripts/promote_and_restart.py --model-name credit-fraud --version 5 --alias production
        """
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default=os.getenv("MODEL_NAME", "credit-fraud"),
        help="Name of the registered model (default: credit-fraud)"
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
        help="Alias to set (default: production)"
    )
    parser.add_argument(
        "--compose-file",
        type=str,
        default="infra/docker-compose.yaml",
        help="Path to docker-compose.yaml (default: infra/docker-compose.yaml)"
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

    success = promote_and_restart(
        model_name=args.model_name,
        version=args.version,
        alias=args.alias,
        compose_file=args.compose_file
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
