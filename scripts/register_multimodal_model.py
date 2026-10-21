#!/usr/bin/env python3
"""
CLI Tool to Register the Multimodal Pitch Predictor in Databricks Unity Catalog.

Packages the trained MultimodalPitchClassifier into an MLflow custom PyFunc
(`MultimodalServingPyfuncModel`), registers the model into Unity Catalog
(`pitch_predictor.baseball_ml_dev.multimodal_pitch_predictor`), assigns the `@prod` alias,
and validates local/remote scoring signatures.

Usage:
    python scripts/register_multimodal_model.py --profile dev
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for terminal printing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from pitch_predictor.serving_model import (
    MultimodalServingPyfuncModel,
    create_pyfunc_input_example,
    get_pyfunc_model_signature,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register Multimodal Pitch Predictor into Databricks Unity Catalog."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/multimodal_pitch_model.joblib"),
        help="Path to serialized MultimodalPitchClassifier joblib file (default: models/multimodal_pitch_model.joblib)",
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default="pitch_predictor",
        help="Unity Catalog catalog name (default: pitch_predictor)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="baseball_ml_dev",
        help="Unity Catalog schema name (default: baseball_ml_dev)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="multimodal_pitch_predictor",
        help="Registered model name inside Unity Catalog (default: multimodal_pitch_predictor)",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="dev",
        help="Databricks CLI authentication profile (default: dev)",
    )
    parser.add_argument(
        "--alias",
        type=str,
        default="prod",
        help="Model alias to assign in Unity Catalog (default: prod)",
    )
    return parser.parse_args()


def register_multimodal_model(
    model_path: Path,
    catalog: str,
    schema: str,
    model_name: str,
    profile: str,
    alias: str = "prod",
) -> dict[str, str]:
    """
    Log and register MultimodalServingPyfuncModel into Databricks Unity Catalog.
    """
    os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run scripts/train_multimodal_model.py first."
        )

    # Resolve user email for experiment location
    try:
        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient(profile=profile)
        user_name = w.current_user.me().user_name
        exp_path = f"/Users/{user_name}/multimodal_pitch_predictor"
    except Exception:  # noqa: BLE001
        exp_path = f"/Shared/pitch_predictor/{schema}/multimodal_pitch_predictor"

    print(f"Setting MLflow experiment to: {exp_path}")
    mlflow.set_experiment(exp_path)

    full_uc_model_name = f"{catalog}.{schema}.{model_name}"
    print(f"Target Unity Catalog model: {full_uc_model_name}")

    signature = get_pyfunc_model_signature()
    input_example = create_pyfunc_input_example()

    # Requirements needed in the serving container
    pip_requirements = [
        "scikit-learn>=1.4.0,<1.6.0",
        "numpy>=1.26.0,<2.0.0",
        "pandas>=2.0.0",
        "joblib>=1.3.0",
        "mlflow>=2.10.0",
    ]

    # Package code path to include pitch_predictor module
    code_path = [str(PROJECT_ROOT / "src" / "pitch_predictor")]

    print("Logging PyFunc model with MLflow...")
    with mlflow.start_run(run_name="register_multimodal_serving_model") as run:
        mlflow.log_params(
            {
                "catalog": catalog,
                "schema": schema,
                "registered_model": full_uc_model_name,
                "framework": "scikit-learn",
                "architecture": "Multimodal (Statcast Context + Release Biomechanics)",
            }
        )

        mlflow.pyfunc.log_model(
            artifact_path="multimodal_pitch_predictor",
            python_model=MultimodalServingPyfuncModel(),
            artifacts={"model_path": str(model_path.resolve())},
            signature=signature,
            input_example=input_example,
            registered_model_name=full_uc_model_name,
            pip_requirements=pip_requirements,
            code_paths=code_path,
        )
        print(f"Model logged successfully to run: {run.info.run_id}")

    client = MlflowClient()

    # Determine registered model version
    model_versions = client.search_model_versions(f"name = '{full_uc_model_name}'")
    if not model_versions:
        raise RuntimeError(f"No versions found for registered model {full_uc_model_name}")

    # Sort versions descending by version number
    sorted_versions = sorted(model_versions, key=lambda v: int(v.version), reverse=True)
    latest_version = sorted_versions[0].version
    print(f"Latest registered version in Unity Catalog: Version {latest_version}")

    # Assign alias
    if alias:
        print(f"Assigning alias '@{alias}' to version {latest_version}...")
        client.set_registered_model_alias(full_uc_model_name, alias, latest_version)

    # Set model description and tags
    client.update_registered_model(
        name=full_uc_model_name,
        description=(
            "Multimodal Pitch Predictor fusing Statcast situational game context with "
            "MediaPipe 3D biomechanical pose features (arm slot, elbow flexion, release height, "
            "trunk tilt, stride). Produces fused predictions, probabilities, and tipping alignment diagnostics."
        ),
    )

    client.set_model_version_tag(
        name=full_uc_model_name,
        version=latest_version,
        key="pipeline_type",
        value="multimodal_fusion",
    )
    client.set_model_version_tag(
        name=full_uc_model_name,
        version=latest_version,
        key="input_modalities",
        value="statcast_tactical+vision_biomechanics",
    )

    print(f"Model successfully registered as {full_uc_model_name} (Version {latest_version}, Alias: @{alias})")

    # Verification scoring
    print("Verifying model scoring from Unity Catalog...")
    loaded_model = mlflow.pyfunc.load_model(f"models:/{full_uc_model_name}/{latest_version}")
    preds_df = loaded_model.predict(input_example)
    print(f"Verification output:\n{preds_df.to_string(index=False)}")

    return {
        "model_name": full_uc_model_name,
        "version": latest_version,
        "alias": alias,
        "run_id": run.info.run_id,
    }


def main() -> None:
    args = parse_args()
    print("=" * 70)
    print("DATABRICKS UNITY CATALOG MULTIMODAL MODEL REGISTRATION")
    print("=" * 70)
    print(f"Profile: {args.profile}")
    print(f"Catalog: {args.catalog}")
    print(f"Schema: {args.schema}")
    print(f"Model: {args.model_name}")
    print(f"Alias: {args.alias}")
    print("=" * 70)

    result = register_multimodal_model(
        model_path=args.model_path,
        catalog=args.catalog,
        schema=args.schema,
        model_name=args.model_name,
        profile=args.profile,
        alias=args.alias,
    )
    print("\n[SUCCESS] Registration completed successfully!")
    print(f"Registered Model: {result['model_name']}")
    print(f"Version: {result['version']}")
    print(f"Alias: @{result['alias']}")


if __name__ == "__main__":
    main()
