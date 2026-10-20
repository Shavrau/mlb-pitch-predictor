"""Configuration management and Databricks profile resolution."""

import os
from pathlib import Path
from typing import Any

import yaml


def get_project_root() -> Path:
    """Return project root directory."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load configuration from config/config.yaml or given path."""
    if config_path:
        path = Path(config_path)
    else:
        path = get_project_root() / "config" / "config.yaml"

    if not path.exists():
        # Fallback default configuration
        return {
            "databricks": {
                "profile": os.getenv("DATABRICKS_CONFIG_PROFILE", "dev"),
                "catalog": "pitch_predictor",
                "schema": "baseball_ml",
            },
            "automl": {
                "target_col": "pitch_type",
                "timeout_minutes": 30,
            },
        }

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_workspace_client(profile: str | None = None):
    """
    Get a Databricks WorkspaceClient configured for a specific CLI profile.
    
    If running locally, uses the specified profile, or DATABRICKS_CONFIG_PROFILE env var,
    or falls back to config.yaml.
    If running on Databricks cluster, WorkspaceClient automatically detects cluster credentials.
    """
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        raise ImportError(
            "databricks-sdk is required to connect to Databricks. Run 'pip install databricks-sdk'."
        )

    resolved_profile = (
        profile
        or os.getenv("DATABRICKS_CONFIG_PROFILE")
        or load_config().get("databricks", {}).get("profile")
    )

    if resolved_profile:
        return WorkspaceClient(profile=resolved_profile)
    return WorkspaceClient()
