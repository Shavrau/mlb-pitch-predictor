#!/usr/bin/env python3
"""
Unity Catalog Setup Automation Script for Baseball ML Pitch Predictor.

Idempotently verifies and creates:
1. Catalog (with graceful fallback if user lacks metastore-level CREATE CATALOG privileges)
2. Schema (e.g. baseball_ml / baseball_ml_dev)
3. Managed Volume (e.g. pitch_data)

Usage:
    python scripts/setup_unity_catalog.py --profile dev --catalog main --schema baseball_ml --volume pitch_data
"""

import argparse
import json
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("setup_unity_catalog")


def run_databricks_cli(args: list[str], profile: str | None = None) -> subprocess.CompletedProcess:
    """Execute a Databricks CLI command and return the result."""
    cmd = ["databricks"] + args
    if profile:
        cmd.extend(["--profile", profile])
    logger.debug("Executing CLI command: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def ensure_catalog(catalog_name: str, profile: str | None = None) -> bool:
    """Ensure catalog exists in Unity Catalog."""
    logger.info("Checking catalog '%s'...", catalog_name)
    res = run_databricks_cli(["catalogs", "list"], profile)
    if res.returncode != 0:
        logger.error("Failed to list catalogs: %s", res.stderr)
        return False

    catalogs = [line.split()[0] for line in res.stdout.strip().split("\n") if line.strip() and not line.startswith("Name")]
    if catalog_name in catalogs:
        logger.info("Catalog '%s' already exists.", catalog_name)
        return True

    logger.info("Catalog '%s' not found. Attempting to create it...", catalog_name)
    create_payload = json.dumps({"name": catalog_name})
    res_create = run_databricks_cli(["catalogs", "create", "--json", create_payload], profile)
    if res_create.returncode == 0:
        logger.info("Catalog '%s' created successfully.", catalog_name)
        return True
    else:
        logger.warning(
            "Could not create catalog '%s' (user may not have metastore admin privileges). Detail: %s",
            catalog_name,
            res_create.stderr.strip(),
        )
        return False


def ensure_schema(catalog_name: str, schema_name: str, profile: str | None = None) -> bool:
    """Ensure schema exists in Unity Catalog."""
    logger.info("Checking schema '%s.%s'...", catalog_name, schema_name)
    res = run_databricks_cli(["schemas", "list", catalog_name], profile)
    if res.returncode != 0:
        logger.error("Failed to list schemas in catalog '%s': %s", catalog_name, res.stderr)
        return False

    # Check if schema exists in output
    schema_full = f"{catalog_name}.{schema_name}"
    existing_schemas = [line.split()[0] for line in res.stdout.strip().split("\n") if line.strip() and not line.startswith("Full Name")]

    if schema_full in existing_schemas or schema_name in existing_schemas:
        logger.info("Schema '%s.%s' already exists.", catalog_name, schema_name)
        return True

    logger.info("Schema '%s.%s' not found. Creating...", catalog_name, schema_name)
    create_payload = json.dumps({
        "name": schema_name,
        "catalog_name": catalog_name,
        "comment": "Schema dedicated to MLB Statcast pitch predictor pipelines and ML models",
    })
    res_create = run_databricks_cli(["schemas", "create", "--json", create_payload], profile)
    if res_create.returncode == 0:
        logger.info("Schema '%s.%s' created successfully.", catalog_name, schema_name)
        return True
    else:
        logger.error("Failed to create schema '%s.%s': %s", catalog_name, schema_name, res_create.stderr)
        return False


def ensure_volume(catalog_name: str, schema_name: str, volume_name: str, profile: str | None = None) -> bool:
    """Ensure managed volume exists in Unity Catalog."""
    logger.info("Checking volume '%s.%s.%s'...", catalog_name, schema_name, volume_name)
    res = run_databricks_cli(["volumes", "list", catalog_name, schema_name], profile)
    if res.returncode != 0:
        logger.error("Failed to list volumes in '%s.%s': %s", catalog_name, schema_name, res.stderr)
        return False

    if volume_name in res.stdout:
        logger.info("Volume '%s.%s.%s' already exists.", catalog_name, schema_name, volume_name)
        return True

    logger.info("Volume '%s.%s.%s' not found. Creating managed volume...", catalog_name, schema_name, volume_name)
    create_payload = json.dumps({
        "name": volume_name,
        "catalog_name": catalog_name,
        "schema_name": schema_name,
        "volume_type": "MANAGED",
        "comment": "Managed volume for staging raw Statcast Parquet batches",
    })
    res_create = run_databricks_cli(["volumes", "create", "--json", create_payload], profile)
    if res_create.returncode == 0:
        logger.info("Managed volume '%s.%s.%s' created successfully.", catalog_name, schema_name, volume_name)
        return True
    else:
        logger.error("Failed to create volume '%s.%s.%s': %s", catalog_name, schema_name, volume_name, res_create.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Idempotent Unity Catalog environment setup.")
    parser.add_argument("--profile", default="dev", help="Databricks CLI profile name (default: dev)")
    parser.add_argument("--catalog", default="main", help="Target catalog name (default: main)")
    parser.add_argument("--schema", default="baseball_ml", help="Target schema name (default: baseball_ml)")
    parser.add_argument("--volume", default="pitch_data", help="Target volume name (default: pitch_data)")

    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("Setting up Unity Catalog environment for Pitch Predictor")
    logger.info("Profile: %s | Catalog: %s | Schema: %s | Volume: %s", args.profile, args.catalog, args.schema, args.volume)
    logger.info("==================================================")

    # 1. Catalog
    ensure_catalog(args.catalog, args.profile)

    # 2. Schema
    if not ensure_schema(args.catalog, args.schema, args.profile):
        sys.exit(1)

    # 3. Volume
    if not ensure_volume(args.catalog, args.schema, args.volume, args.profile):
        sys.exit(1)

    logger.info("Unity Catalog environment successfully configured!")
    logger.info("UC Volume landing path: /Volumes/%s/%s/%s/statcast_landing", args.catalog, args.schema, args.volume)


if __name__ == "__main__":
    main()
